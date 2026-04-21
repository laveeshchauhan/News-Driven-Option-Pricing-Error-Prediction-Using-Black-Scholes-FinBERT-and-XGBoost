"""
inference.py — Phase 4: Live Inference Engine.

Loads a trained Phase 3 XGBoost model (``outputs/xgb_model.json``) and
produces predicted ΔX values plus actionable trading signals for a given
option chain DataFrame — without retraining.

Signal logic
------------
  BUY  (long)  : predicted ΔX < −buy_threshold   → option is underpriced
  SELL (short) : predicted ΔX >  sell_threshold   → option is overpriced
  HOLD         : |predicted ΔX| ≤ threshold       → fairly priced

All monetary values are in ₹ (INR).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from config import (
    INFERENCE_SIGNAL_BUY_THRESHOLD,
    INFERENCE_SIGNAL_SELL_THRESHOLD,
    ML_MODEL_PATH,
)
from src.model import DeltaXModel, engineer_features, select_features


# ──────────────────────────────────────────────────────────────────────────────
# Signal generation
# ──────────────────────────────────────────────────────────────────────────────


def generate_signals(
    predicted_delta_x: np.ndarray,
    buy_threshold: float = INFERENCE_SIGNAL_BUY_THRESHOLD,
    sell_threshold: float = INFERENCE_SIGNAL_SELL_THRESHOLD,
) -> List[str]:
    """
    Convert predicted ΔX values into actionable trading signals.

    Parameters
    ----------
    predicted_delta_x : Array of predicted ΔX values (₹).
    buy_threshold     : Abs ΔX below which we signal BUY (option underpriced).
    sell_threshold    : ΔX above which we signal SELL (option overpriced).

    Returns
    -------
    List of signal strings: 'BUY', 'SELL', or 'HOLD'.
    """
    signals: List[str] = []
    for dx in predicted_delta_x:
        if dx < -abs(buy_threshold):
            signals.append("BUY")
        elif dx > abs(sell_threshold):
            signals.append("SELL")
        else:
            signals.append("HOLD")
    return signals


# ──────────────────────────────────────────────────────────────────────────────
# Core inference function
# ──────────────────────────────────────────────────────────────────────────────


def run_inference(
    df: pd.DataFrame,
    model_path: str = ML_MODEL_PATH,
    feature_columns: Optional[List[str]] = None,
    buy_threshold: float = INFERENCE_SIGNAL_BUY_THRESHOLD,
    sell_threshold: float = INFERENCE_SIGNAL_SELL_THRESHOLD,
) -> pd.DataFrame:
    """
    Run live inference on *df* using the saved Phase 3 model.

    Parameters
    ----------
    df              : DataFrame with Phase 1 outputs (bs_price, greeks, …).
                      Optional Phase 2 sentiment columns are used when present.
    model_path      : Path to the saved XGBoost model JSON.
    feature_columns : Feature columns to use.  Defaults to the feature list
                      stored alongside the model at training time.
    buy_threshold   : Abs predicted ΔX below which we signal BUY.
    sell_threshold  : Predicted ΔX above which we signal SELL.

    Returns
    -------
    Copy of *df* with three extra columns:

    ``predicted_delta_x``
        Model-predicted ΔX in ₹.
    ``signal``
        'BUY', 'SELL', or 'HOLD'.
    ``confidence``
        |predicted_delta_x| normalised to [0, 1] relative to the row with
        the largest absolute predicted ΔX in this batch.

    Raises
    ------
    FileNotFoundError
        If *model_path* does not exist.
    ValueError
        If *df* is empty or contains no usable feature columns.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty — cannot run inference.")

    # Load model
    model = DeltaXModel()
    model.load(model_path)

    # Engineer derived features (moneyness, option_type_enc, sentiment fill)
    enriched = engineer_features(df)

    # Use model's training feature list if not explicitly specified
    cols = feature_columns or model._used_features
    X, used_cols = select_features(enriched, cols)

    if X.empty or len(used_cols) == 0:
        raise ValueError(
            "No usable feature columns found in the input DataFrame. "
            f"Expected at least one of: {cols}"
        )

    # Fill remaining NaN in features with column median
    X = X.fillna(X.median(numeric_only=True))

    # Predict ΔX
    predictions = model.predict(X)

    # Trading signals
    signals = generate_signals(predictions, buy_threshold, sell_threshold)

    # Confidence: |ΔX| normalised by the batch maximum (stays in [0, 1])
    abs_pred = np.abs(predictions)
    batch_max = float(abs_pred.max()) if abs_pred.size > 0 and abs_pred.max() > 0 else 1.0
    confidence = (abs_pred / batch_max).round(4)

    result = df.copy()
    result["predicted_delta_x"] = np.round(predictions, 4)
    result["signal"] = signals
    result["confidence"] = confidence

    return result
