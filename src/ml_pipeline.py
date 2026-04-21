"""
ml_pipeline.py — Phase 3 Orchestrator: XGBoost ΔX Prediction Pipeline.

Steps performed by :func:`run_ml_pipeline`:
  1. Load enriched option chain (Phase 2 output or Phase 1 output as fallback)
  2. Engineer features (moneyness, option_type_enc, fill missing sentiment)
  3. Split into train / test sets (stratified by option type)
  4. Train XGBoost regression model to predict ΔX
  5. Evaluate on the held-out test set (RMSE, MAE, R²)
  6. Save the trained model to ``outputs/xgb_model.json``
  7. Export predictions CSV to ``outputs/predictions.csv``
  8. Print a formatted Phase 3 summary to terminal
  9. Optionally generate feature-importance and residual charts

All monetary values are in ₹ (INR).
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from config import (
    DEFAULT_OUTPUT_CSV,
    ML_FEATURE_COLUMNS,
    ML_INPUT_CSV,
    ML_MODEL_PATH,
    ML_PREDICTIONS_CSV,
    ML_RANDOM_STATE,
    ML_TARGET_COLUMN,
    ML_TEST_SIZE,
    OUTPUT_DIR,
    XGBOOST_COLSAMPLE_BYTREE,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_MIN_CHILD_WEIGHT,
    XGBOOST_N_ESTIMATORS,
    XGBOOST_REG_ALPHA,
    XGBOOST_REG_LAMBDA,
    XGBOOST_SUBSAMPLE,
)
from src.model import DeltaXModel, prepare_dataset


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _load_input(path: str, fallback_path: str, verbose: bool) -> pd.DataFrame:
    """
    Try to load *path*; if it does not exist fall back to *fallback_path*.
    Parses 'date' column as datetime when present.
    """
    for candidate in (path, fallback_path):
        if os.path.exists(candidate):
            if verbose:
                print(f"  Loading data from: {candidate}")
            parse_dates = ["date"] if "date" in pd.read_csv(candidate, nrows=0).columns else []
            return pd.read_csv(candidate, parse_dates=parse_dates)
    raise FileNotFoundError(
        f"Could not find input data at '{path}' or fallback '{fallback_path}'.\n"
        "Run Phase 1 (python main.py --demo) to generate the required CSV first."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline function
# ──────────────────────────────────────────────────────────────────────────────


def run_ml_pipeline(
    input_csv: str = ML_INPUT_CSV,
    fallback_input_csv: str = DEFAULT_OUTPUT_CSV,
    model_path: str = ML_MODEL_PATH,
    predictions_csv: str = ML_PREDICTIONS_CSV,
    feature_columns: Optional[List[str]] = None,
    target_column: str = ML_TARGET_COLUMN,
    test_size: float = ML_TEST_SIZE,
    n_estimators: int = XGBOOST_N_ESTIMATORS,
    max_depth: int = XGBOOST_MAX_DEPTH,
    learning_rate: float = XGBOOST_LEARNING_RATE,
    subsample: float = XGBOOST_SUBSAMPLE,
    colsample_bytree: float = XGBOOST_COLSAMPLE_BYTREE,
    min_child_weight: int = XGBOOST_MIN_CHILD_WEIGHT,
    reg_alpha: float = XGBOOST_REG_ALPHA,
    reg_lambda: float = XGBOOST_REG_LAMBDA,
    random_state: int = ML_RANDOM_STATE,
    plot: bool = False,
    verbose: bool = True,
) -> DeltaXModel:
    """
    Run the full Phase 3 XGBoost training pipeline.

    Parameters
    ----------
    input_csv       : Path to the enriched input CSV (Phase 2 output preferred;
                      falls back to *fallback_input_csv*).
    fallback_input_csv : Fallback CSV path if *input_csv* is not found
                      (default: ``outputs/results.csv``).
    model_path      : Where to save the trained model JSON.
    predictions_csv : Where to save the test-set predictions CSV.
    feature_columns : Features to use (defaults to ``config.ML_FEATURE_COLUMNS``).
    target_column   : Target column (default ``'delta_x'``).
    test_size       : Fraction held out for the test set (default 0.2).
    n_estimators    : XGBoost boosting rounds.
    max_depth       : Maximum tree depth.
    learning_rate   : Learning rate (eta).
    subsample       : Row sub-sampling ratio.
    colsample_bytree: Feature sub-sampling ratio.
    min_child_weight: Minimum child weight.
    reg_alpha       : L1 regularisation.
    reg_lambda      : L2 regularisation.
    random_state    : Random seed.
    plot            : Generate and save charts.
    verbose         : Print progress to stdout.

    Returns
    -------
    Trained :class:`DeltaXModel` instance.
    """
    # ── Step 1: Load data ──────────────────────────────────────────────────
    if verbose:
        print("\n[Phase 3 — Step 1/8] Loading input data...")

    df = _load_input(input_csv, fallback_input_csv, verbose)

    if verbose:
        print(f"  Rows loaded: {len(df)}")
        has_sentiment = "daily_sentiment_score" in df.columns
        print(f"  Sentiment features present: {has_sentiment}")

    # ── Step 2: Prepare dataset ────────────────────────────────────────────
    if verbose:
        print("[Phase 3 — Step 2/8] Engineering features and splitting data...")

    X_train, X_test, y_train, y_test, used_features = prepare_dataset(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    if verbose:
        print(f"  Features used    : {len(used_features)}")
        print(f"  Train samples    : {len(X_train)}")
        print(f"  Test  samples    : {len(X_test)}")
        print(f"  Target range     : [{y_train.min():.2f}, {y_train.max():.2f}] ₹")

    # ── Step 3: Train model ────────────────────────────────────────────────
    if verbose:
        print(
            f"[Phase 3 — Step 3/8] Training XGBoost model "
            f"({n_estimators} estimators, depth={max_depth}, lr={learning_rate})..."
        )

    model = DeltaXModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        feature_columns=used_features,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    if verbose:
        print("  Training complete.")

    # ── Step 4: Evaluate ───────────────────────────────────────────────────
    if verbose:
        print("[Phase 3 — Step 4/8] Evaluating on test set...")

    metrics = model.evaluate(X_test, y_test)

    if verbose:
        _print_metrics(metrics)

    # ── Step 5: Feature importance ─────────────────────────────────────────
    if verbose:
        print("[Phase 3 — Step 5/8] Computing feature importance...")

    importance_df = model.feature_importance()
    if verbose:
        print("\n  Top-10 most important features:")
        top10 = importance_df.head(10)
        for _, row in top10.iterrows():
            bar = "█" * int(row["importance"] * 40)
            print(f"    {row['feature']:<30s} {bar}  ({row['importance']:.4f})")
        print()

    # ── Step 6: Save model ─────────────────────────────────────────────────
    if verbose:
        print(f"[Phase 3 — Step 6/8] Saving model to: {model_path}")

    model.save(model_path)
    if verbose:
        print(f"  Model saved → {model_path}")

    # ── Step 7: Export predictions CSV ────────────────────────────────────
    if verbose:
        print(f"[Phase 3 — Step 7/8] Exporting predictions to: {predictions_csv}")

    y_pred = model.predict(X_test)
    pred_df = X_test.copy()
    pred_df[target_column] = y_test.values
    pred_df["predicted_delta_x"] = y_pred
    pred_df["residual"] = pred_df[target_column] - pred_df["predicted_delta_x"]

    _ensure_dir(predictions_csv)
    pred_df.to_csv(predictions_csv, index=False)
    if verbose:
        print(f"  Saved {len(pred_df)} rows → {predictions_csv}")

    # ── Step 8: Plots ──────────────────────────────────────────────────────
    if plot:
        if verbose:
            print("[Phase 3 — Step 8/8] Generating Phase 3 charts...")
        _generate_plots(
            pred_df=pred_df,
            importance_df=importance_df,
            metrics=metrics,
            output_dir=OUTPUT_DIR,
            verbose=verbose,
        )
    elif verbose:
        print("[Phase 3 — Step 8/8] Plots skipped (use --plot to enable).")

    # Final banner
    if verbose:
        _print_summary(metrics, used_features, len(X_train), len(X_test), model_path)

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Summary / metric printing
# ──────────────────────────────────────────────────────────────────────────────


def _print_metrics(metrics: dict) -> None:
    print(f"  RMSE             : ₹{metrics['rmse']:>10.4f}")
    print(f"  MAE              : ₹{metrics['mae']:>10.4f}")
    print(f"  R²               : {metrics['r2']:>12.4f}")
    print(f"  Mean Error       : ₹{metrics['mean_error']:>10.4f}")
    print(f"  Std  Error       : ₹{metrics['std_error']:>10.4f}")


def _print_summary(
    metrics: dict,
    used_features: list,
    n_train: int,
    n_test: int,
    model_path: str,
) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  PHASE 3 — XGBOOST ΔX PREDICTION RESULTS SUMMARY")
    print("  All prices in ₹ (INR)  |  RELIANCE  |  NSE India")
    print(sep)
    print(f"  Training samples                 : {n_train:>8,}")
    print(f"  Test samples                     : {n_test:>8,}")
    print(f"  Features used                    : {len(used_features):>8,}")
    print(sep)
    print(f"  RMSE  (test)                     : ₹{metrics['rmse']:>10.4f}")
    print(f"  MAE   (test)                     : ₹{metrics['mae']:>10.4f}")
    print(f"  R²    (test)                     : {metrics['r2']:>12.4f}")
    print(f"  Mean prediction error            : ₹{metrics['mean_error']:>10.4f}")
    print(f"  Std  prediction error            : ₹{metrics['std_error']:>10.4f}")
    print(sep)
    print(f"  Model saved to: {model_path}")
    print(sep)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────


def _generate_plots(
    pred_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    metrics: dict,
    output_dir: str,
    verbose: bool = True,
) -> None:
    """Generate and save Phase 3 visualisation charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        warnings.warn(
            "matplotlib / seaborn not installed. Skipping Phase 3 plots.",
            RuntimeWarning,
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ── Plot 1: Actual ΔX vs Predicted ΔX ───────────────────────────────
    if "delta_x" in pred_df.columns and "predicted_delta_x" in pred_df.columns:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(
            pred_df["delta_x"],
            pred_df["predicted_delta_x"],
            alpha=0.55,
            color="steelblue",
            s=25,
            label="Test samples",
        )
        # Perfect-prediction diagonal
        mn = min(pred_df["delta_x"].min(), pred_df["predicted_delta_x"].min())
        mx = max(pred_df["delta_x"].max(), pred_df["predicted_delta_x"].max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_title(
            f"Actual ΔX vs Predicted ΔX  |  R² = {metrics['r2']:.3f}  |  RMSE = ₹{metrics['rmse']:.2f}"
        )
        ax.set_xlabel("Actual ΔX (₹)")
        ax.set_ylabel("Predicted ΔX (₹)")
        ax.legend()
        plt.tight_layout()
        path1 = os.path.join(output_dir, "xgb_actual_vs_predicted.png")
        plt.savefig(path1, dpi=150, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"  Saved chart: {path1}")

    # ── Plot 2: Residual distribution ────────────────────────────────────
    if "residual" in pred_df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(pred_df["residual"], bins=30, kde=True, ax=ax, color="tomato", alpha=0.65)
        ax.axvline(0, color="black", linewidth=1.3, linestyle="--", label="Residual = 0")
        ax.axvline(
            pred_df["residual"].mean(),
            color="darkred",
            linewidth=1.3,
            linestyle="-",
            label=f"Mean = {pred_df['residual'].mean():.2f}",
        )
        ax.set_title("XGBoost Prediction Residuals  (Actual ΔX − Predicted ΔX)")
        ax.set_xlabel("Residual (₹)")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        path2 = os.path.join(output_dir, "xgb_residuals.png")
        plt.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"  Saved chart: {path2}")

    # ── Plot 3: Feature importance bar chart ─────────────────────────────
    if not importance_df.empty:
        top_n = importance_df.head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=top_n,
            x="importance",
            y="feature",
            ax=ax,
            palette="Blues_d",
            orient="h",
        )
        ax.set_title("XGBoost Feature Importance (top 15 — normalised gain)")
        ax.set_xlabel("Importance (normalised)")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        path3 = os.path.join(output_dir, "xgb_feature_importance.png")
        plt.savefig(path3, dpi=150, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"  Saved chart: {path3}")
