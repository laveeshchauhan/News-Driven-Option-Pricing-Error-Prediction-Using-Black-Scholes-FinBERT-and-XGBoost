"""
model.py — Phase 3: XGBoost ΔX Prediction Model.

Provides a thin, testable wrapper around XGBoost for training and evaluating
a regression model that predicts option mispricing

    ΔX = Actual Market Premium − Black-Scholes Theoretical Price

The model uses option-chain features (Greeks, moneyness, volatility, …) plus
the FinBERT sentiment score from Phase 2 as inputs.

All monetary values are in ₹ (INR).
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ML_FEATURE_COLUMNS,
    ML_MODEL_PATH,
    ML_RANDOM_STATE,
    ML_TARGET_COLUMN,
    ML_TEST_SIZE,
    XGBOOST_COLSAMPLE_BYTREE,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_MIN_CHILD_WEIGHT,
    XGBOOST_N_ESTIMATORS,
    XGBOOST_REG_ALPHA,
    XGBOOST_REG_LAMBDA,
    XGBOOST_SUBSAMPLE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived feature columns to *df* in-place and return it.

    New columns added
    -----------------
    moneyness       : underlying_value / strike_price  (S / K)
    option_type_enc : CE → 1, PE → 0  (integer)

    Missing sentiment columns (absent when Phase 2 was not run) are filled
    with 0.0 so that the model can still be trained on Phase 1-only data.
    """
    df = df.copy()

    # Moneyness — guard against zero / near-zero strike prices
    if "underlying_value" in df.columns and "strike_price" in df.columns:
        safe_strike = df["strike_price"].where(df["strike_price"].abs() > 1e-8, other=np.nan)
        df["moneyness"] = df["underlying_value"] / safe_strike
    else:
        df["moneyness"] = np.nan

    # Encode option type
    if "option_type" in df.columns:
        df["option_type_enc"] = (
            df["option_type"].str.upper().map({"CE": 1, "PE": 0}).fillna(0).astype(int)
        )
    else:
        df["option_type_enc"] = 0

    # Fill missing sentiment columns with neutral defaults
    sentiment_defaults: Dict[str, float] = {
        "daily_sentiment_score": 0.0,
        "daily_pos_mean": 0.0,
        "daily_neg_mean": 0.0,
        "daily_article_count": 0.0,
    }
    for col, default in sentiment_defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    return df


def select_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return a DataFrame containing only the available feature columns.

    Parameters
    ----------
    df              : DataFrame with engineered features.
    feature_columns : List of desired feature column names.
                      Defaults to ``config.ML_FEATURE_COLUMNS``.

    Returns
    -------
    (X, used_features) — the feature matrix and the list of columns used.
    """
    desired = feature_columns or ML_FEATURE_COLUMNS
    available = [c for c in desired if c in df.columns]
    missing = [c for c in desired if c not in df.columns]
    if missing:
        warnings.warn(
            f"Feature columns not found in DataFrame and will be skipped: {missing}",
            RuntimeWarning,
            stacklevel=2,
        )
    return df[available].copy(), available


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ──────────────────────────────────────────────────────────────────────────────


class DeltaXModel:
    """
    XGBoost regression model that predicts ΔX (option mispricing).

    Parameters
    ----------
    n_estimators        : Number of boosting rounds.
    max_depth           : Maximum tree depth.
    learning_rate       : Step size shrinkage.
    subsample           : Row sub-sampling ratio per tree.
    colsample_bytree    : Feature sub-sampling ratio per tree.
    min_child_weight    : Minimum sum of instance weight in a child.
    reg_alpha           : L1 regularisation term on leaf weights.
    reg_lambda          : L2 regularisation term on leaf weights.
    random_state        : Random seed for reproducibility.
    feature_columns     : Feature columns to use (defaults to config list).
    """

    def __init__(
        self,
        n_estimators: int = XGBOOST_N_ESTIMATORS,
        max_depth: int = XGBOOST_MAX_DEPTH,
        learning_rate: float = XGBOOST_LEARNING_RATE,
        subsample: float = XGBOOST_SUBSAMPLE,
        colsample_bytree: float = XGBOOST_COLSAMPLE_BYTREE,
        min_child_weight: int = XGBOOST_MIN_CHILD_WEIGHT,
        reg_alpha: float = XGBOOST_REG_ALPHA,
        reg_lambda: float = XGBOOST_REG_LAMBDA,
        random_state: int = ML_RANDOM_STATE,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        try:
            import xgboost as xgb  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for Phase 3. "
                "Install it with: pip install xgboost>=2.1.0"
            ) from exc

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.feature_columns: List[str] = list(feature_columns or ML_FEATURE_COLUMNS)

        self._model: Any = None          # xgb.XGBRegressor instance after fit
        self._used_features: List[str] = []

    # ── Training ────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
        verbose: bool = False,
    ) -> "DeltaXModel":
        """
        Fit the XGBoost model.

        Parameters
        ----------
        X_train  : Training feature matrix.
        y_train  : Training target (ΔX values).
        eval_set : Optional list of (X, y) pairs for early-stopping validation.
        verbose  : If True, print XGBoost training logs.

        Returns
        -------
        self — for method chaining.
        """
        import xgboost as xgb

        self._used_features = list(X_train.columns)

        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
            tree_method="hist",
            verbosity=1 if verbose else 0,
        )

        fit_kwargs: Dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = verbose

        self._model.fit(X_train, y_train, **fit_kwargs)
        return self

    # ── Inference ───────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ΔX for *X*.

        Parameters
        ----------
        X : Feature matrix (must contain the same columns as training data).

        Returns
        -------
        np.ndarray of predicted ΔX values (₹).
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self._model.predict(X[self._used_features])

    # ── Evaluation ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Compute regression metrics on the test set.

        Returns
        -------
        dict with keys: rmse, mae, r2, mean_error, std_error
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_pred = self.predict(X_test)
        errors = y_test.to_numpy() - y_pred

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mean_error": mean_err,
            "std_error": std_err,
        }

    # ── Feature importance ──────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances sorted descending.

        Columns: feature, importance (normalised gain).
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        scores = self._model.get_booster().get_fscore()
        total = sum(scores.values()) or 1.0
        rows = [
            {"feature": feat, "importance": score / total}
            for feat, score in sorted(scores.items(), key=lambda x: -x[1])
        ]
        # Include features with zero score
        zero_features = [f for f in self._used_features if f not in scores]
        for feat in zero_features:
            rows.append({"feature": feat, "importance": 0.0})

        return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str = ML_MODEL_PATH) -> None:
        """Persist the trained model to *path* (XGBoost native JSON format)."""
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._model.save_model(path)

        # Also save the feature list alongside the model
        meta_path = path.replace(".json", "_meta.json")
        with open(meta_path, "w") as fh:
            json.dump({"feature_columns": self._used_features}, fh, indent=2)

    def load(self, path: str = ML_MODEL_PATH) -> "DeltaXModel":
        """Load a persisted model from *path*."""
        import xgboost as xgb

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        self._model = xgb.XGBRegressor()
        self._model.load_model(path)

        # Restore feature list from metadata if available
        meta_path = path.replace(".json", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                meta = json.load(fh)
            self._used_features = meta.get("feature_columns", self.feature_columns)
        else:
            self._used_features = self.feature_columns

        return self


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: prepare data for train/test split
# ──────────────────────────────────────────────────────────────────────────────


def prepare_dataset(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_column: str = ML_TARGET_COLUMN,
    test_size: float = ML_TEST_SIZE,
    random_state: int = ML_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Engineer features, drop rows with missing targets, and split into
    train/test sets.

    Parameters
    ----------
    df              : Raw DataFrame (Phase 1 + optional Phase 2 output).
    feature_columns : Desired feature columns (defaults to config list).
    target_column   : Target column name (default ``'delta_x'``).
    test_size       : Fraction for test set (default 0.2).
    random_state    : Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test, used_features
    """
    from sklearn.model_selection import train_test_split

    # Engineer derived features
    df = engineer_features(df)

    # Drop rows where the target is missing
    df = df.dropna(subset=[target_column])
    if df.empty:
        raise ValueError(
            f"No valid rows remain after dropping NaN in target column '{target_column}'."
        )

    X, used_features = select_features(df, feature_columns)
    y = df[target_column].reset_index(drop=True)
    X = X.reset_index(drop=True)

    # Fill any remaining NaN in features with column median
    X = X.fillna(X.median(numeric_only=True))

    if len(X) < 4:
        raise ValueError(
            f"Not enough rows ({len(X)}) to split into train/test sets. "
            "Use at least 4 data points."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, used_features
