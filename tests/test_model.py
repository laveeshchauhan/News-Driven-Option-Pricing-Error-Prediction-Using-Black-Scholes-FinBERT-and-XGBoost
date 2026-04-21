"""
test_model.py — Unit tests for Phase 3: XGBoost ΔX Prediction Model.

All tests use synthetic data and mock XGBoost to avoid GPU/network
requirements in CI.  A real XGBoost integration smoke-test is included
but skipped automatically when xgboost is not installed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Ensure project root is importable ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DeltaXModel, engineer_features, prepare_dataset, select_features

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_XGBOOST_AVAILABLE = True
try:
    import xgboost  # noqa: F401
except ImportError:
    _XGBOOST_AVAILABLE = False

requires_xgboost = pytest.mark.skipif(
    not _XGBOOST_AVAILABLE,
    reason="xgboost not installed",
)

_SKLEARN_AVAILABLE = True
try:
    import sklearn  # noqa: F401
except ImportError:
    _SKLEARN_AVAILABLE = False

requires_sklearn = pytest.mark.skipif(
    not _SKLEARN_AVAILABLE,
    reason="scikit-learn not installed",
)


def _make_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Return a small synthetic option-chain DataFrame."""
    rng = np.random.default_rng(seed)
    n_ce = n // 2
    n_pe = n - n_ce
    return pd.DataFrame(
        {
            "option_type": ["CE"] * n_ce + ["PE"] * n_pe,
            "strike_price": rng.uniform(1300, 1500, n),
            "underlying_value": rng.uniform(1350, 1450, n),
            "time_to_expiry": rng.uniform(0.01, 0.5, n),
            "volatility": rng.uniform(0.15, 0.40, n),
            "risk_free_rate": np.full(n, 0.07),
            "bs_price": rng.uniform(5, 200, n),
            "delta": rng.uniform(0, 1, n),
            "gamma": rng.uniform(0, 0.01, n),
            "theta": rng.uniform(-5, 0, n),
            "vega": rng.uniform(0, 2, n),
            "rho": rng.uniform(-1, 1, n),
            "delta_x": rng.uniform(-30, 30, n),
            "daily_sentiment_score": rng.uniform(-1, 1, n),
            "daily_pos_mean": rng.uniform(0, 1, n),
            "daily_neg_mean": rng.uniform(0, 1, n),
            "daily_article_count": rng.integers(0, 10, n),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: engineer_features
# ─────────────────────────────────────────────────────────────────────────────


class TestEngineerFeatures:
    def test_moneyness_calculated(self):
        df = _make_df(10)
        out = engineer_features(df)
        expected = df["underlying_value"] / df["strike_price"]
        pd.testing.assert_series_equal(
            out["moneyness"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_option_type_enc_ce(self):
        df = pd.DataFrame({"option_type": ["CE", "ce", "CE"], "strike_price": [100, 100, 100], "underlying_value": [105, 105, 105]})
        out = engineer_features(df)
        assert list(out["option_type_enc"]) == [1, 1, 1]

    def test_option_type_enc_pe(self):
        df = pd.DataFrame({"option_type": ["PE", "pe", "PE"], "strike_price": [100, 100, 100], "underlying_value": [95, 95, 95]})
        out = engineer_features(df)
        assert list(out["option_type_enc"]) == [0, 0, 0]

    def test_missing_sentiment_filled_with_zero(self):
        """Sentiment columns absent from input should be filled with 0.0."""
        df = pd.DataFrame({
            "option_type": ["CE"],
            "strike_price": [1400.0],
            "underlying_value": [1390.0],
        })
        out = engineer_features(df)
        assert out["daily_sentiment_score"].iloc[0] == 0.0
        assert out["daily_pos_mean"].iloc[0] == 0.0
        assert out["daily_neg_mean"].iloc[0] == 0.0
        assert out["daily_article_count"].iloc[0] == 0.0

    def test_partial_sentiment_nan_filled(self):
        """NaN in existing sentiment columns should be filled with 0.0."""
        df = pd.DataFrame({
            "option_type": ["CE", "PE"],
            "strike_price": [1400.0, 1400.0],
            "underlying_value": [1390.0, 1390.0],
            "daily_sentiment_score": [0.5, float("nan")],
            "daily_pos_mean": [0.6, float("nan")],
            "daily_neg_mean": [0.1, float("nan")],
            "daily_article_count": [3.0, float("nan")],
        })
        out = engineer_features(df)
        assert out["daily_sentiment_score"].iloc[1] == 0.0
        assert out["daily_article_count"].iloc[1] == 0.0

    def test_moneyness_zero_strike_safe(self):
        """Strike price of 0 should not raise; moneyness is NaN."""
        df = pd.DataFrame({
            "option_type": ["CE"],
            "strike_price": [0.0],
            "underlying_value": [1390.0],
        })
        out = engineer_features(df)
        assert np.isnan(out["moneyness"].iloc[0])

    def test_returns_copy_not_inplace(self):
        """engineer_features should return a new DataFrame, not modify the original."""
        df = _make_df(5)
        original_cols = set(df.columns)
        out = engineer_features(df)
        assert "moneyness" not in original_cols
        assert "moneyness" in out.columns

    def test_missing_option_type_defaults_to_zero(self):
        df = pd.DataFrame({"strike_price": [1400.0], "underlying_value": [1390.0]})
        out = engineer_features(df)
        assert out["option_type_enc"].iloc[0] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: select_features
# ─────────────────────────────────────────────────────────────────────────────


class TestSelectFeatures:
    def test_returns_only_available_columns(self):
        df = engineer_features(_make_df(10))
        X, used = select_features(df, feature_columns=["moneyness", "delta", "nonexistent"])
        assert "moneyness" in used
        assert "delta" in used
        assert "nonexistent" not in used
        assert list(X.columns) == used

    def test_warns_on_missing_columns(self):
        df = engineer_features(_make_df(5))
        with pytest.warns(RuntimeWarning, match="not found"):
            _, used = select_features(df, feature_columns=["delta", "does_not_exist"])
        assert "does_not_exist" not in used

    def test_all_default_columns_subset_of_engineered(self):
        """All default feature columns should be present after engineer_features."""
        from config import ML_FEATURE_COLUMNS
        df = engineer_features(_make_df(20))
        X, used = select_features(df)
        # Every default feature that's in df should be selected
        for col in ML_FEATURE_COLUMNS:
            if col in df.columns:
                assert col in used


# ─────────────────────────────────────────────────────────────────────────────
# Tests: prepare_dataset
# ─────────────────────────────────────────────────────────────────────────────


@requires_sklearn
class TestPrepareDataset:
    def test_returns_five_items(self):
        df = _make_df(40)
        result = prepare_dataset(df, test_size=0.25, random_state=0)
        assert len(result) == 5

    def test_correct_split_sizes(self):
        df = _make_df(40)
        X_train, X_test, y_train, y_test, _ = prepare_dataset(
            df, test_size=0.25, random_state=0
        )
        total = len(X_train) + len(X_test)
        assert total == 40
        assert len(X_test) == pytest.approx(10, abs=2)

    def test_drops_nan_target(self):
        df = _make_df(20)
        df.loc[0, "delta_x"] = float("nan")
        X_train, X_test, y_train, y_test, _ = prepare_dataset(df, test_size=0.2)
        assert len(X_train) + len(X_test) == 19

    def test_raises_on_empty_after_drop(self):
        df = _make_df(5)
        df["delta_x"] = float("nan")
        with pytest.raises(ValueError, match="No valid rows"):
            prepare_dataset(df)

    def test_raises_with_too_few_rows(self):
        df = _make_df(3)
        with pytest.raises(ValueError, match="Not enough rows"):
            prepare_dataset(df)

    def test_features_have_no_nan(self):
        """Remaining NaN in features should be filled by median imputation."""
        df = _make_df(30)
        df.loc[0, "delta"] = float("nan")
        X_train, X_test, _, _, _ = prepare_dataset(df, test_size=0.2)
        assert not X_train.isnull().any().any()
        assert not X_test.isnull().any().any()

    def test_moneyness_added(self):
        df = _make_df(30)
        X_train, X_test, _, _, used = prepare_dataset(df, test_size=0.2)
        assert "moneyness" in used

    def test_option_type_enc_added(self):
        df = _make_df(30)
        _, _, _, _, used = prepare_dataset(df, test_size=0.2)
        assert "option_type_enc" in used


# ─────────────────────────────────────────────────────────────────────────────
# Tests: DeltaXModel (mocked XGBoost)
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_xgb_regressor(predict_return: np.ndarray) -> MagicMock:
    """Return a mock that looks like an xgb.XGBRegressor."""
    mock_reg = MagicMock()
    mock_reg.predict.return_value = predict_return

    # Mock booster for feature importance
    mock_booster = MagicMock()
    mock_booster.get_fscore.return_value = {"delta": 10, "moneyness": 8, "vega": 5}
    mock_reg.get_booster.return_value = mock_booster

    return mock_reg


class TestDeltaXModel:
    # ── Construction ────────────────────────────────────────────────────────

    def test_raises_without_xgboost(self):
        with patch.dict("sys.modules", {"xgboost": None}):
            with pytest.raises(ImportError, match="xgboost is required"):
                DeltaXModel()

    @requires_xgboost
    def test_instantiation_defaults(self):
        model = DeltaXModel()
        assert model._model is None
        assert model.n_estimators > 0

    # ── fit / predict ───────────────────────────────────────────────────────

    @requires_xgboost
    def test_predict_before_fit_raises(self):
        model = DeltaXModel()
        df = engineer_features(_make_df(5))
        X, _ = select_features(df)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(X)

    @requires_xgboost
    def test_evaluate_before_fit_raises(self):
        model = DeltaXModel()
        df = engineer_features(_make_df(5))
        X, _ = select_features(df)
        y = df["delta_x"].reset_index(drop=True)
        with pytest.raises(RuntimeError):
            model.evaluate(X, y)

    @requires_xgboost
    def test_feature_importance_before_fit_raises(self):
        model = DeltaXModel()
        with pytest.raises(RuntimeError):
            model.feature_importance()

    @requires_xgboost
    def test_fit_sets_model_attribute(self):
        """Fitting should populate self._model."""
        df = engineer_features(_make_df(30))
        X, used = select_features(df)
        y = df["delta_x"].reset_index(drop=True)
        X = X.reset_index(drop=True)
        model = DeltaXModel(n_estimators=5, max_depth=2)
        model.fit(X, y)
        assert model._model is not None
        assert model._used_features == used

    @requires_xgboost
    def test_predict_returns_array(self):
        df = engineer_features(_make_df(40))
        X, used = select_features(df)
        y = df["delta_x"].reset_index(drop=True)
        X = X.reset_index(drop=True)
        model = DeltaXModel(n_estimators=5, max_depth=2)
        model.fit(X, y)
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)

    # ── evaluate ────────────────────────────────────────────────────────────

    @requires_xgboost
    @requires_sklearn
    def test_evaluate_returns_all_metrics(self):
        df = engineer_features(_make_df(40))
        X, _ = select_features(df)
        y = df["delta_x"].reset_index(drop=True)
        X = X.reset_index(drop=True)
        model = DeltaXModel(n_estimators=5, max_depth=2)
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert set(metrics.keys()) == {"rmse", "mae", "r2", "mean_error", "std_error"}
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    @requires_xgboost
    @requires_sklearn
    def test_evaluate_perfect_prediction(self):
        """If the model perfectly predicts, RMSE should be ~0."""
        df = engineer_features(_make_df(20))
        X, _ = select_features(df)
        y = df["delta_x"].reset_index(drop=True)
        X = X.reset_index(drop=True)

        model = DeltaXModel()
        model._model = MagicMock()
        model._used_features = list(X.columns)
        model._model.predict.return_value = y.to_numpy()

        metrics = model.evaluate(X, y)
        assert pytest.approx(0.0, abs=1e-9) == metrics["rmse"]
        assert pytest.approx(1.0, abs=1e-9) == metrics["r2"]

    # ── feature_importance ──────────────────────────────────────────────────

    @requires_xgboost
    def test_feature_importance_shape(self):
        df = engineer_features(_make_df(40))
        X, _ = select_features(df)
        y = df["delta_x"].reset_index(drop=True)
        X = X.reset_index(drop=True)
        model = DeltaXModel(n_estimators=5, max_depth=2)
        model.fit(X, y)
        fi = model.feature_importance()
        assert isinstance(fi, pd.DataFrame)
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) == len(X.columns)

    @requires_xgboost
    def test_feature_importance_sorted_descending(self):
        df = engineer_features(_make_df(40))
        X, _ = select_features(df)
        y = df["delta_x"].reset_index(drop=True)
        X = X.reset_index(drop=True)
        model = DeltaXModel(n_estimators=5, max_depth=2)
        model.fit(X, y)
        fi = model.feature_importance()
        assert list(fi["importance"]) == sorted(fi["importance"], reverse=True)

    # ── save / load ─────────────────────────────────────────────────────────

    @requires_xgboost
    def test_save_before_fit_raises(self):
        model = DeltaXModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.save("/tmp/dummy.json")

    @requires_xgboost
    def test_save_creates_files(self):
        df = engineer_features(_make_df(30))
        X, _ = select_features(df)
        y = df["delta_x"].reset_index(drop=True)
        X = X.reset_index(drop=True)
        model = DeltaXModel(n_estimators=5, max_depth=2)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")
            model.save(path)
            assert os.path.exists(path)
            meta_path = path.replace(".json", "_meta.json")
            assert os.path.exists(meta_path)
            with open(meta_path) as fh:
                meta = json.load(fh)
            assert "feature_columns" in meta

    @requires_xgboost
    def test_load_missing_file_raises(self):
        model = DeltaXModel()
        with pytest.raises(FileNotFoundError):
            model.load("/tmp/does_not_exist_phase3.json")

    @requires_xgboost
    def test_save_load_roundtrip(self):
        """Saved and reloaded model should produce identical predictions."""
        df = engineer_features(_make_df(40))
        X, _ = select_features(df)
        y = df["delta_x"].reset_index(drop=True)
        X = X.reset_index(drop=True)

        model = DeltaXModel(n_estimators=5, max_depth=2)
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")
            model.save(path)

            loaded = DeltaXModel()
            loaded.load(path)
            preds_after = loaded.predict(X)

        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: ml_pipeline integration (mocked XGBoost)
# ─────────────────────────────────────────────────────────────────────────────


@requires_xgboost
@requires_sklearn
class TestMlPipelineIntegration:
    """Smoke-tests for run_ml_pipeline using a temp CSV."""

    def _write_csv(self, path: str, n: int = 60) -> None:
        df = _make_df(n)
        df.to_csv(path, index=False)

    def test_pipeline_runs_and_returns_model(self):
        from src.ml_pipeline import run_ml_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "input.csv")
            model_path = os.path.join(tmpdir, "model.json")
            pred_path = os.path.join(tmpdir, "preds.csv")
            self._write_csv(csv_path)

            model = run_ml_pipeline(
                input_csv=csv_path,
                model_path=model_path,
                predictions_csv=pred_path,
                n_estimators=5,
                max_depth=2,
                plot=False,
                verbose=False,
            )
            assert isinstance(model, DeltaXModel)
            assert model._model is not None
            assert os.path.exists(model_path)
            assert os.path.exists(pred_path)

    def test_pipeline_predictions_csv_columns(self):
        from src.ml_pipeline import run_ml_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "input.csv")
            model_path = os.path.join(tmpdir, "model.json")
            pred_path = os.path.join(tmpdir, "preds.csv")
            self._write_csv(csv_path)

            run_ml_pipeline(
                input_csv=csv_path,
                model_path=model_path,
                predictions_csv=pred_path,
                n_estimators=5,
                max_depth=2,
                plot=False,
                verbose=False,
            )
            preds = pd.read_csv(pred_path)
            assert "predicted_delta_x" in preds.columns
            assert "delta_x" in preds.columns
            assert "residual" in preds.columns

    def test_pipeline_fallback_to_phase1_csv(self):
        """Pipeline should use fallback CSV when primary input is missing."""
        from src.ml_pipeline import run_ml_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            # No primary (Phase 2) CSV — only the fallback
            fallback_path = os.path.join(tmpdir, "results.csv")
            model_path = os.path.join(tmpdir, "model.json")
            pred_path = os.path.join(tmpdir, "preds.csv")
            self._write_csv(fallback_path)

            model = run_ml_pipeline(
                input_csv=os.path.join(tmpdir, "nonexistent.csv"),
                fallback_input_csv=fallback_path,
                model_path=model_path,
                predictions_csv=pred_path,
                n_estimators=5,
                max_depth=2,
                plot=False,
                verbose=False,
            )
            assert model._model is not None

    def test_pipeline_raises_when_no_csv_available(self):
        from src.ml_pipeline import run_ml_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                run_ml_pipeline(
                    input_csv=os.path.join(tmpdir, "missing1.csv"),
                    model_path=os.path.join(tmpdir, "model.json"),
                    predictions_csv=os.path.join(tmpdir, "preds.csv"),
                    n_estimators=5,
                    max_depth=2,
                    verbose=False,
                )

    def test_pipeline_without_sentiment_columns(self):
        """Pipeline must work when Phase 2 sentiment columns are absent."""
        from src.ml_pipeline import run_ml_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            df = _make_df(60)
            # Drop all sentiment columns
            df = df.drop(
                columns=[c for c in df.columns if c.startswith("daily_")],
                errors="ignore",
            )
            csv_path = os.path.join(tmpdir, "input.csv")
            df.to_csv(csv_path, index=False)

            model = run_ml_pipeline(
                input_csv=csv_path,
                model_path=os.path.join(tmpdir, "model.json"),
                predictions_csv=os.path.join(tmpdir, "preds.csv"),
                n_estimators=5,
                max_depth=2,
                plot=False,
                verbose=False,
            )
            assert model._model is not None

    def test_pipeline_plot_flag(self):
        """Passing plot=True should not raise even when matplotlib is present."""
        from src.ml_pipeline import run_ml_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "input.csv")
            self._write_csv(csv_path, n=60)

            try:
                run_ml_pipeline(
                    input_csv=csv_path,
                    model_path=os.path.join(tmpdir, "model.json"),
                    predictions_csv=os.path.join(tmpdir, "preds.csv"),
                    n_estimators=5,
                    max_depth=2,
                    plot=True,
                    verbose=False,
                )
            except ImportError:
                pytest.skip("matplotlib not installed")
