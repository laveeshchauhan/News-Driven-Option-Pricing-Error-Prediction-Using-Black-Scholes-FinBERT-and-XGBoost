"""
test_inference.py — Unit tests for Phase 4: Live Inference Engine.

All tests use synthetic data and a mocked DeltaXModel to avoid
XGBoost/network requirements in CI.  A lightweight integration smoke-test
is included and auto-skipped when xgboost is not installed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Ensure project root is importable ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import generate_signals, run_inference
from src.live_pipeline import _merge_sentiment, _rank_signals

# ─────────────────────────────────────────────────────────────────────────────
# Availability guards
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

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Return a minimal synthetic option-chain DataFrame with Phase 1 outputs."""
    rng = np.random.default_rng(seed)
    n_ce = n // 2
    n_pe = n - n_ce
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "option_type": ["CE"] * n_ce + ["PE"] * n_pe,
            "strike_price": rng.uniform(1200, 1600, n).round(0),
            "underlying_value": rng.uniform(1350, 1430, n).round(2),
            "time_to_expiry": rng.uniform(0.01, 0.5, n).round(4),
            "volatility": rng.uniform(0.15, 0.40, n).round(4),
            "risk_free_rate": np.full(n, 0.07),
            "bs_price": rng.uniform(5, 120, n).round(2),
            "actual_premium": rng.uniform(3, 130, n).round(2),
            "delta_x": rng.uniform(-20, 20, n).round(4),
            "delta": rng.uniform(-1, 1, n).round(4),
            "gamma": rng.uniform(0, 0.005, n).round(6),
            "theta": rng.uniform(-5, 0, n).round(4),
            "vega": rng.uniform(0, 2, n).round(4),
            "rho": rng.uniform(-1, 1, n).round(4),
        }
    )


def _mock_model(predictions: np.ndarray) -> MagicMock:
    """Return a MagicMock that behaves like a loaded DeltaXModel."""
    m = MagicMock()
    m._used_features = [
        "strike_price", "underlying_value", "time_to_expiry",
        "volatility", "risk_free_rate", "bs_price",
        "delta", "gamma", "theta", "vega", "rho",
        "moneyness", "option_type_enc",
        "daily_sentiment_score", "daily_pos_mean",
        "daily_neg_mean", "daily_article_count",
    ]
    m.predict.return_value = predictions
    return m


# ═════════════════════════════════════════════════════════════════════════════
# generate_signals
# ═════════════════════════════════════════════════════════════════════════════


class TestGenerateSignals:

    def test_buy_signal(self):
        signals = generate_signals(np.array([-5.0]), buy_threshold=2.0, sell_threshold=2.0)
        assert signals == ["BUY"]

    def test_sell_signal(self):
        signals = generate_signals(np.array([5.0]), buy_threshold=2.0, sell_threshold=2.0)
        assert signals == ["SELL"]

    def test_hold_signal_positive(self):
        signals = generate_signals(np.array([1.0]), buy_threshold=2.0, sell_threshold=2.0)
        assert signals == ["HOLD"]

    def test_hold_signal_negative(self):
        signals = generate_signals(np.array([-1.5]), buy_threshold=2.0, sell_threshold=2.0)
        assert signals == ["HOLD"]

    def test_hold_at_exact_threshold(self):
        # Exactly at threshold is NOT beyond it → HOLD
        signals = generate_signals(np.array([2.0, -2.0]), buy_threshold=2.0, sell_threshold=2.0)
        assert signals == ["HOLD", "HOLD"]

    def test_mixed_signals(self):
        arr = np.array([-10.0, 0.5, 8.0, -1.0, 3.5])
        signals = generate_signals(arr, buy_threshold=2.0, sell_threshold=2.0)
        assert signals == ["BUY", "HOLD", "SELL", "HOLD", "SELL"]

    def test_empty_array(self):
        signals = generate_signals(np.array([]), buy_threshold=2.0, sell_threshold=2.0)
        assert signals == []

    def test_zero_is_hold(self):
        signals = generate_signals(np.array([0.0]), buy_threshold=2.0, sell_threshold=2.0)
        assert signals == ["HOLD"]

    def test_custom_asymmetric_thresholds(self):
        arr = np.array([-3.0, 4.0])
        signals = generate_signals(arr, buy_threshold=5.0, sell_threshold=3.0)
        assert signals == ["HOLD", "SELL"]

    def test_returns_list(self):
        result = generate_signals(np.array([1.0, -3.0]))
        assert isinstance(result, list)

    def test_length_matches_input(self):
        arr = np.arange(-5.0, 6.0, 1.0)
        assert len(generate_signals(arr)) == len(arr)


# ═════════════════════════════════════════════════════════════════════════════
# run_inference
# ═════════════════════════════════════════════════════════════════════════════


class TestRunInference:

    def _do_inference(self, df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        """Helper: run run_inference with a mocked DeltaXModel."""
        mock = _mock_model(predictions)
        with (
            patch("src.inference.DeltaXModel") as MockClass,
        ):
            instance = MockClass.return_value
            instance._used_features = mock._used_features
            instance.load.return_value = None
            instance.predict.return_value = predictions

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump({}, f)
                tmp_path = f.name

            try:
                result = run_inference(df, model_path=tmp_path)
            finally:
                os.unlink(tmp_path)
                meta = tmp_path.replace(".json", "_meta.json")
                if os.path.exists(meta):
                    os.unlink(meta)

        return result

    def test_output_columns_added(self):
        df = _make_df(10)
        preds = np.zeros(10)
        result = self._do_inference(df, preds)
        assert "predicted_delta_x" in result.columns
        assert "signal" in result.columns
        assert "confidence" in result.columns

    def test_row_count_preserved(self):
        df = _make_df(15)
        preds = np.linspace(-10, 10, 15)
        result = self._do_inference(df, preds)
        assert len(result) == 15

    def test_original_columns_preserved(self):
        df = _make_df(8)
        preds = np.ones(8)
        result = self._do_inference(df, preds)
        for col in df.columns:
            assert col in result.columns

    def test_confidence_range(self):
        df = _make_df(12)
        preds = np.linspace(-10, 10, 12)
        result = self._do_inference(df, preds)
        assert result["confidence"].between(0.0, 1.0).all()

    def test_confidence_max_is_one(self):
        df = _make_df(6)
        preds = np.array([-10.0, -5.0, 0.0, 0.0, 5.0, 10.0])
        result = self._do_inference(df, preds)
        assert pytest.approx(result["confidence"].max(), abs=1e-3) == 1.0

    def test_all_zero_predictions_confidence_is_zero(self):
        df = _make_df(5)
        preds = np.zeros(5)
        result = self._do_inference(df, preds)
        assert (result["confidence"] == 0.0).all()

    def test_signal_values_valid(self):
        df = _make_df(20)
        preds = np.linspace(-15, 15, 20)
        result = self._do_inference(df, preds)
        assert set(result["signal"]).issubset({"BUY", "SELL", "HOLD"})

    def test_empty_df_raises(self):
        df = pd.DataFrame()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="empty"):
                run_inference(df, model_path=tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_missing_model_raises(self):
        df = _make_df(5)
        with pytest.raises(FileNotFoundError):
            run_inference(df, model_path="/nonexistent/model.json")

    def test_predicted_delta_x_rounded(self):
        df = _make_df(4)
        preds = np.array([-3.141592, 0.0, 2.71828, 9.999999])
        result = self._do_inference(df, preds)
        for val in result["predicted_delta_x"]:
            # max 4 decimal places
            assert round(val, 4) == val


# ═════════════════════════════════════════════════════════════════════════════
# _merge_sentiment
# ═════════════════════════════════════════════════════════════════════════════


class TestMergeSentiment:

    def _make_sentiment_csv(self, tmp_dir: str) -> str:
        path = os.path.join(tmp_dir, "sent.csv")
        sent = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=5, freq="D"),
                "daily_sentiment_score": [0.1, -0.2, 0.3, 0.0, 0.5],
                "daily_pos_mean": [0.6, 0.3, 0.7, 0.5, 0.8],
                "daily_neg_mean": [0.1, 0.4, 0.1, 0.3, 0.1],
                "daily_article_count": [3, 2, 5, 1, 4],
            }
        )
        sent.to_csv(path, index=False)
        return path

    def test_merge_adds_sentiment_columns(self):
        df = _make_df(5)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_sentiment_csv(tmp)
            result = _merge_sentiment(df, sentiment_csv=path, verbose=False)
        assert "daily_sentiment_score" in result.columns

    def test_missing_csv_returns_original(self):
        df = _make_df(5)
        result = _merge_sentiment(df, sentiment_csv="/nonexistent/file.csv", verbose=False)
        assert list(result.columns) == list(df.columns)

    def test_sentiment_csv_without_date_returns_original(self):
        df = _make_df(5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "no_date.csv")
            pd.DataFrame({"x": [1, 2, 3]}).to_csv(path, index=False)
            result = _merge_sentiment(df, sentiment_csv=path, verbose=False)
        assert "daily_sentiment_score" not in result.columns

    def test_df_without_date_column_returns_original(self):
        df = _make_df(5).drop(columns=["date"])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_sentiment_csv(tmp)
            result = _merge_sentiment(df, sentiment_csv=path, verbose=False)
        assert "daily_sentiment_score" not in result.columns

    def test_no_sentiment_csv_argument_uses_default(self):
        df = _make_df(5)
        # Default path does not exist in test env → should return df unchanged
        result = _merge_sentiment(df, sentiment_csv=None, verbose=False)
        assert list(result.columns) == list(df.columns)


# ═════════════════════════════════════════════════════════════════════════════
# _rank_signals
# ═════════════════════════════════════════════════════════════════════════════


class TestRankSignals:

    def _make_signals_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "strike_price": [1300, 1350, 1400, 1250, 1450, 1500],
                "signal": ["HOLD", "BUY", "SELL", "BUY", "HOLD", "SELL"],
                "predicted_delta_x": [0.5, -8.0, 12.0, -3.0, -1.0, 5.0],
                "confidence": [0.04, 0.67, 1.00, 0.25, 0.08, 0.42],
            }
        )

    def test_buy_comes_before_sell(self):
        result = _rank_signals(self._make_signals_df())
        first_sell = result[result["signal"] == "SELL"].index[0]
        last_buy = result[result["signal"] == "BUY"].index[-1]
        assert last_buy < first_sell

    def test_sell_comes_before_hold(self):
        result = _rank_signals(self._make_signals_df())
        first_hold = result[result["signal"] == "HOLD"].index[0]
        last_sell = result[result["signal"] == "SELL"].index[-1]
        assert last_sell < first_hold

    def test_within_buy_group_largest_abs_first(self):
        result = _rank_signals(self._make_signals_df())
        buys = result[result["signal"] == "BUY"]["predicted_delta_x"].abs().tolist()
        assert buys == sorted(buys, reverse=True)

    def test_within_sell_group_largest_abs_first(self):
        result = _rank_signals(self._make_signals_df())
        sells = result[result["signal"] == "SELL"]["predicted_delta_x"].abs().tolist()
        assert sells == sorted(sells, reverse=True)

    def test_row_count_preserved(self):
        df = self._make_signals_df()
        result = _rank_signals(df)
        assert len(result) == len(df)

    def test_index_reset(self):
        result = _rank_signals(self._make_signals_df())
        assert list(result.index) == list(range(len(result)))

    def test_no_side_effects_on_input(self):
        df = self._make_signals_df()
        original_signals = df["signal"].tolist()
        _rank_signals(df)
        assert df["signal"].tolist() == original_signals  # input unchanged

    def test_all_hold(self):
        df = pd.DataFrame(
            {"signal": ["HOLD"] * 4, "predicted_delta_x": [0.1, 0.3, 0.2, 0.4]}
        )
        result = _rank_signals(df)
        # Largest |ΔX| first within HOLD group
        assert result["predicted_delta_x"].abs().tolist() == sorted(
            [0.1, 0.3, 0.2, 0.4], reverse=True
        )

    def test_all_buy(self):
        df = pd.DataFrame(
            {"signal": ["BUY"] * 3, "predicted_delta_x": [-5.0, -10.0, -2.0]}
        )
        result = _rank_signals(df)
        assert result["predicted_delta_x"].tolist() == [-10.0, -5.0, -2.0]

    def test_temp_columns_not_in_output(self):
        result = _rank_signals(self._make_signals_df())
        assert "_signal_rank" not in result.columns
        assert "_abs_pred_dx" not in result.columns


# ═════════════════════════════════════════════════════════════════════════════
# Integration smoke-test (requires xgboost + sklearn)
# ═════════════════════════════════════════════════════════════════════════════


@requires_xgboost
class TestRunInferenceIntegration:
    """End-to-end smoke-test: train a tiny model, save it, run inference."""

    def test_full_inference_roundtrip(self, tmp_path):
        from sklearn.model_selection import train_test_split

        from src.model import DeltaXModel, prepare_dataset

        df = _make_df(40, seed=7)

        X_train, X_test, y_train, y_test, used = prepare_dataset(
            df, test_size=0.25, random_state=1
        )
        model = DeltaXModel(
            n_estimators=10, max_depth=2, random_state=1, feature_columns=used
        )
        model.fit(X_train, y_train)
        model_path = str(tmp_path / "test_model.json")
        model.save(model_path)

        result = run_inference(
            df, model_path=model_path, buy_threshold=2.0, sell_threshold=2.0
        )

        assert "predicted_delta_x" in result.columns
        assert "signal" in result.columns
        assert "confidence" in result.columns
        assert len(result) == len(df)
        assert set(result["signal"]).issubset({"BUY", "SELL", "HOLD"})
        assert result["confidence"].between(0.0, 1.0).all()

    def test_inference_signals_respect_thresholds(self, tmp_path):
        from src.model import DeltaXModel, prepare_dataset

        df = _make_df(40, seed=3)
        X_train, X_test, y_train, y_test, used = prepare_dataset(
            df, test_size=0.25, random_state=2
        )
        model = DeltaXModel(n_estimators=5, max_depth=2, random_state=2, feature_columns=used)
        model.fit(X_train, y_train)
        model_path = str(tmp_path / "test_model2.json")
        model.save(model_path)

        buy_th, sell_th = 1.0, 1.0
        result = run_inference(df, model_path=model_path, buy_threshold=buy_th, sell_threshold=sell_th)

        buy_rows = result[result["signal"] == "BUY"]
        sell_rows = result[result["signal"] == "SELL"]
        hold_rows = result[result["signal"] == "HOLD"]

        if not buy_rows.empty:
            assert (buy_rows["predicted_delta_x"] < -buy_th).all()
        if not sell_rows.empty:
            assert (sell_rows["predicted_delta_x"] > sell_th).all()
        if not hold_rows.empty:
            assert (hold_rows["predicted_delta_x"].abs() <= max(buy_th, sell_th)).all()
