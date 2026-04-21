"""
test_monitor.py — Unit tests for Phase 5: MLOps & Monitoring.

Covers:
  - log_run / load_run_log
  - save_baseline / load_baseline
  - compute_psi
  - detect_data_drift
  - monitor_predictions
  - check_alerts
  - generate_report
  - _json_safe helper
  - monitoring_pipeline helpers
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.monitor import (
    _json_safe,
    check_alerts,
    compute_psi,
    detect_data_drift,
    generate_report,
    load_baseline,
    load_run_log,
    log_run,
    monitor_predictions,
    save_baseline,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a temporary directory for file I/O tests."""
    return tmp_path


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """30-row synthetic option chain DataFrame with all required columns."""
    rng = np.random.default_rng(0)
    n = 30
    strikes = rng.integers(1300, 1500, n).astype(float)
    spot = 1390.20
    bs_price = rng.uniform(5, 150, n)
    actual = bs_price + rng.normal(0, 10, n)
    delta_x = actual - bs_price
    predicted = delta_x + rng.normal(0, 2, n)
    signals = ["BUY" if p < -2 else ("SELL" if p > 2 else "HOLD") for p in predicted]
    return pd.DataFrame(
        {
            "strike_price": strikes,
            "underlying_value": spot,
            "time_to_expiry": rng.uniform(0.01, 0.3, n),
            "volatility": rng.uniform(0.15, 0.35, n),
            "risk_free_rate": 0.07,
            "bs_price": bs_price,
            "actual_premium": actual,
            "delta_x": delta_x,
            "predicted_delta_x": predicted,
            "signal": signals,
            "confidence": rng.uniform(0, 1, n),
            "delta": rng.uniform(0.1, 0.9, n),
            "gamma": rng.uniform(0.001, 0.01, n),
            "theta": rng.uniform(-0.5, -0.05, n),
            "vega": rng.uniform(0.5, 3.0, n),
            "rho": rng.uniform(-1.0, 1.0, n),
            "option_type": rng.choice(["CE", "PE"], n),
            "moneyness": spot / strikes,
            "option_type_enc": rng.integers(0, 2, n).astype(int),
            "daily_sentiment_score": rng.uniform(-0.3, 0.3, n),
            "daily_pos_mean": rng.uniform(0.1, 0.6, n),
            "daily_neg_mean": rng.uniform(0.1, 0.4, n),
            "daily_article_count": rng.integers(1, 20, n).astype(float),
        }
    )


@pytest.fixture()
def baseline_dict(sample_df, tmp_dir) -> Dict[str, Any]:
    """Pre-built baseline saved in tmp_dir."""
    path = str(tmp_dir / "baseline.json")
    return save_baseline(sample_df, baseline_path=path)


# ──────────────────────────────────────────────────────────────────────────────
# log_run / load_run_log
# ──────────────────────────────────────────────────────────────────────────────


class TestLogRun:
    def test_creates_file(self, tmp_dir):
        path = str(tmp_dir / "run_log.jsonl")
        log_run("phase3", {"rmse": 1.23}, log_path=path)
        assert os.path.exists(path)

    def test_record_structure(self, tmp_dir):
        path = str(tmp_dir / "run_log.jsonl")
        record = log_run("phase3", {"rmse": 1.23, "mae": 0.9}, params={"lr": 0.05}, log_path=path)
        assert record["phase"] == "phase3"
        assert record["metrics"]["rmse"] == 1.23
        assert record["params"]["lr"] == 0.05
        assert "timestamp" in record

    def test_appends_multiple_records(self, tmp_dir):
        path = str(tmp_dir / "run_log.jsonl")
        log_run("phase3", {"rmse": 1.0}, log_path=path)
        log_run("phase4", {"n_buy": 5}, log_path=path)
        records = load_run_log(path)
        assert len(records) == 2
        assert records[0]["phase"] == "phase3"
        assert records[1]["phase"] == "phase4"

    def test_tags_stored(self, tmp_dir):
        path = str(tmp_dir / "run_log.jsonl")
        record = log_run("phase5", {}, tags={"mode": "demo"}, log_path=path)
        assert record["tags"]["mode"] == "demo"

    def test_empty_metrics(self, tmp_dir):
        path = str(tmp_dir / "run_log.jsonl")
        record = log_run("phase5", {}, log_path=path)
        assert record["metrics"] == {}

    def test_load_empty_file_returns_empty_list(self, tmp_dir):
        path = str(tmp_dir / "nonexistent.jsonl")
        records = load_run_log(path)
        assert records == []

    def test_numpy_scalar_serialised(self, tmp_dir):
        path = str(tmp_dir / "run_log.jsonl")
        log_run("phase3", {"rmse": np.float64(3.14)}, log_path=path)
        records = load_run_log(path)
        assert isinstance(records[0]["metrics"]["rmse"], float)

    def test_nan_value_serialised_as_none(self, tmp_dir):
        path = str(tmp_dir / "run_log.jsonl")
        log_run("phase3", {"rmse": float("nan")}, log_path=path)
        records = load_run_log(path)
        assert records[0]["metrics"]["rmse"] is None


# ──────────────────────────────────────────────────────────────────────────────
# save_baseline / load_baseline
# ──────────────────────────────────────────────────────────────────────────────


class TestBaseline:
    def test_save_creates_file(self, sample_df, tmp_dir):
        path = str(tmp_dir / "baseline.json")
        save_baseline(sample_df, baseline_path=path)
        assert os.path.exists(path)

    def test_baseline_has_features(self, sample_df, tmp_dir):
        path = str(tmp_dir / "baseline.json")
        b = save_baseline(sample_df, feature_columns=["strike_price", "volatility"], baseline_path=path)
        assert "strike_price" in b["features"]
        assert "volatility" in b["features"]

    def test_feature_stats_keys(self, sample_df, tmp_dir):
        path = str(tmp_dir / "baseline.json")
        b = save_baseline(sample_df, feature_columns=["strike_price"], baseline_path=path)
        feat = b["features"]["strike_price"]
        for key in ("count", "mean", "std", "min", "p25", "p50", "p75", "max", "histogram"):
            assert key in feat

    def test_histogram_shape(self, sample_df, tmp_dir):
        path = str(tmp_dir / "baseline.json")
        b = save_baseline(sample_df, feature_columns=["volatility"], baseline_path=path)
        hist = b["features"]["volatility"]["histogram"]
        assert len(hist["counts"]) == 20
        assert len(hist["edges"]) == 21

    def test_load_baseline_roundtrip(self, sample_df, tmp_dir):
        path = str(tmp_dir / "baseline.json")
        original = save_baseline(sample_df, feature_columns=["strike_price"], baseline_path=path)
        loaded = load_baseline(path)
        assert loaded["features"]["strike_price"]["mean"] == pytest.approx(
            original["features"]["strike_price"]["mean"], rel=1e-6
        )

    def test_load_baseline_missing_file_raises(self, tmp_dir):
        path = str(tmp_dir / "missing.json")
        with pytest.raises(FileNotFoundError):
            load_baseline(path)

    def test_n_rows_stored(self, sample_df, tmp_dir):
        path = str(tmp_dir / "baseline.json")
        b = save_baseline(sample_df, baseline_path=path)
        assert b["n_rows"] == len(sample_df)

    def test_skips_columns_not_in_df(self, sample_df, tmp_dir):
        path = str(tmp_dir / "baseline.json")
        b = save_baseline(
            sample_df,
            feature_columns=["strike_price", "nonexistent_column"],
            baseline_path=path,
        )
        assert "nonexistent_column" not in b["features"]
        assert "strike_price" in b["features"]


# ──────────────────────────────────────────────────────────────────────────────
# compute_psi
# ──────────────────────────────────────────────────────────────────────────────


class TestComputePSI:
    def _make_histogram(self, values: np.ndarray, bins: int = 20):
        counts, edges = np.histogram(values, bins=bins)
        return counts.tolist(), edges.tolist()

    def test_identical_distribution_gives_near_zero(self):
        rng = np.random.default_rng(1)
        data = rng.uniform(0, 1, 500)
        counts, edges = self._make_histogram(data)
        psi = compute_psi(counts, edges, pd.Series(data))
        assert psi < 0.05  # nearly identical → near zero PSI

    def test_different_distribution_gives_high_psi(self):
        rng = np.random.default_rng(2)
        baseline_data = rng.uniform(0, 1, 500)
        new_data = rng.uniform(2, 3, 500)  # completely different range
        counts, edges = self._make_histogram(baseline_data)
        psi = compute_psi(counts, edges, pd.Series(new_data))
        assert psi > 0.2  # significant shift

    def test_empty_series_returns_zero(self):
        counts, edges = self._make_histogram(np.ones(50))
        psi = compute_psi(counts, edges, pd.Series([], dtype=float))
        assert psi == 0.0

    def test_returns_float(self):
        rng = np.random.default_rng(3)
        data = rng.uniform(0, 1, 100)
        counts, edges = self._make_histogram(data)
        psi = compute_psi(counts, edges, pd.Series(data))
        assert isinstance(psi, float)

    def test_psi_non_negative(self):
        rng = np.random.default_rng(4)
        baseline = rng.normal(0, 1, 300)
        new = rng.normal(0.5, 1, 300)
        counts, edges = self._make_histogram(baseline)
        psi = compute_psi(counts, edges, pd.Series(new))
        assert psi >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# detect_data_drift
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectDataDrift:
    def test_no_drift_on_identical_data(self, sample_df, baseline_dict):
        result = detect_data_drift(
            sample_df,
            baseline=baseline_dict,
            feature_columns=["strike_price", "volatility"],
        )
        assert result["overall_drift_status"] == "ok"

    def test_drift_detected_on_shifted_data(self, sample_df, baseline_dict):
        # Shift strike prices by +1000 to force obvious drift
        shifted = sample_df.copy()
        shifted["strike_price"] = shifted["strike_price"] + 1000.0
        result = detect_data_drift(
            shifted,
            baseline=baseline_dict,
            feature_columns=["strike_price"],
        )
        assert result["overall_drift_status"] in ("warning", "alert")
        assert "strike_price" in result["drifted_features"]

    def test_result_keys_present(self, sample_df, baseline_dict):
        result = detect_data_drift(
            sample_df,
            baseline=baseline_dict,
            feature_columns=["strike_price"],
        )
        for key in ("feature_drift", "overall_drift_status", "drifted_features",
                    "n_features_checked", "timestamp"):
            assert key in result

    def test_feature_drift_keys(self, sample_df, baseline_dict):
        result = detect_data_drift(
            sample_df,
            baseline=baseline_dict,
            feature_columns=["volatility"],
        )
        feat = result["feature_drift"]["volatility"]
        assert "psi" in feat
        assert "psi_status" in feat
        assert "ks_drifted" in feat

    def test_n_features_checked(self, sample_df, baseline_dict):
        result = detect_data_drift(
            sample_df,
            baseline=baseline_dict,
            feature_columns=["strike_price", "volatility"],
        )
        assert result["n_features_checked"] == 2

    def test_handles_column_not_in_new_df(self, sample_df, baseline_dict):
        df_no_vol = sample_df.drop(columns=["volatility"])
        result = detect_data_drift(
            df_no_vol,
            baseline=baseline_dict,
            feature_columns=["strike_price", "volatility"],
        )
        # volatility missing from new_df → checked only on available columns
        assert "volatility" not in result["feature_drift"]
        assert result["n_features_checked"] >= 1

    def test_psi_status_levels(self, sample_df, baseline_dict):
        # Large shift should yield psi_status = 'alert'
        shifted = sample_df.copy()
        shifted["strike_price"] = shifted["strike_price"] + 2000.0
        result = detect_data_drift(
            shifted,
            baseline=baseline_dict,
            feature_columns=["strike_price"],
        )
        assert result["feature_drift"]["strike_price"]["psi_status"] in ("warning", "alert")


# ──────────────────────────────────────────────────────────────────────────────
# monitor_predictions
# ──────────────────────────────────────────────────────────────────────────────


class TestMonitorPredictions:
    def test_basic_metrics_computed(self, sample_df):
        result = monitor_predictions(sample_df)
        assert result["n_samples"] == len(sample_df)
        assert "rmse" in result
        assert "mae" in result
        assert "r2" in result

    def test_rmse_is_positive(self, sample_df):
        result = monitor_predictions(sample_df)
        assert result["rmse"] >= 0.0

    def test_mae_le_rmse(self, sample_df):
        result = monitor_predictions(sample_df)
        assert result["mae"] <= result["rmse"] + 1e-6

    def test_pct_within_columns_present(self, sample_df):
        result = monitor_predictions(sample_df)
        assert "pct_within_5" in result
        assert "pct_within_10" in result

    def test_pct_within_10_ge_pct_within_5(self, sample_df):
        result = monitor_predictions(sample_df)
        assert result["pct_within_10"] >= result["pct_within_5"] - 1e-6

    def test_signal_accuracy_computed_when_signal_present(self, sample_df):
        result = monitor_predictions(sample_df)
        assert "signal_accuracy" in result
        assert 0.0 <= result["signal_accuracy"] <= 100.0

    def test_missing_columns_returns_error(self, sample_df):
        df_no_pred = sample_df.drop(columns=["predicted_delta_x"])
        result = monitor_predictions(df_no_pred)
        assert result["n_samples"] == 0
        assert "error" in result

    def test_perfect_predictions_give_rmse_zero(self):
        df = pd.DataFrame(
            {"delta_x": [1.0, 2.0, 3.0], "predicted_delta_x": [1.0, 2.0, 3.0]}
        )
        result = monitor_predictions(df)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-6)

    def test_fewer_than_2_rows_returns_error(self):
        df = pd.DataFrame({"delta_x": [1.0], "predicted_delta_x": [1.5]})
        result = monitor_predictions(df)
        assert "error" in result

    def test_max_abs_error_present(self, sample_df):
        result = monitor_predictions(sample_df)
        assert "max_abs_error" in result
        assert result["max_abs_error"] >= result["mae"]


# ──────────────────────────────────────────────────────────────────────────────
# check_alerts
# ──────────────────────────────────────────────────────────────────────────────


class TestCheckAlerts:
    def _make_drift_result(self, psi: float, ks_pvalue: float = 0.5):
        ks_drifted = ks_pvalue < 0.05
        psi_status = "ok" if psi < 0.1 else ("warning" if psi < 0.2 else "alert")
        return {
            "feature_drift": {
                "strike_price": {
                    "psi": psi,
                    "psi_status": psi_status,
                    "ks_statistic": 0.05,
                    "ks_pvalue": ks_pvalue,
                    "ks_drifted": ks_drifted,
                }
            },
            "overall_drift_status": psi_status,
            "drifted_features": ["strike_price"] if psi >= 0.1 else [],
        }

    def test_no_alerts_on_clean_data(self):
        drift = self._make_drift_result(psi=0.02)
        alerts = check_alerts(drift_results=drift)
        assert alerts == []

    def test_warning_alert_on_moderate_psi(self):
        drift = self._make_drift_result(psi=0.15)
        alerts = check_alerts(drift_results=drift)
        assert any(a["level"] == "warning" for a in alerts)

    def test_critical_alert_on_high_psi(self):
        drift = self._make_drift_result(psi=0.25)
        alerts = check_alerts(drift_results=drift)
        assert any(a["level"] == "critical" for a in alerts)

    def test_no_alerts_when_no_drift(self):
        alerts = check_alerts(drift_results=None, perf_metrics=None)
        assert alerts == []

    def test_critical_alert_on_rmse_degradation(self):
        perf = {"rmse": 30.0, "mae": 15.0, "n_samples": 20}
        baseline = {"rmse": 5.0}
        alerts = check_alerts(perf_metrics=perf, baseline_metrics=baseline)
        assert any(a["category"] == "model_performance" for a in alerts)
        assert any(a["level"] == "critical" for a in alerts)

    def test_no_perf_alert_when_rmse_acceptable(self):
        perf = {"rmse": 6.0, "n_samples": 20}
        baseline = {"rmse": 5.0}  # delta = 1 < threshold
        alerts = check_alerts(perf_metrics=perf, baseline_metrics=baseline)
        assert not any(a["category"] == "model_performance" for a in alerts)

    def test_alert_has_required_keys(self):
        drift = self._make_drift_result(psi=0.25)
        alerts = check_alerts(drift_results=drift)
        for a in alerts:
            assert "level" in a
            assert "category" in a
            assert "message" in a

    def test_ks_only_drift_generates_warning(self):
        # PSI is below warning threshold but KS p-value is very small
        drift = self._make_drift_result(psi=0.05, ks_pvalue=0.001)
        alerts = check_alerts(drift_results=drift)
        assert any(a["category"] == "data_drift" for a in alerts)


# ──────────────────────────────────────────────────────────────────────────────
# generate_report
# ──────────────────────────────────────────────────────────────────────────────


class TestGenerateReport:
    def test_report_created(self, tmp_dir, sample_df, baseline_dict):
        path = str(tmp_dir / "report.json")
        report = generate_report(report_path=path)
        assert os.path.exists(path)

    def test_report_structure(self, tmp_dir):
        path = str(tmp_dir / "report.json")
        report = generate_report(report_path=path)
        for key in ("report_generated_at", "overall_health", "summary", "alerts", "run_history"):
            assert key in report

    def test_health_ok_when_no_issues(self, tmp_dir):
        path = str(tmp_dir / "report.json")
        report = generate_report(alerts=[], report_path=path)
        assert report["overall_health"] in ("ok", "unknown")

    def test_health_critical_when_critical_alert(self, tmp_dir):
        path = str(tmp_dir / "report.json")
        alerts = [{"level": "critical", "category": "data_drift", "message": "test"}]
        report = generate_report(alerts=alerts, report_path=path)
        assert report["overall_health"] == "critical"

    def test_health_warning_when_warning_alert(self, tmp_dir):
        path = str(tmp_dir / "report.json")
        alerts = [{"level": "warning", "category": "data_drift", "message": "test"}]
        report = generate_report(alerts=alerts, report_path=path)
        assert report["overall_health"] == "warning"

    def test_run_history_populated(self, tmp_dir):
        log_path = str(tmp_dir / "run_log.jsonl")
        log_run("phase3", {"rmse": 2.0}, log_path=log_path)
        report_path = str(tmp_dir / "report.json")
        report = generate_report(run_log_path=log_path, report_path=report_path)
        assert len(report["run_history"]) == 1

    def test_report_is_json_serialisable(self, tmp_dir):
        path = str(tmp_dir / "report.json")
        report = generate_report(report_path=path)
        # Should not raise
        _ = json.dumps(report)

    def test_report_n_critical_alerts_in_summary(self, tmp_dir):
        path = str(tmp_dir / "report.json")
        alerts = [
            {"level": "critical", "category": "x", "message": "a"},
            {"level": "warning", "category": "y", "message": "b"},
        ]
        report = generate_report(alerts=alerts, report_path=path)
        assert report["summary"]["n_critical_alerts"] == 1
        assert report["summary"]["n_warning_alerts"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# _json_safe helper
# ──────────────────────────────────────────────────────────────────────────────


class TestJsonSafe:
    def test_int_passthrough(self):
        assert _json_safe(42) == 42

    def test_float_passthrough(self):
        assert _json_safe(3.14) == pytest.approx(3.14)

    def test_string_passthrough(self):
        assert _json_safe("hello") == "hello"

    def test_numpy_int(self):
        v = _json_safe(np.int64(7))
        assert isinstance(v, int)
        assert v == 7

    def test_numpy_float(self):
        v = _json_safe(np.float32(1.5))
        assert isinstance(v, float)
        assert v == pytest.approx(1.5, abs=1e-4)

    def test_nan_becomes_none(self):
        assert _json_safe(float("nan")) is None

    def test_inf_becomes_none(self):
        assert _json_safe(float("inf")) is None

    def test_numpy_nan_becomes_none(self):
        assert _json_safe(np.float64("nan")) is None

    def test_numpy_array_to_list(self):
        arr = np.array([1, 2, 3])
        result = _json_safe(arr)
        assert result == [1, 2, 3]


# ──────────────────────────────────────────────────────────────────────────────
# Monitoring pipeline integration smoke-test
# ──────────────────────────────────────────────────────────────────────────────


class TestMonitoringPipelineIntegration:
    def test_pipeline_runs_with_demo_data(self, tmp_dir):
        """End-to-end smoke test using fully synthetic data (no disk CSV needed)."""
        from src.monitoring_pipeline import run_monitoring_pipeline

        # Non-existent CSV → pipeline falls back to demo data
        report = run_monitoring_pipeline(
            predictions_csv=str(tmp_dir / "no_predictions.csv"),
            training_csv=str(tmp_dir / "no_training.csv"),
            baseline_path=str(tmp_dir / "baseline.json"),
            run_log_path=str(tmp_dir / "run_log.jsonl"),
            report_path=str(tmp_dir / "report.json"),
            verbose=False,
            plot=False,
        )
        assert report is not None
        assert "overall_health" in report

    def test_pipeline_creates_report_file(self, tmp_dir):
        from src.monitoring_pipeline import run_monitoring_pipeline

        report_path = str(tmp_dir / "report.json")
        run_monitoring_pipeline(
            predictions_csv=str(tmp_dir / "no_predictions.csv"),
            training_csv=str(tmp_dir / "no_training.csv"),
            baseline_path=str(tmp_dir / "baseline.json"),
            run_log_path=str(tmp_dir / "run_log.jsonl"),
            report_path=report_path,
            verbose=False,
            plot=False,
        )
        assert os.path.exists(report_path)

    def test_pipeline_creates_run_log(self, tmp_dir):
        from src.monitoring_pipeline import run_monitoring_pipeline

        log_path = str(tmp_dir / "run_log.jsonl")
        run_monitoring_pipeline(
            predictions_csv=str(tmp_dir / "no_predictions.csv"),
            training_csv=str(tmp_dir / "no_training.csv"),
            baseline_path=str(tmp_dir / "baseline.json"),
            run_log_path=log_path,
            report_path=str(tmp_dir / "report.json"),
            verbose=False,
            plot=False,
        )
        records = load_run_log(log_path)
        assert any(r["phase"] == "phase5" for r in records)

    def test_pipeline_with_existing_predictions_csv(self, tmp_dir, sample_df):
        from src.monitoring_pipeline import run_monitoring_pipeline

        # Write the sample DataFrame as a "predictions" CSV
        pred_csv = str(tmp_dir / "predictions.csv")
        sample_df.to_csv(pred_csv, index=False)

        report = run_monitoring_pipeline(
            predictions_csv=pred_csv,
            training_csv=str(tmp_dir / "no_training.csv"),
            baseline_path=str(tmp_dir / "baseline.json"),
            run_log_path=str(tmp_dir / "run_log.jsonl"),
            report_path=str(tmp_dir / "report.json"),
            verbose=False,
            plot=False,
        )
        # n_predictions is logged in the run history, not the top-level summary
        assert len(report["run_history"]) >= 1
        last_run = report["run_history"][-1]
        assert last_run["metrics"]["n_predictions"] == len(sample_df)

    def test_pipeline_plot_flag(self, tmp_dir, sample_df):
        from src.monitoring_pipeline import run_monitoring_pipeline

        # Force enough drift to trigger a chart
        pred_csv = str(tmp_dir / "predictions.csv")
        sample_df_shifted = sample_df.copy()
        sample_df_shifted["strike_price"] = sample_df_shifted["strike_price"] + 2000
        sample_df_shifted.to_csv(pred_csv, index=False)

        run_monitoring_pipeline(
            predictions_csv=pred_csv,
            training_csv=str(tmp_dir / "no_training.csv"),
            baseline_path=str(tmp_dir / "baseline.json"),
            run_log_path=str(tmp_dir / "run_log.jsonl"),
            report_path=str(tmp_dir / "report.json"),
            verbose=False,
            plot=True,
        )
        # Test passes as long as no exception is raised; chart is optional
