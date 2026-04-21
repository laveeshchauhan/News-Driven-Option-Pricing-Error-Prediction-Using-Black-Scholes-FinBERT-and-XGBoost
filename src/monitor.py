"""
monitor.py — Phase 5: MLOps & Monitoring.

Provides lightweight, dependency-free model monitoring utilities:

  log_run            — Append a run record (metrics + config) to a JSONL log.
  save_baseline      — Persist training-set feature statistics as the drift
                       reference baseline.
  load_baseline      — Load the persisted baseline from disk.
  compute_psi        — Population Stability Index for a single feature.
  detect_data_drift  — Per-feature drift assessment (PSI + KS test) comparing
                       new data to the baseline.
  monitor_predictions — Evaluate live predictions against ground-truth ΔX
                        when actuals are available.
  check_alerts       — Convert drift/performance results into human-readable
                        alert messages and severity levels.
  generate_report    — Aggregate everything into a structured report dict and
                        optionally persist it as JSON.

All monetary values are in ₹ (INR).
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ML_FEATURE_COLUMNS,
    MONITOR_BASELINE_PATH,
    MONITOR_KS_PVALUE_THRESHOLD,
    MONITOR_MAX_HISTORY_RECORDS,
    MONITOR_PSI_ALERT_THRESHOLD,
    MONITOR_PSI_WARNING_THRESHOLD,
    MONITOR_REPORT_PATH,
    MONITOR_RMSE_DEGRADATION_THRESHOLD,
    MONITOR_RUN_LOG_PATH,
)


# ──────────────────────────────────────────────────────────────────────────────
# Run logging
# ──────────────────────────────────────────────────────────────────────────────


def log_run(
    phase: str,
    metrics: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
    log_path: str = MONITOR_RUN_LOG_PATH,
) -> Dict[str, Any]:
    """
    Append a structured run record to the JSONL run log.

    Parameters
    ----------
    phase    : Phase identifier string, e.g. ``'phase3'``, ``'phase4'``.
    metrics  : Dict of numeric metrics (RMSE, MAE, R², signal counts, …).
    params   : Optional dict of hyper-parameters or pipeline config used.
    tags     : Optional key/value string tags (e.g. ``{'mode': 'demo'}``).
    log_path : Path to the JSONL run log file.

    Returns
    -------
    The run record that was written (dict).
    """
    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "metrics": {k: _json_safe(v) for k, v in metrics.items()},
        "params": {k: _json_safe(v) for k, v in (params or {}).items()},
        "tags": tags or {},
    }

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    return record


def load_run_log(log_path: str = MONITOR_RUN_LOG_PATH) -> List[Dict[str, Any]]:
    """
    Read all records from the JSONL run log.

    Returns an empty list if the file does not exist or is empty.
    """
    if not os.path.exists(log_path):
        return []
    records: List[Dict[str, Any]] = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    warnings.warn(
                        f"Skipping malformed line in run log: {line[:80]}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Baseline management
# ──────────────────────────────────────────────────────────────────────────────


def save_baseline(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    baseline_path: str = MONITOR_BASELINE_PATH,
) -> Dict[str, Any]:
    """
    Compute per-feature statistics from *df* and persist them as the
    monitoring baseline (used later for drift detection).

    Statistics stored per feature
    ------------------------------
    count, mean, std, min, p25, p50, p75, max, histogram (edges + counts)

    Parameters
    ----------
    df              : Training / reference DataFrame.
    feature_columns : Feature columns to baseline.
                      Defaults to ``config.ML_FEATURE_COLUMNS``.
    baseline_path   : Where to write the JSON baseline file.

    Returns
    -------
    The baseline dict that was persisted.
    """
    cols = feature_columns or ML_FEATURE_COLUMNS
    available = [c for c in cols if c in df.columns]

    baseline: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "features": {},
    }

    for col in available:
        series = df[col].dropna().astype(float)
        if series.empty:
            continue
        counts, edges = np.histogram(series, bins=20)
        baseline["features"][col] = {
            "count": int(len(series)),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "histogram": {
                "edges": edges.tolist(),
                "counts": counts.tolist(),
            },
        }

    os.makedirs(os.path.dirname(os.path.abspath(baseline_path)), exist_ok=True)
    with open(baseline_path, "w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=2)

    return baseline


def load_baseline(
    baseline_path: str = MONITOR_BASELINE_PATH,
) -> Dict[str, Any]:
    """
    Load the monitoring baseline from disk.

    Raises
    ------
    FileNotFoundError
        If the baseline file does not exist.
    """
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Monitoring baseline not found at '{baseline_path}'.\n"
            "Run Phase 5 with an existing model to create the baseline, or call "
            "save_baseline() on your training data."
        )
    with open(baseline_path, encoding="utf-8") as fh:
        return json.load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Drift detection
# ──────────────────────────────────────────────────────────────────────────────


def compute_psi(
    baseline_counts: List[float],
    baseline_edges: List[float],
    new_series: pd.Series,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute the Population Stability Index (PSI) for a single feature.

    PSI = Σ (actual% − expected%) × ln(actual% / expected%)

    *baseline_counts* and *baseline_edges* come from the training histogram
    (stored in :func:`save_baseline`).  *new_series* is the new data column.

    A higher PSI indicates greater distributional shift.

    Interpretation
    --------------
    PSI < 0.10  : No significant shift
    PSI 0.10–0.20: Moderate shift — monitor closely
    PSI > 0.20  : Significant shift — consider retraining
    """
    expected_counts = np.array(baseline_counts, dtype=float)
    edges = np.array(baseline_edges)

    new_vals = new_series.dropna().astype(float)
    if new_vals.empty:
        return 0.0

    # Bin new data into the same histogram edges
    actual_counts, _ = np.histogram(new_vals, bins=edges)
    actual_counts = actual_counts.astype(float)

    # Normalise to proportions
    expected_pct = expected_counts / (expected_counts.sum() + epsilon)
    actual_pct = actual_counts / (actual_counts.sum() + epsilon)

    # Replace zeros to avoid log(0)
    expected_pct = np.where(expected_pct < epsilon, epsilon, expected_pct)
    actual_pct = np.where(actual_pct < epsilon, epsilon, actual_pct)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return round(psi, 6)


def detect_data_drift(
    new_df: pd.DataFrame,
    baseline: Optional[Dict[str, Any]] = None,
    baseline_path: str = MONITOR_BASELINE_PATH,
    feature_columns: Optional[List[str]] = None,
    psi_warning: float = MONITOR_PSI_WARNING_THRESHOLD,
    psi_alert: float = MONITOR_PSI_ALERT_THRESHOLD,
    ks_pvalue_threshold: float = MONITOR_KS_PVALUE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Detect distributional drift between *new_df* and the stored baseline.

    Uses two complementary tests per feature:
    - **PSI** (Population Stability Index) — bucket-based comparison.
    - **KS test** (Kolmogorov-Smirnov two-sample) — non-parametric.

    Parameters
    ----------
    new_df           : DataFrame of new / live data.
    baseline         : Pre-loaded baseline dict.  If *None*, loaded from disk.
    baseline_path    : Where to read the baseline when *baseline* is *None*.
    feature_columns  : Feature columns to check.  Defaults to config list.
    psi_warning      : PSI threshold for "warning" level (default 0.10).
    psi_alert        : PSI threshold for "alert" level (default 0.20).
    ks_pvalue_threshold : KS p-value below which drift is flagged (default 0.05).

    Returns
    -------
    dict with keys:

    ``feature_drift``
        Per-feature dict: psi, ks_statistic, ks_pvalue, psi_status,
        ks_drifted.
    ``overall_drift_status``
        ``'ok'``, ``'warning'``, or ``'alert'``.
    ``drifted_features``
        List of feature names where drift was detected.
    ``n_features_checked``
        Number of features assessed.
    ``timestamp``
        ISO-8601 UTC timestamp of the check.
    """
    if baseline is None:
        baseline = load_baseline(baseline_path)

    cols = feature_columns or list(baseline.get("features", {}).keys())
    available = [c for c in cols if c in new_df.columns and c in baseline.get("features", {})]

    results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_features_checked": len(available),
        "feature_drift": {},
        "drifted_features": [],
        "overall_drift_status": "ok",
    }

    max_psi = 0.0
    any_ks_drift = False

    for col in available:
        feat_baseline = baseline["features"][col]
        new_series = new_df[col].dropna().astype(float)

        # PSI
        psi_val = compute_psi(
            baseline_counts=feat_baseline["histogram"]["counts"],
            baseline_edges=feat_baseline["histogram"]["edges"],
            new_series=new_series,
        )

        if psi_val < psi_warning:
            psi_status = "ok"
        elif psi_val < psi_alert:
            psi_status = "warning"
        else:
            psi_status = "alert"

        # KS test — reconstruct baseline sample from histogram midpoints
        ks_stat = float("nan")
        ks_pvalue = float("nan")
        ks_drifted = False
        try:
            from scipy import stats as _stats

            edges = np.array(feat_baseline["histogram"]["edges"])
            counts = np.array(feat_baseline["histogram"]["counts"])
            midpoints = (edges[:-1] + edges[1:]) / 2.0
            baseline_sample = np.repeat(midpoints, counts.astype(int))

            if len(baseline_sample) > 0 and len(new_series) > 0:
                ks_result = _stats.ks_2samp(baseline_sample, new_series.values)
                ks_stat = float(ks_result.statistic)
                ks_pvalue = float(ks_result.pvalue)
                ks_drifted = ks_pvalue < ks_pvalue_threshold
        except ImportError:
            warnings.warn(
                "scipy not installed — KS test skipped. "
                "Install scipy for full drift detection.",
                RuntimeWarning,
                stacklevel=2,
            )

        feature_result = {
            "psi": psi_val,
            "psi_status": psi_status,
            "ks_statistic": round(ks_stat, 6) if not np.isnan(ks_stat) else None,
            "ks_pvalue": round(ks_pvalue, 6) if not np.isnan(ks_pvalue) else None,
            "ks_drifted": ks_drifted,
        }
        results["feature_drift"][col] = feature_result

        if psi_status != "ok" or ks_drifted:
            results["drifted_features"].append(col)
            any_ks_drift = any_ks_drift or ks_drifted

        max_psi = max(max_psi, psi_val)

    # Overall status driven by worst PSI
    if max_psi >= psi_alert or any_ks_drift:
        results["overall_drift_status"] = "alert"
    elif max_psi >= psi_warning:
        results["overall_drift_status"] = "warning"
    else:
        results["overall_drift_status"] = "ok"

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Prediction monitoring
# ──────────────────────────────────────────────────────────────────────────────


def monitor_predictions(
    df: pd.DataFrame,
    predicted_col: str = "predicted_delta_x",
    actual_col: str = "delta_x",
) -> Dict[str, Any]:
    """
    Compare predicted ΔX values against actual ΔX ground truth.

    Parameters
    ----------
    df            : DataFrame containing both predicted and actual columns.
    predicted_col : Name of the predicted ΔX column (default ``'predicted_delta_x'``).
    actual_col    : Name of the actual ΔX column (default ``'delta_x'``).

    Returns
    -------
    dict with keys: rmse, mae, r2, mean_error, std_error, n_samples,
    max_abs_error, pct_within_5, pct_within_10, signal_accuracy
    (when a ``signal`` column is present).

    Returns an empty dict (with ``n_samples: 0``) when the required columns are
    absent or there are fewer than 2 valid rows.
    """
    if predicted_col not in df.columns or actual_col not in df.columns:
        return {"n_samples": 0, "error": "Required columns not found"}

    valid = df[[predicted_col, actual_col]].dropna()
    n = len(valid)
    if n < 2:
        return {"n_samples": n, "error": "Insufficient data for metrics"}

    y_pred = valid[predicted_col].astype(float).values
    y_true = valid[actual_col].astype(float).values
    errors = y_true - y_pred

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    result: Dict[str, Any] = {
        "n_samples": n,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4) if not np.isnan(r2) else None,
        "mean_error": round(float(np.mean(errors)), 4),
        "std_error": round(float(np.std(errors)), 4),
        "max_abs_error": round(float(np.max(np.abs(errors))), 4),
        "pct_within_5": round(float(np.mean(np.abs(errors) <= 5.0) * 100), 2),
        "pct_within_10": round(float(np.mean(np.abs(errors) <= 10.0) * 100), 2),
    }

    # Signal accuracy: fraction of rows where signal direction matches actual
    if "signal" in df.columns and n > 0:
        signals = df.loc[valid.index, "signal"]
        correct = 0
        for sig, actual, pred in zip(signals, y_true, y_pred):
            if sig == "BUY" and actual < 0:
                correct += 1
            elif sig == "SELL" and actual > 0:
                correct += 1
            elif sig == "HOLD" and abs(actual) <= 5:
                correct += 1
        result["signal_accuracy"] = round(correct / n * 100, 2)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Alerts
# ──────────────────────────────────────────────────────────────────────────────


def check_alerts(
    drift_results: Optional[Dict[str, Any]] = None,
    perf_metrics: Optional[Dict[str, Any]] = None,
    baseline_metrics: Optional[Dict[str, float]] = None,
    psi_warning: float = MONITOR_PSI_WARNING_THRESHOLD,
    psi_alert: float = MONITOR_PSI_ALERT_THRESHOLD,
    rmse_degradation_threshold: float = MONITOR_RMSE_DEGRADATION_THRESHOLD,
) -> List[Dict[str, str]]:
    """
    Convert drift and performance results into structured alert messages.

    Parameters
    ----------
    drift_results         : Output of :func:`detect_data_drift`.
    perf_metrics          : Output of :func:`monitor_predictions`.
    baseline_metrics      : Baseline performance metrics (from Phase 3 training)
                            to compare against.  If supplied, RMSE degradation
                            is checked.
    psi_warning           : PSI warning threshold.
    psi_alert             : PSI alert threshold.
    rmse_degradation_threshold : Max tolerable RMSE increase (₹) vs baseline.

    Returns
    -------
    List of alert dicts, each with keys ``level`` (``'warning'`` or ``'critical'``),
    ``category``, and ``message``.
    """
    alerts: List[Dict[str, str]] = []

    # ── Drift alerts ───────────────────────────────────────────────────────
    if drift_results:
        for feat, result in drift_results.get("feature_drift", {}).items():
            psi = result.get("psi", 0.0)
            ks_drifted = result.get("ks_drifted", False)

            if psi >= psi_alert:
                alerts.append(
                    {
                        "level": "critical",
                        "category": "data_drift",
                        "message": (
                            f"Feature '{feat}' has PSI={psi:.3f} ≥ {psi_alert} — "
                            "significant distribution shift detected. "
                            "Consider retraining the model."
                        ),
                    }
                )
            elif psi >= psi_warning:
                alerts.append(
                    {
                        "level": "warning",
                        "category": "data_drift",
                        "message": (
                            f"Feature '{feat}' has PSI={psi:.3f} (moderate drift). "
                            "Monitor for further changes."
                        ),
                    }
                )
            if ks_drifted and psi < psi_warning:
                # KS-only drift (distribution shape changed without large PSI)
                pval = result.get("ks_pvalue")
                alerts.append(
                    {
                        "level": "warning",
                        "category": "data_drift",
                        "message": (
                            f"Feature '{feat}' KS test p-value={pval} indicates "
                            "a statistically significant distribution shift."
                        ),
                    }
                )

    # ── Performance degradation alerts ────────────────────────────────────
    if perf_metrics and baseline_metrics:
        current_rmse = perf_metrics.get("rmse")
        baseline_rmse = baseline_metrics.get("rmse")
        if (
            current_rmse is not None
            and baseline_rmse is not None
            and isinstance(current_rmse, (int, float))
            and isinstance(baseline_rmse, (int, float))
        ):
            degradation = current_rmse - baseline_rmse
            if degradation > rmse_degradation_threshold:
                alerts.append(
                    {
                        "level": "critical",
                        "category": "model_performance",
                        "message": (
                            f"RMSE increased by ₹{degradation:.2f} vs. baseline "
                            f"(current: ₹{current_rmse:.2f}, baseline: ₹{baseline_rmse:.2f}). "
                            "Model retraining is recommended."
                        ),
                    }
                )

    if perf_metrics and not baseline_metrics:
        current_rmse = perf_metrics.get("rmse")
        if (
            current_rmse is not None
            and isinstance(current_rmse, (int, float))
            and current_rmse > 50.0
        ):
            alerts.append(
                {
                    "level": "warning",
                    "category": "model_performance",
                    "message": (
                        f"RMSE = ₹{current_rmse:.2f} is unusually high. "
                        "Review model predictions."
                    ),
                }
            )

    return alerts


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────


def generate_report(
    drift_results: Optional[Dict[str, Any]] = None,
    perf_metrics: Optional[Dict[str, Any]] = None,
    alerts: Optional[List[Dict[str, str]]] = None,
    run_log_path: str = MONITOR_RUN_LOG_PATH,
    report_path: str = MONITOR_REPORT_PATH,
    baseline_metrics: Optional[Dict[str, float]] = None,
    max_history: int = MONITOR_MAX_HISTORY_RECORDS,
) -> Dict[str, Any]:
    """
    Aggregate monitoring results into a structured report and persist as JSON.

    Parameters
    ----------
    drift_results    : Output of :func:`detect_data_drift` (optional).
    perf_metrics     : Output of :func:`monitor_predictions` (optional).
    alerts           : Output of :func:`check_alerts` (optional).
    run_log_path     : Path to the JSONL run log.
    report_path      : Where to write the JSON report.
    baseline_metrics : Baseline model performance (from Phase 3).
    max_history      : Max run-log records to embed in the report.

    Returns
    -------
    The report dict (also written to *report_path*).
    """
    # Load recent run history
    all_records = load_run_log(run_log_path)
    recent_records = all_records[-max_history:]

    # Determine overall system health
    drift_status = (drift_results or {}).get("overall_drift_status", "unknown")
    n_critical = sum(1 for a in (alerts or []) if a.get("level") == "critical")
    n_warning = sum(1 for a in (alerts or []) if a.get("level") == "warning")

    if n_critical > 0:
        health = "critical"
    elif n_warning > 0 or drift_status == "warning":
        health = "warning"
    elif drift_status == "ok":
        health = "ok"
    else:
        health = "unknown"

    report: Dict[str, Any] = {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_health": health,
        "summary": {
            "drift_status": drift_status,
            "n_critical_alerts": n_critical,
            "n_warning_alerts": n_warning,
            "n_features_drifted": len((drift_results or {}).get("drifted_features", [])),
        },
        "drift": drift_results,
        "performance": perf_metrics,
        "baseline_performance": baseline_metrics,
        "alerts": alerts or [],
        "run_history": recent_records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _json_safe(value: Any) -> Any:
    """Convert numpy types and special floats to JSON-serialisable Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value
