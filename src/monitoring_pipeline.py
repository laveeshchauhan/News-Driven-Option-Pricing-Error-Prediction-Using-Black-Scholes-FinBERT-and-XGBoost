"""
monitoring_pipeline.py — Phase 5 Orchestrator: MLOps & Monitoring.

Steps performed by :func:`run_monitoring_pipeline`:
  1. Run Phase 1–4 pipeline (or load an existing predictions CSV).
  2. Save / update the monitoring baseline from the training set (Phase 3
     output), creating it on first run if absent.
  3. Detect data drift by comparing the current inference data against the
     baseline (PSI + KS test per feature).
  4. Monitor model performance against ground-truth ΔX when actuals are
     available in the live predictions.
  5. Check for alerts (drift / performance degradation).
  6. Log the run to the JSONL run log.
  7. Generate and persist the monitoring report JSON.
  8. Print a formatted Phase 5 summary to the terminal.
  9. Optionally generate a feature-drift bar chart.

All monetary values are in ₹ (INR).
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd

from config import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_TICKER,
    DEFAULT_VOLATILITY_WINDOW,
    INFERENCE_OUTPUT_CSV,
    INFERENCE_SIGNAL_BUY_THRESHOLD,
    INFERENCE_SIGNAL_SELL_THRESHOLD,
    ML_FEATURE_COLUMNS,
    ML_MODEL_PATH,
    MONITOR_BASELINE_PATH,
    MONITOR_REPORT_PATH,
    MONITOR_RUN_LOG_PATH,
    OUTPUT_DIR,
)
from src.monitor import (
    check_alerts,
    detect_data_drift,
    generate_report,
    load_baseline,
    load_run_log,
    log_run,
    monitor_predictions,
    save_baseline,
)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline function
# ──────────────────────────────────────────────────────────────────────────────


def run_monitoring_pipeline(
    # Data source options
    predictions_csv: str = INFERENCE_OUTPUT_CSV,
    training_csv: str = DEFAULT_OUTPUT_CSV,
    # Baseline options
    baseline_path: str = MONITOR_BASELINE_PATH,
    force_baseline_refresh: bool = False,
    # Run log / report paths
    run_log_path: str = MONITOR_RUN_LOG_PATH,
    report_path: str = MONITOR_REPORT_PATH,
    # Phase 1/4 re-run options
    run_inference: bool = False,
    input_path: Optional[str] = None,
    demo: bool = False,
    ticker: str = DEFAULT_TICKER,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
    model_path: str = ML_MODEL_PATH,
    buy_threshold: float = INFERENCE_SIGNAL_BUY_THRESHOLD,
    sell_threshold: float = INFERENCE_SIGNAL_SELL_THRESHOLD,
    # Output
    feature_columns: Optional[List[str]] = None,
    plot: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the Phase 5 MLOps & Monitoring pipeline end-to-end.

    Parameters
    ----------
    predictions_csv       : Path to Phase 4 live predictions CSV.
    training_csv          : Path to Phase 1/3 results CSV for baseline stats.
    baseline_path         : Path to monitoring baseline JSON.
    force_baseline_refresh: Overwrite the baseline even if it exists.
    run_log_path          : JSONL run log path.
    report_path           : JSON report output path.
    run_inference         : Re-run Phase 1–4 before monitoring (optional).
    input_path            : NSE CSV path for Phase 1 (when *run_inference* is True).
    demo                  : Use demo data for Phase 1 re-run.
    ticker                : Ticker for historical vol download.
    risk_free_rate        : Risk-free rate for Phase 1 re-run.
    volatility_window     : Volatility window for Phase 1 re-run.
    dividend_yield        : Dividend yield for Phase 1 re-run.
    model_path            : Phase 3 model path for Phase 4 re-run.
    buy_threshold         : BUY signal threshold for Phase 4 re-run.
    sell_threshold        : SELL signal threshold for Phase 4 re-run.
    feature_columns       : Features to monitor for drift (defaults to config).
    plot                  : Generate and save a drift chart.
    verbose               : Print progress to stdout.

    Returns
    -------
    The Phase 5 monitoring report dict.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("  PHASE 5 — MLOps & MONITORING")
        print("  RELIANCE  |  NSE India")
        print("=" * 70)

    # ── Step 1: (Optional) re-run Phase 4 ─────────────────────────────────
    if run_inference:
        if verbose:
            print("\n[Phase 5 — Step 1/7] Re-running Phase 4 live inference...")
        predictions_df = _rerun_phase4(
            input_path=input_path,
            demo=demo,
            ticker=ticker,
            risk_free_rate=risk_free_rate,
            volatility_window=volatility_window,
            dividend_yield=dividend_yield,
            model_path=model_path,
            output_csv=predictions_csv,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            verbose=verbose,
        )
    else:
        if verbose:
            print(f"\n[Phase 5 — Step 1/7] Loading predictions from: {predictions_csv}")
        predictions_df = _load_csv(predictions_csv, verbose)

    # ── Step 2: Build / refresh baseline ──────────────────────────────────
    if verbose:
        print(f"\n[Phase 5 — Step 2/7] Checking monitoring baseline at: {baseline_path}")

    cols = feature_columns or ML_FEATURE_COLUMNS
    baseline = _ensure_baseline(
        baseline_path=baseline_path,
        training_csv=training_csv,
        feature_columns=cols,
        force_refresh=force_baseline_refresh,
        verbose=verbose,
    )

    # ── Step 3: Detect data drift ─────────────────────────────────────────
    if verbose:
        print("\n[Phase 5 — Step 3/7] Detecting data drift...")

    drift_results: Optional[Dict[str, Any]] = None
    if predictions_df is not None and not predictions_df.empty and baseline:
        from src.model import engineer_features
        enriched = engineer_features(predictions_df)
        drift_results = detect_data_drift(
            new_df=enriched,
            baseline=baseline,
            baseline_path=baseline_path,
            feature_columns=cols,
        )
        if verbose:
            n_drifted = len(drift_results.get("drifted_features", []))
            overall = drift_results.get("overall_drift_status", "unknown")
            print(f"  Drift status: {overall.upper()}  |  features drifted: {n_drifted}")
    else:
        if verbose:
            print("  Skipping drift detection — predictions data not available.")

    # ── Step 4: Monitor predictions ───────────────────────────────────────
    if verbose:
        print("\n[Phase 5 — Step 4/7] Monitoring prediction performance...")

    perf_metrics: Optional[Dict[str, Any]] = None
    if predictions_df is not None and not predictions_df.empty:
        perf_metrics = monitor_predictions(predictions_df)
        n = perf_metrics.get("n_samples", 0)
        if n > 0:
            if verbose:
                rmse = perf_metrics.get("rmse")
                mae = perf_metrics.get("mae")
                r2 = perf_metrics.get("r2")
                if rmse is not None:
                    print(f"  RMSE: ₹{rmse:.4f}  |  MAE: ₹{mae:.4f}  |  R²: {r2}")
                else:
                    print(f"  Metrics computed on {n} samples.")
        else:
            perf_metrics = None
            if verbose:
                print("  No ground-truth ΔX available — performance metrics skipped.")

    # ── Step 5: Check alerts ───────────────────────────────────────────────
    if verbose:
        print("\n[Phase 5 — Step 5/7] Checking alerts...")

    baseline_metrics = _extract_baseline_metrics(run_log_path)
    alerts = check_alerts(
        drift_results=drift_results,
        perf_metrics=perf_metrics,
        baseline_metrics=baseline_metrics,
    )

    if verbose:
        if alerts:
            print(f"  {len(alerts)} alert(s) raised:")
            for a in alerts:
                level = a.get("level", "info").upper()
                print(f"    [{level}] {a.get('message', '')}")
        else:
            print("  No alerts — system looks healthy.")

    # ── Step 6: Log the run ────────────────────────────────────────────────
    if verbose:
        print(f"\n[Phase 5 — Step 6/7] Logging run to: {run_log_path}")

    run_metrics: Dict[str, Any] = {
        "n_predictions": int(len(predictions_df)) if predictions_df is not None else 0,
        "drift_status": drift_results.get("overall_drift_status") if drift_results else None,
        "n_drifted_features": len(drift_results.get("drifted_features", [])) if drift_results else 0,
        "n_alerts": len(alerts),
    }
    if perf_metrics:
        run_metrics.update({k: v for k, v in perf_metrics.items() if k != "error"})

    run_record = log_run(
        phase="phase5",
        metrics=run_metrics,
        params={
            "baseline_path": baseline_path,
            "predictions_csv": predictions_csv,
        },
        tags={"demo": str(demo)},
        log_path=run_log_path,
    )
    if verbose:
        print(f"  Run logged at {run_record['timestamp']}")

    # ── Step 7: Generate report ────────────────────────────────────────────
    if verbose:
        print(f"\n[Phase 5 — Step 7/7] Generating monitoring report → {report_path}")

    report = generate_report(
        drift_results=drift_results,
        perf_metrics=perf_metrics,
        alerts=alerts,
        run_log_path=run_log_path,
        report_path=report_path,
        baseline_metrics=baseline_metrics,
    )

    if verbose:
        _print_summary(report, alerts, drift_results)

    if plot and drift_results:
        _generate_plots(drift_results, output_dir=OUTPUT_DIR, verbose=verbose)
    elif plot and verbose:
        print("  Skipping drift chart — no drift results available.")

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_csv(path: str, verbose: bool) -> Optional[pd.DataFrame]:
    """Load a CSV file, returning None if it does not exist."""
    if not os.path.exists(path):
        if verbose:
            print(f"  File not found: {path} — creating demo data for monitoring.")
        return _make_demo_predictions()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if verbose:
        print(f"  Loaded {len(df)} rows from {path}")
    return df


def _make_demo_predictions() -> pd.DataFrame:
    """Create a small synthetic predictions DataFrame for demo / testing."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 30
    strikes = rng.integers(1300, 1500, n)
    spot = 1390.20
    bs_price = rng.uniform(5, 150, n)
    actual_premium = bs_price + rng.normal(0, 10, n)
    delta_x = actual_premium - bs_price
    predicted_delta_x = delta_x + rng.normal(0, 2, n)
    signals = [
        "BUY" if p < -2 else ("SELL" if p > 2 else "HOLD")
        for p in predicted_delta_x
    ]
    return pd.DataFrame(
        {
            "strike_price": strikes.astype(float),
            "underlying_value": spot,
            "time_to_expiry": rng.uniform(0.01, 0.3, n),
            "volatility": rng.uniform(0.15, 0.35, n),
            "risk_free_rate": 0.07,
            "bs_price": bs_price,
            "actual_premium": actual_premium,
            "delta_x": delta_x,
            "predicted_delta_x": predicted_delta_x,
            "signal": signals,
            "confidence": rng.uniform(0, 1, n),
            "delta": rng.uniform(0.1, 0.9, n),
            "gamma": rng.uniform(0.001, 0.01, n),
            "theta": rng.uniform(-0.5, -0.05, n),
            "vega": rng.uniform(0.5, 3.0, n),
            "rho": rng.uniform(-1.0, 1.0, n),
            "option_type": rng.choice(["CE", "PE"], n),
            "moneyness": spot / strikes.astype(float),
            "option_type_enc": rng.integers(0, 2, n).astype(int),
            "daily_sentiment_score": rng.uniform(-0.3, 0.3, n),
            "daily_pos_mean": rng.uniform(0.1, 0.6, n),
            "daily_neg_mean": rng.uniform(0.1, 0.4, n),
            "daily_article_count": rng.integers(1, 20, n).astype(float),
        }
    )


def _ensure_baseline(
    baseline_path: str,
    training_csv: str,
    feature_columns: List[str],
    force_refresh: bool,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    """Load or create the monitoring baseline."""
    if os.path.exists(baseline_path) and not force_refresh:
        try:
            baseline = load_baseline(baseline_path)
            if verbose:
                n = baseline.get("n_rows", "?")
                print(f"  Baseline loaded ({n} training rows, {len(baseline.get('features', {}))} features).")
            return baseline
        except Exception as exc:
            warnings.warn(f"Could not load baseline: {exc}", RuntimeWarning, stacklevel=2)

    # Create baseline from training CSV
    if os.path.exists(training_csv):
        if verbose:
            print(f"  Building new baseline from: {training_csv}")
        df = pd.read_csv(training_csv)
        try:
            from src.model import engineer_features
            df = engineer_features(df)
        except Exception:
            pass
        baseline = save_baseline(df, feature_columns=feature_columns, baseline_path=baseline_path)
        if verbose:
            print(f"  Baseline saved → {baseline_path}")
        return baseline
    else:
        # Fallback: build baseline from demo data
        if verbose:
            print(
                f"  Training CSV not found at '{training_csv}' — "
                "building baseline from demo data."
            )
        demo_df = _make_demo_predictions()
        baseline = save_baseline(
            demo_df, feature_columns=feature_columns, baseline_path=baseline_path
        )
        if verbose:
            print(f"  Demo baseline saved → {baseline_path}")
        return baseline


def _extract_baseline_metrics(run_log_path: str) -> Optional[Dict[str, float]]:
    """
    Extract the most recent Phase 3 metrics from the run log to use as the
    performance baseline for comparison.
    """
    records = load_run_log(run_log_path)
    for record in reversed(records):
        if record.get("phase") == "phase3":
            return record.get("metrics", {})
    return None


def _rerun_phase4(
    input_path: Optional[str],
    demo: bool,
    ticker: str,
    risk_free_rate: float,
    volatility_window: int,
    dividend_yield: float,
    model_path: str,
    output_csv: str,
    buy_threshold: float,
    sell_threshold: float,
    verbose: bool,
) -> pd.DataFrame:
    """Run Phase 4 and return the predictions DataFrame."""
    try:
        from src.live_pipeline import run_live_pipeline

        return run_live_pipeline(
            input_path=input_path,
            demo=demo,
            ticker=ticker,
            risk_free_rate=risk_free_rate,
            volatility_window=volatility_window,
            dividend_yield=dividend_yield,
            model_path=model_path,
            output_csv=output_csv,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            plot=False,
            verbose=verbose,
        )
    except Exception as exc:
        warnings.warn(
            f"Phase 4 re-run failed: {exc} — falling back to demo data.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _make_demo_predictions()


def _print_summary(
    report: Dict[str, Any],
    alerts: List[Dict[str, str]],
    drift_results: Optional[Dict[str, Any]],
) -> None:
    """Print a formatted Phase 5 monitoring summary."""
    sep = "=" * 70
    health = report.get("overall_health", "unknown").upper()
    summary = report.get("summary", {})

    print(f"\n{sep}")
    print("  PHASE 5 — MONITORING REPORT SUMMARY")
    print(f"  RELIANCE  |  NSE India  |  Health: {health}")
    print(sep)
    print(f"  Overall system health      : {health}")
    print(f"  Drift status               : {summary.get('drift_status', 'N/A').upper()}")
    print(f"  Features drifted           : {summary.get('n_features_drifted', 0)}")
    print(f"  Critical alerts            : {summary.get('n_critical_alerts', 0)}")
    print(f"  Warning alerts             : {summary.get('n_warning_alerts', 0)}")

    perf = report.get("performance") or {}
    if perf.get("n_samples", 0) > 0:
        print(sep)
        print("  Prediction Performance (vs ground-truth ΔX):")
        print(f"    RMSE             : ₹{perf.get('rmse', 'N/A')}")
        print(f"    MAE              : ₹{perf.get('mae', 'N/A')}")
        print(f"    R²               : {perf.get('r2', 'N/A')}")
        print(f"    % within ₹5     : {perf.get('pct_within_5', 'N/A')} %")
        print(f"    % within ₹10    : {perf.get('pct_within_10', 'N/A')} %")
        if "signal_accuracy" in perf:
            print(f"    Signal accuracy  : {perf.get('signal_accuracy', 'N/A')} %")

    if drift_results and drift_results.get("feature_drift"):
        print(sep)
        print("  Feature Drift (PSI scores, top 10):")
        drift_items = sorted(
            drift_results["feature_drift"].items(),
            key=lambda x: x[1].get("psi", 0),
            reverse=True,
        )[:10]
        for feat, res in drift_items:
            psi = res.get("psi", 0.0)
            status = res.get("psi_status", "ok").upper()
            bar = "█" * min(int(psi * 20), 20)
            print(f"    {feat:<30s} PSI={psi:.3f}  {bar}  [{status}]")

    if alerts:
        print(sep)
        print("  Alerts:")
        for a in alerts:
            level = a.get("level", "info").upper()
            print(f"    [{level}] {a.get('message', '')}")

    print(sep)
    print("[Phase 5] Monitoring pipeline complete.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────


def _generate_plots(
    drift_results: Dict[str, Any],
    output_dir: str,
    verbose: bool = True,
) -> None:
    """Generate a feature-drift PSI bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        warnings.warn(
            "matplotlib / seaborn not installed. Skipping Phase 5 plots.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    feature_drift = drift_results.get("feature_drift", {})
    if not feature_drift:
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    features = list(feature_drift.keys())
    psi_scores = [feature_drift[f].get("psi", 0.0) for f in features]
    statuses = [feature_drift[f].get("psi_status", "ok") for f in features]
    colors = {"ok": "steelblue", "warning": "orange", "alert": "tomato"}
    bar_colors = [colors.get(s, "gray") for s in statuses]

    # Sort by PSI descending
    sorted_items = sorted(zip(features, psi_scores, bar_colors), key=lambda x: -x[1])
    features_sorted = [x[0] for x in sorted_items]
    psi_sorted = [x[1] for x in sorted_items]
    colors_sorted = [x[2] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.4)))
    bars = ax.barh(features_sorted, psi_sorted, color=colors_sorted, alpha=0.8)
    ax.axvline(
        MONITOR_PSI_WARNING_THRESHOLD,
        color="orange",
        linestyle="--",
        linewidth=1.2,
        label=f"Warning threshold ({MONITOR_PSI_WARNING_THRESHOLD})",
    )
    ax.axvline(
        MONITOR_PSI_ALERT_THRESHOLD,
        color="tomato",
        linestyle="--",
        linewidth=1.2,
        label=f"Alert threshold ({MONITOR_PSI_ALERT_THRESHOLD})",
    )
    ax.set_title("Phase 5 — Feature Drift (PSI Score per Feature)")
    ax.set_xlabel("PSI Score")
    ax.set_ylabel("Feature")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(output_dir, "monitoring_drift.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved drift chart: {path}")


# Make the threshold importable for the chart function
from config import MONITOR_PSI_WARNING_THRESHOLD, MONITOR_PSI_ALERT_THRESHOLD  # noqa: E402
