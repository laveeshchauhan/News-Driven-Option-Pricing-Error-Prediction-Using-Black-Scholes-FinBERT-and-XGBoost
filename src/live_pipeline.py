"""
live_pipeline.py — Phase 4 Orchestrator: Live Inference Pipeline.

Steps performed by :func:`run_live_pipeline`:
  1. Run Phase 1 (Black-Scholes + Greeks) on the supplied option chain
  2. Optionally merge Phase 2 sentiment scores
  3. Verify that the Phase 3 trained model exists
  4. Run inference → predicted ΔX + trading signals
  5. Rank signals (BUY first, then SELL, then HOLD; within group by |ΔX| desc)
  6. Export ranked predictions to ``outputs/live_predictions.csv``
  7. Print a formatted signals table to terminal
  8. Optionally generate a predicted-ΔX-vs-strike scatter chart

All monetary values are in ₹ (INR).
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import pandas as pd

from config import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_TICKER,
    DEFAULT_VOLATILITY_WINDOW,
    INFERENCE_OUTPUT_CSV,
    INFERENCE_SIGNAL_BUY_THRESHOLD,
    INFERENCE_SIGNAL_SELL_THRESHOLD,
    ML_MODEL_PATH,
    OUTPUT_DIR,
    SENTIMENT_OUTPUT_CSV,
)
from src.inference import run_inference
from src.pipeline import run_pipeline


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline function
# ──────────────────────────────────────────────────────────────────────────────


def run_live_pipeline(
    input_path: Optional[str] = None,
    demo: bool = False,
    ticker: str = DEFAULT_TICKER,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
    model_path: str = ML_MODEL_PATH,
    sentiment_csv: Optional[str] = None,
    output_csv: str = INFERENCE_OUTPUT_CSV,
    buy_threshold: float = INFERENCE_SIGNAL_BUY_THRESHOLD,
    sell_threshold: float = INFERENCE_SIGNAL_SELL_THRESHOLD,
    plot: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the Phase 4 live inference pipeline end-to-end.

    Parameters
    ----------
    input_path       : Path to NSE option chain CSV (glob supported).
                       If *None* and *demo* is False, uses the bundled sample.
    demo             : If True, use synthetically generated sample data.
    ticker           : Yahoo Finance ticker for historical volatility download.
    risk_free_rate   : Annualised risk-free rate (decimal, e.g. 0.07 for 7 %).
    volatility_window: Rolling volatility window in trading days (default 30).
    dividend_yield   : Continuous dividend yield (default 0.0).
    model_path       : Path to the trained Phase 3 XGBoost model JSON.
    sentiment_csv    : Path to Phase 2 enriched CSV for optional sentiment
                       merging.  Defaults to ``outputs/results_with_sentiment.csv``.
    output_csv       : Where to write the live predictions CSV.
    buy_threshold    : |predicted ΔX| (₹) below which → BUY signal.
    sell_threshold   : predicted ΔX (₹) above which → SELL signal.
    plot             : Generate and save a signal scatter chart.
    verbose          : Print progress and results to stdout.

    Returns
    -------
    pd.DataFrame with all Phase 1 columns plus ``predicted_delta_x``,
    ``signal``, and ``confidence``.

    Raises
    ------
    FileNotFoundError
        If the trained model is not found at *model_path*.
    """
    # ── Step 1: Phase 1 — BS pricing + Greeks ─────────────────────────────
    if verbose:
        print("\n[Phase 4 — Step 1/7] Running Phase 1 (Black-Scholes + Greeks)...")

    p1_csv = os.path.join(OUTPUT_DIR, "live_p1_intermediate.csv")
    df = run_pipeline(
        input_path=input_path,
        demo=demo,
        ticker=ticker,
        risk_free_rate=risk_free_rate,
        volatility_window=volatility_window,
        output_csv=p1_csv,
        dividend_yield=dividend_yield,
        plot=False,
        verbose=verbose,
    )

    if verbose:
        print(f"  Phase 1 complete — {len(df)} rows processed.")

    # ── Step 2: Optional Phase 2 sentiment merge ───────────────────────────
    if verbose:
        print("[Phase 4 — Step 2/7] Loading sentiment features (if available)...")

    df = _merge_sentiment(df, sentiment_csv=sentiment_csv, verbose=verbose)

    # ── Step 3: Verify model exists ────────────────────────────────────────
    if verbose:
        print(f"[Phase 4 — Step 3/7] Loading trained model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'.\n"
            "Run Phase 3 first (python main.py --phase 3) to train the model."
        )

    # ── Step 4: Inference → predicted ΔX + signals ────────────────────────
    if verbose:
        print("[Phase 4 — Step 4/7] Running inference → predicted ΔX + signals...")

    result_df = run_inference(
        df=df,
        model_path=model_path,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    n_buy = (result_df["signal"] == "BUY").sum()
    n_sell = (result_df["signal"] == "SELL").sum()
    n_hold = (result_df["signal"] == "HOLD").sum()

    if verbose:
        print(f"  Signals — BUY: {n_buy}, SELL: {n_sell}, HOLD: {n_hold}")

    # ── Step 5: Rank signals ───────────────────────────────────────────────
    if verbose:
        print("[Phase 4 — Step 5/7] Ranking signals by confidence...")

    result_df = _rank_signals(result_df)

    # ── Step 6: Export predictions CSV ────────────────────────────────────
    if verbose:
        print(f"[Phase 4 — Step 6/7] Exporting predictions to: {output_csv}")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    if verbose:
        print(f"  Saved {len(result_df)} rows → {output_csv}")

    # ── Step 7: Print summary + optional chart ─────────────────────────────
    if verbose:
        _print_summary(result_df, buy_threshold, sell_threshold)

    if plot:
        _generate_plots(result_df, output_dir=OUTPUT_DIR, verbose=verbose)
    elif verbose:
        print("[Phase 4 — Step 7/7] Plots skipped (use --plot to enable).")

    return result_df


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _merge_sentiment(
    df: pd.DataFrame,
    sentiment_csv: Optional[str],
    verbose: bool,
) -> pd.DataFrame:
    """
    Try to left-join Phase 2 sentiment scores into *df* on the ``date`` column.
    If the sentiment CSV is absent or missing a ``date`` column, *df* is
    returned unchanged — sentiment features will be filled with zeros by the
    inference engine's feature engineer.
    """
    candidate = sentiment_csv or SENTIMENT_OUTPUT_CSV
    if not os.path.exists(candidate):
        if verbose:
            print(
                f"  Sentiment CSV not found at '{candidate}' "
                "— sentiment features will default to 0."
            )
        return df

    try:
        sent_df = pd.read_csv(candidate)
        sentiment_cols = [
            "date",
            "daily_sentiment_score",
            "daily_pos_mean",
            "daily_neg_mean",
            "daily_article_count",
        ]
        available_cols = [c for c in sentiment_cols if c in sent_df.columns]

        if "date" not in available_cols:
            if verbose:
                print("  Sentiment CSV has no 'date' column — skipping merge.")
            return df

        sent_df["date"] = pd.to_datetime(sent_df["date"])

        if "date" not in df.columns:
            if verbose:
                print("  Input data has no 'date' column — skipping sentiment merge.")
            return df

        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(
            sent_df[available_cols].drop_duplicates("date"),
            on="date",
            how="left",
        )
        if verbose:
            n_merged = int(df["daily_sentiment_score"].notna().sum())
            print(f"  Merged sentiment scores for {n_merged}/{len(df)} rows.")
    except Exception as exc:
        warnings.warn(f"Could not merge sentiment data: {exc}", RuntimeWarning, stacklevel=2)

    return df


def _rank_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort rows so that the most actionable signals appear first:

    1. BUY rows (most underpriced first by |predicted_delta_x|)
    2. SELL rows (most overpriced first)
    3. HOLD rows

    Returns a reset-index copy.
    """
    order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    df = df.copy()
    df["_signal_rank"] = df["signal"].map(order)
    df["_abs_pred_dx"] = df["predicted_delta_x"].abs()
    df = df.sort_values(
        ["_signal_rank", "_abs_pred_dx"], ascending=[True, False]
    ).drop(columns=["_signal_rank", "_abs_pred_dx"])
    return df.reset_index(drop=True)


def _print_summary(
    df: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
) -> None:
    """Print a formatted Phase 4 signals summary to stdout."""
    n_buy = int((df["signal"] == "BUY").sum())
    n_sell = int((df["signal"] == "SELL").sum())
    n_hold = int((df["signal"] == "HOLD").sum())

    sep = "=" * 70
    print(f"\n{sep}")
    print("  PHASE 4 — LIVE INFERENCE RESULTS SUMMARY")
    print("  All prices in ₹ (INR)  |  RELIANCE  |  NSE India")
    print(sep)
    print(f"  Total options analysed              : {len(df):>8,}")
    print(f"  BUY  signals  (pred ΔX < −₹{buy_threshold:.0f})    : {n_buy:>8,}")
    print(f"  SELL signals  (pred ΔX >  ₹{sell_threshold:.0f})    : {n_sell:>8,}")
    print(f"  HOLD signals  (|pred ΔX| ≤ ₹{buy_threshold:.0f})   : {n_hold:>8,}")
    print(sep)

    display_cols = [
        "option_type", "strike_price", "underlying_value",
        "bs_price", "actual_premium", "delta_x",
        "predicted_delta_x", "signal", "confidence",
        "delta", "gamma",
    ]
    available = [c for c in display_cols if c in df.columns]

    top_signals = df[df["signal"].isin(["BUY", "SELL"])].head(15)
    if not top_signals.empty:
        print("\n  Top actionable signals (BUY / SELL):\n")
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 150,
            "display.float_format", "{:.4f}".format,
        ):
            print(top_signals[available].to_string(index=False))
    else:
        print("\n  No BUY or SELL signals — all options are within the fair-price band.")

    print(f"\n{sep}\n")
    print("[Phase 4 — Step 7/7] Pipeline complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────


def _generate_plots(
    df: pd.DataFrame,
    output_dir: str,
    verbose: bool = True,
) -> None:
    """Generate Phase 4 signal scatter chart (predicted ΔX vs strike price)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        warnings.warn(
            "matplotlib / seaborn not installed. Skipping Phase 4 plots.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    if "strike_price" not in df.columns or "predicted_delta_x" not in df.columns:
        return

    color_map = {"BUY": "steelblue", "SELL": "tomato", "HOLD": "gray"}
    fig, ax = plt.subplots(figsize=(11, 6))

    for sig, color in color_map.items():
        subset = df[df["signal"] == sig]
        if subset.empty:
            continue
        ax.scatter(
            subset["strike_price"],
            subset["predicted_delta_x"],
            color=color,
            label=sig,
            alpha=0.75,
            s=55,
            zorder=3,
        )

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
    ax.axhline(
        INFERENCE_SIGNAL_SELL_THRESHOLD,
        color="tomato",
        linewidth=1.0,
        linestyle=":",
        alpha=0.65,
        label=f"SELL threshold (₹{INFERENCE_SIGNAL_SELL_THRESHOLD:.0f})",
    )
    ax.axhline(
        -INFERENCE_SIGNAL_BUY_THRESHOLD,
        color="steelblue",
        linewidth=1.0,
        linestyle=":",
        alpha=0.65,
        label=f"BUY threshold (−₹{INFERENCE_SIGNAL_BUY_THRESHOLD:.0f})",
    )

    ax.set_title("Phase 4 — Predicted ΔX vs Strike Price  (Live Inference Signals)")
    ax.set_xlabel("Strike Price (₹)")
    ax.set_ylabel("Predicted ΔX (₹)")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "live_inference_signals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  Saved chart: {path}")
