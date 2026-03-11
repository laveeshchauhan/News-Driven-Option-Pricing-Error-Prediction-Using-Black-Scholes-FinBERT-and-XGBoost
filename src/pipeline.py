"""
pipeline.py — Phase 1 Orchestrator: End-to-End Black-Scholes Pricing Pipeline.

Steps performed by :func:`run_pipeline`:
  1. Load NSE option chain data (CSV file or synthetic sample)
  2. Download / calculate historical volatility for RELIANCE.NS
  3. Run Black-Scholes pricing for every row (Call & Put)
  4. Calculate all Greeks (Delta, Gamma, Theta, Vega, Rho)
  5. Compute ΔX = Actual Market Premium − BS Theoretical Price
  6. Classify options as Overpriced / Underpriced / Fair
  7. Generate summary statistics
  8. Export results to CSV in ``outputs/``
  9. Print a formatted summary table to terminal

All monetary values are in ₹ (INR).
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ACTUAL_PREMIUM_COLUMN,
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_TICKER,
    DEFAULT_VOLATILITY_WINDOW,
    MIN_VOLATILITY,
    OUTPUT_DIR,
    SAMPLE_OPTION_CHAIN_PATH,
)
from src.black_scholes import black_scholes_price, pricing_error
from src.greeks import all_greeks
from src.volatility import (
    download_and_calculate_volatility,
    get_volatility_for_date,
)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ensure_output_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _classify(delta_x: float, threshold: float = 1.0) -> str:
    """
    Classify an option as Overpriced, Underpriced, or Fair.

    Parameters
    ----------
    delta_x   : ΔX = actual premium − BS theoretical price (₹).
    threshold : Absolute ₹ band considered 'Fair' (default ₹1).
    """
    if delta_x > threshold:
        return "Overpriced"
    elif delta_x < -threshold:
        return "Underpriced"
    return "Fair"


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline function
# ──────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    input_path: Optional[str] = None,
    demo: bool = False,
    ticker: str = DEFAULT_TICKER,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
    plot: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full Phase 1 Black-Scholes option pricing pipeline.

    Parameters
    ----------
    input_path       : Path to NSE option chain CSV file(s) (glob supported).
                       If *None* and *demo* is False, uses the bundled sample.
    demo             : If True, generates synthetic data for a quick demo.
    ticker           : Yahoo Finance ticker for volatility download (default
                       ``'RELIANCE.NS'``).
    risk_free_rate   : Annualised risk-free rate (default 0.07 for 7 %).
    volatility_window: Rolling volatility window in trading days (default 30).
    output_csv       : Path for the results CSV output.
    dividend_yield   : Continuous dividend yield (default 0 — not needed for
                       short-term RELIANCE options).
    plot             : If True, generate and save visualisation charts.
    verbose          : If True, print progress and summary to stdout.

    Returns
    -------
    pd.DataFrame containing all pricing and Greeks results.
    """
    # ── Step 1: Load data ──────────────────────────────────────────────────
    if demo:
        if verbose:
            print("[1/9] Generating synthetic demo data...")
        from src.data_loader import generate_sample_dataframe
        df = generate_sample_dataframe(risk_free_rate=risk_free_rate)
    else:
        path = input_path or SAMPLE_OPTION_CHAIN_PATH
        if verbose:
            print(f"[1/9] Loading option chain data from: {path}")
        from src.data_loader import load_option_chain
        df = load_option_chain(path, risk_free_rate=risk_free_rate)

    if df.empty:
        raise ValueError("No data loaded — DataFrame is empty after cleaning.")

    if verbose:
        print(f"      Loaded {len(df)} rows × {len(df.columns)} columns.")

    # ── Step 2: Calculate historical volatility ────────────────────────────
    if verbose:
        print(f"[2/9] Downloading historical volatility for {ticker}...")

    ann_vol: float
    rolling_vol: Optional[pd.Series] = None

    try:
        # Use the earliest trade date in the dataset as reference
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
            ref_date = df["date"].min()
        else:
            ref_date = datetime.today()

        ann_vol, rolling_vol = download_and_calculate_volatility(
            ticker=ticker,
            window=volatility_window,
            reference_date=ref_date,
        )
        if verbose:
            print(f"      Annualised historical volatility: {ann_vol:.4f} ({ann_vol*100:.2f}%)")
    except Exception as exc:
        warnings.warn(
            f"Could not download volatility data: {exc}\n"
            "Falling back to default volatility of 25 %.",
            RuntimeWarning,
            stacklevel=2,
        )
        ann_vol = 0.25

    # ── Assign per-row volatility ─────────────────────────────────────────
    if rolling_vol is not None and "date" in df.columns:
        def _get_vol(row_date: pd.Timestamp) -> float:
            try:
                return get_volatility_for_date(rolling_vol, row_date, fallback=ann_vol)
            except Exception:
                return ann_vol

        df["volatility"] = df["date"].apply(_get_vol)
    else:
        df["volatility"] = ann_vol

    # Clamp very small values
    df["volatility"] = df["volatility"].clip(lower=MIN_VOLATILITY)

    if verbose:
        print(f"      Volatility range: {df['volatility'].min():.4f} – {df['volatility'].max():.4f}")

    # ── Step 3: Black-Scholes pricing ──────────────────────────────────────
    if verbose:
        print("[3/9] Running Black-Scholes pricing for every row...")

    S = df["underlying_value"].to_numpy()
    K = df["strike_price"].to_numpy()
    T = df["time_to_expiry"].to_numpy()
    r = df["risk_free_rate"].to_numpy()
    sigma = df["volatility"].to_numpy()
    q = np.full(len(df), dividend_yield)

    # Vectorised pricing
    bs_prices = np.array([
        float(black_scholes_price(otype, s, k, t, rv, sig, qi))
        for otype, s, k, t, rv, sig, qi in zip(
            df["option_type"], S, K, T, r, sigma, q
        )
    ])
    df["bs_price"] = bs_prices

    # ── Step 4: Calculate Greeks ───────────────────────────────────────────
    if verbose:
        print("[4/9] Calculating Greeks (Δ, Γ, Θ, ν, ρ)...")

    greeks_records: dict[str, list] = {
        "delta": [], "gamma": [], "theta": [], "vega": [], "rho": []
    }
    for _, row in df.iterrows():
        g = all_greeks(
            row["option_type"],
            row["underlying_value"],
            row["strike_price"],
            row["time_to_expiry"],
            row["risk_free_rate"],
            row["volatility"],
            dividend_yield,
        )
        for k_name in greeks_records:
            greeks_records[k_name].append(float(g[k_name]))

    for k_name, vals in greeks_records.items():
        df[k_name] = vals

    # ── Step 5: Compute ΔX ────────────────────────────────────────────────
    if verbose:
        print("[5/9] Computing ΔX = Actual Premium − BS Theoretical Price...")

    actual_col = ACTUAL_PREMIUM_COLUMN if ACTUAL_PREMIUM_COLUMN in df.columns else "ltp"
    if actual_col not in df.columns:
        # Fall back to 'close' or 'settle_price'
        for alt in ("close", "settle_price"):
            if alt in df.columns:
                actual_col = alt
                break

    df["actual_premium"] = df[actual_col]
    df["delta_x"] = pricing_error(df["actual_premium"].to_numpy(), bs_prices)

    # ── Step 6: Classify ──────────────────────────────────────────────────
    if verbose:
        print("[6/9] Classifying options (Overpriced / Underpriced / Fair)...")

    df["classification"] = df["delta_x"].apply(_classify)

    # ── Step 7: Summary statistics ────────────────────────────────────────
    if verbose:
        print("[7/9] Generating summary statistics...")

    _print_summary(df, verbose=verbose)

    # ── Step 8: Export CSV ────────────────────────────────────────────────
    if verbose:
        print(f"[8/9] Exporting results to: {output_csv}")

    _ensure_output_dir(output_csv)
    df.to_csv(output_csv, index=False)
    if verbose:
        print(f"      Saved {len(df)} rows to {output_csv}")

    # ── Step 9: Plots ──────────────────────────────────────────────────────
    if plot:
        if verbose:
            print("[9/9] Generating visualisation charts...")
        _generate_plots(df, output_dir=OUTPUT_DIR, verbose=verbose)
    elif verbose:
        print("[9/9] Plots skipped (use --plot to enable).")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Summary printing
# ──────────────────────────────────────────────────────────────────────────────


def _print_summary(df: pd.DataFrame, verbose: bool = True) -> None:
    """Print a formatted summary of the pricing results."""
    if not verbose:
        return

    n_total = len(df)
    n_call = (df["option_type"].str.upper() == "CE").sum()
    n_put = (df["option_type"].str.upper() == "PE").sum()
    n_over = (df["classification"] == "Overpriced").sum()
    n_under = (df["classification"] == "Underpriced").sum()
    n_fair = (df["classification"] == "Fair").sum()

    mean_dx = df["delta_x"].mean()
    std_dx = df["delta_x"].std()
    rmse = np.sqrt((df["delta_x"] ** 2).mean())

    sep = "=" * 65
    print(f"\n{sep}")
    print("  PHASE 1 — BLACK-SCHOLES PRICING RESULTS SUMMARY")
    print(f"  All prices in ₹ (INR)  |  RELIANCE  |  NSE India")
    print(sep)
    print(f"  Total option rows processed  : {n_total:>8,}")
    print(f"  Call options (CE)            : {n_call:>8,}")
    print(f"  Put options  (PE)            : {n_put:>8,}")
    print(sep)
    print(f"  Mean ΔX (Actual − BS)        : ₹{mean_dx:>10.4f}")
    print(f"  Std  ΔX                      : ₹{std_dx:>10.4f}")
    print(f"  RMSE                         : ₹{rmse:>10.4f}")
    print(sep)
    print(f"  Overpriced  (ΔX >  ₹1)      : {n_over:>8,}")
    print(f"  Underpriced (ΔX < −₹1)      : {n_under:>8,}")
    print(f"  Fair        (|ΔX| ≤ ₹1)     : {n_fair:>8,}")
    print(sep)

    # Show top 10 rows
    display_cols = [
        "option_type", "strike_price", "underlying_value",
        "actual_premium", "bs_price", "delta_x", "classification",
        "delta", "gamma", "theta", "vega",
    ]
    available = [c for c in display_cols if c in df.columns]
    print(f"\n  Sample rows (top 10):\n")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 130,
        "display.float_format", "{:.4f}".format,
    ):
        print(df[available].head(10).to_string(index=False))
    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────


def _generate_plots(df: pd.DataFrame, output_dir: str, verbose: bool = True) -> None:
    """Generate and save visualisation charts using matplotlib / seaborn."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        warnings.warn("matplotlib / seaborn not installed. Skipping plots.", RuntimeWarning)
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ── Plot 1: BS Price vs Actual Premium ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, otype, color in zip(axes, ["CE", "PE"], ["steelblue", "tomato"]):
        subset = df[df["option_type"].str.upper() == otype]
        if subset.empty:
            continue
        ax.scatter(
            subset["strike_price"], subset["actual_premium"],
            label="Market (LTP)", color=color, alpha=0.7, s=40
        )
        ax.scatter(
            subset["strike_price"], subset["bs_price"],
            label="BS Theoretical", color="black", marker="x", s=50
        )
        ax.set_title(f"{'Call (CE)' if otype == 'CE' else 'Put (PE)'} — Market vs BS Price")
        ax.set_xlabel("Strike Price (₹)")
        ax.set_ylabel("Premium (₹)")
        ax.legend()

    plt.suptitle("RELIANCE NSE Options — Black-Scholes vs Market (Phase 1)", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "bs_vs_market.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"      Saved chart: {plot_path}")

    # ── Plot 2: ΔX distribution ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for otype, color in [("CE", "steelblue"), ("PE", "tomato")]:
        subset = df[df["option_type"].str.upper() == otype]
        if subset.empty:
            continue
        sns.histplot(
            subset["delta_x"], bins=30, kde=True, ax=ax,
            label=otype, color=color, alpha=0.5
        )
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="ΔX = 0")
    ax.set_title("Distribution of ΔX = Actual Premium − BS Price")
    ax.set_xlabel("ΔX (₹)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    dx_path = os.path.join(output_dir, "delta_x_distribution.png")
    plt.savefig(dx_path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"      Saved chart: {dx_path}")

    # ── Plot 3: Greeks smile ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, otype in zip(axes, ["CE", "PE"]):
        subset = df[df["option_type"].str.upper() == otype].sort_values("strike_price")
        if subset.empty:
            continue
        ax.plot(subset["strike_price"], subset["delta"],
                label="Delta", marker="o", markersize=3)
        ax2 = ax.twinx()
        ax2.plot(subset["strike_price"], subset["vega"],
                 label="Vega", color="orange", linestyle="--", marker="s", markersize=3)
        ax.set_title(f"{'Call' if otype == 'CE' else 'Put'} Greeks vs Strike")
        ax.set_xlabel("Strike Price (₹)")
        ax.set_ylabel("Delta", color="steelblue" if otype == "CE" else "tomato")
        ax2.set_ylabel("Vega (₹ per 1% σ)", color="orange")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    plt.suptitle("RELIANCE Options — Delta & Vega Smile (Phase 1)", fontsize=13)
    plt.tight_layout()
    greeks_path = os.path.join(output_dir, "greeks_smile.png")
    plt.savefig(greeks_path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"      Saved chart: {greeks_path}")
