"""
data_loader.py — Load, clean, and enrich NSE option chain data.

Responsibilities:
  • Read NSE CSV files (single file or glob pattern for multiple files)
  • Rename columns to internal snake_case names
  • Parse dates (DATE, EXPIRY columns)
  • Compute Time-to-Expiry T = (expiry − date) / 365
  • Auto-download Reliance spot prices via yfinance when needed
  • Generate synthetic sample data for testing / demo

All monetary values are in ₹ (INR).
"""

from __future__ import annotations

import glob as _glob
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from config import (
    ACTUAL_PREMIUM_COLUMN,
    CALENDAR_DAYS_PER_YEAR,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_TICKER,
    MIN_TIME_TO_EXPIRY,
    NSE_COLUMN_MAP,
    SAMPLE_OPTION_CHAIN_PATH,
)


# ──────────────────────────────────────────────────────────────────────────────
# Date-parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

_DATE_FORMATS = ["%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]


def _parse_date_series(series: pd.Series) -> pd.Series:
    """Try multiple date formats and return a datetime Series."""
    for fmt in _DATE_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt)
            if parsed.notna().all():
                return parsed
        except (ValueError, TypeError):
            continue
    # Fallback: let pandas infer
    return pd.to_datetime(series, infer_datetime_format=True, dayfirst=True)


# ──────────────────────────────────────────────────────────────────────────────
# Column normalisation
# ──────────────────────────────────────────────────────────────────────────────


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw NSE column headers to internal snake_case names.

    Handles both exact matches and case-insensitive / stripped variants.
    """
    # Build a case-insensitive lookup
    lookup = {k.strip().upper(): v for k, v in NSE_COLUMN_MAP.items()}
    rename_map = {}
    for col in df.columns:
        key = col.strip().upper()
        if key in lookup:
            rename_map[col] = lookup[key]
    return df.rename(columns=rename_map)


# ──────────────────────────────────────────────────────────────────────────────
# Core loader
# ──────────────────────────────────────────────────────────────────────────────


def load_option_chain(
    path: str | List[str],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> pd.DataFrame:
    """
    Load one or more NSE option chain CSV files and return a cleaned DataFrame.

    Parameters
    ----------
    path           : Path to a single CSV file, a glob pattern (e.g.
                     ``'data/raw/*.csv'``), or a list of file paths.
    risk_free_rate : Risk-free rate to attach to every row (default 0.07).

    Returns
    -------
    pd.DataFrame with standardised columns including a computed ``time_to_expiry``
    column (T in years).

    Raises
    ------
    FileNotFoundError
        If no files are found at the specified path(s).
    """
    # ── Resolve file list ──────────────────────────────────────────────────
    if isinstance(path, list):
        file_list = path
    else:
        file_list = sorted(_glob.glob(path))
        if not file_list:
            # Single exact path
            file_list = [path]

    missing = [f for f in file_list if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"File(s) not found: {missing}")

    frames = []
    for fpath in file_list:
        try:
            raw = pd.read_csv(fpath, encoding="utf-8")
        except UnicodeDecodeError:
            raw = pd.read_csv(fpath, encoding="latin-1")
        raw.columns = raw.columns.str.strip()
        frames.append(raw)

    df = pd.concat(frames, ignore_index=True)

    # ── Rename columns ────────────────────────────────────────────────────
    df = _normalise_columns(df)

    # ── Strip whitespace from string columns ──────────────────────────────
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # ── Parse dates ───────────────────────────────────────────────────────
    if "date" in df.columns:
        df["date"] = _parse_date_series(df["date"])
    if "expiry" in df.columns:
        df["expiry"] = _parse_date_series(df["expiry"])

    # ── Numeric coercions ─────────────────────────────────────────────────
    numeric_cols = [
        "strike_price", "open", "high", "low", "close", "ltp",
        "settle_price", "contracts", "turnover_lakhs",
        "premium_turnover_lakhs", "open_interest", "change_in_oi",
        "underlying_value",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Time to expiry T = (expiry − date) / 365 ─────────────────────────
    if "date" in df.columns and "expiry" in df.columns:
        days_to_expiry = (df["expiry"] - df["date"]).dt.days
        df["time_to_expiry"] = np.maximum(
            days_to_expiry / CALENDAR_DAYS_PER_YEAR, MIN_TIME_TO_EXPIRY
        )

    # ── Attach risk-free rate ─────────────────────────────────────────────
    df["risk_free_rate"] = risk_free_rate

    # ── Drop rows with missing critical columns ───────────────────────────
    required = ["strike_price", "underlying_value", "option_type", "time_to_expiry"]
    existing_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing_required).reset_index(drop=True)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# yfinance spot-price downloader
# ──────────────────────────────────────────────────────────────────────────────


def download_spot_prices(
    ticker: str = DEFAULT_TICKER,
    start: str = "2025-03-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download daily spot prices for *ticker* using yfinance.

    Parameters
    ----------
    ticker : Yahoo Finance ticker (default ``'RELIANCE.NS'``).
    start  : Start date string ``'YYYY-MM-DD'``.
    end    : End date string ``'YYYY-MM-DD'`` (defaults to today).

    Returns
    -------
    pd.DataFrame with columns ``['date', 'close']`` indexed by ``date``.

    Raises
    ------
    RuntimeError
        If the download fails or returns an empty DataFrame.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required. Install with: pip install yfinance"
        ) from exc

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        raise RuntimeError(
            f"No data returned for ticker '{ticker}' between {start} and {end}."
        )

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze()

    result = close.reset_index()
    result.columns = ["date", "close"]
    result["date"] = pd.to_datetime(result["date"])
    return result.set_index("date")


# ──────────────────────────────────────────────────────────────────────────────
# Sample data generator
# ──────────────────────────────────────────────────────────────────────────────


def load_sample_data(
    path: str = SAMPLE_OPTION_CHAIN_PATH,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> pd.DataFrame:
    """
    Load the bundled sample NSE option chain CSV.

    Parameters
    ----------
    path           : Path to the sample CSV (default: data/sample/sample_option_chain.csv).
    risk_free_rate : Risk-free rate to attach to every row.

    Returns
    -------
    Cleaned pd.DataFrame ready for the Black-Scholes pipeline.
    """
    return load_option_chain(path, risk_free_rate=risk_free_rate)


def generate_sample_dataframe(
    spot_price: float = 1390.20,
    strikes: list[int] | None = None,
    trade_date: str = "2026-03-11",
    expiry_date: str = "2026-04-28",
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    volatility: float = 0.25,
) -> pd.DataFrame:
    """
    Generate a synthetic NSE-style option chain DataFrame for testing / demos.

    Parameters
    ----------
    spot_price     : Underlying asset price (₹).
    strikes        : List of strike prices to generate.  Defaults to a
                     range around *spot_price*.
    trade_date     : ISO date string for the trade date.
    expiry_date    : ISO date string for the option expiry.
    risk_free_rate : Risk-free rate.
    volatility     : Volatility used to generate dummy premiums (not fed to BS
                     later — the pipeline recalculates it).

    Returns
    -------
    pd.DataFrame matching the internal schema produced by :func:`load_option_chain`.
    """
    from src.black_scholes import black_scholes_call, black_scholes_put

    if strikes is None:
        base = int(round(spot_price / 10) * 10)
        strikes = list(range(base - 200, base + 201, 10))

    t_date = pd.Timestamp(trade_date)
    e_date = pd.Timestamp(expiry_date)
    T = max((e_date - t_date).days / CALENDAR_DAYS_PER_YEAR, MIN_TIME_TO_EXPIRY)

    rows = []
    rng = np.random.default_rng(42)

    for K in strikes:
        for opt_type in ["CE", "PE"]:
            if opt_type == "CE":
                fair = float(
                    black_scholes_call(spot_price, K, T, risk_free_rate, volatility)
                )
            else:
                fair = float(
                    black_scholes_put(spot_price, K, T, risk_free_rate, volatility)
                )
            # Add ±5 % noise to simulate market vs model spread
            noise = rng.uniform(-0.05, 0.05) * fair
            market_price = max(round(fair + noise, 2), 0.05)

            rows.append(
                {
                    "symbol": "RELIANCE",
                    "date": t_date,
                    "expiry": e_date,
                    "option_type": opt_type,
                    "strike_price": K,
                    "open": market_price,
                    "high": round(market_price * 1.02, 2),
                    "low": round(market_price * 0.98, 2),
                    "close": market_price,
                    "ltp": market_price,
                    "settle_price": market_price,
                    "contracts": int(rng.integers(50, 1500)),
                    "turnover_lakhs": round(market_price * 100 * 250 / 1e5, 2),
                    "premium_turnover_lakhs": round(market_price * 50 / 1e5 * 100, 2),
                    "open_interest": int(rng.integers(5000, 200000)),
                    "change_in_oi": int(rng.integers(-5000, 5000)),
                    "underlying_value": spot_price,
                    "time_to_expiry": T,
                    "risk_free_rate": risk_free_rate,
                }
            )

    return pd.DataFrame(rows)
