"""
volatility.py — Historical Volatility Calculator for RELIANCE.NS.

Downloads 1 year of daily closing prices via yfinance and computes:
  • Daily log returns: ln(P_t / P_{t-1})
  • Annualised historical volatility: σ = std(log_returns) × √252
  • Rolling-window volatility (configurable window, default 30 trading days)

All prices are in ₹ (INR).
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import (
    DEFAULT_TICKER,
    DEFAULT_VOLATILITY_WINDOW,
    MIN_VOLATILITY,
    TRADING_DAYS_PER_YEAR,
    VOLATILITY_HISTORY_YEARS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate daily log returns from a series of closing prices.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices (₹), indexed by date.

    Returns
    -------
    pd.Series of log returns (NaN for the first observation).
    """
    return np.log(prices / prices.shift(1))


# ──────────────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────────────


def calculate_historical_volatility(
    prices: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate the **full-sample** annualised historical volatility.

    Parameters
    ----------
    prices       : Daily closing prices (₹), time-ordered.
    trading_days : Number of trading days per year used for annualisation
                   (default 252).

    Returns
    -------
    float — annualised volatility (e.g. 0.25 means 25 %).

    Notes
    -----
    Uses ``ddof=1`` (sample standard deviation) for an unbiased estimator.
    """
    log_ret = _log_returns(prices).dropna()
    if len(log_ret) < 2:
        return MIN_VOLATILITY
    return float(log_ret.std(ddof=1) * np.sqrt(trading_days))


def calculate_rolling_volatility(
    prices: pd.Series,
    window: int = DEFAULT_VOLATILITY_WINDOW,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Calculate **rolling** annualised historical volatility.

    Parameters
    ----------
    prices       : Daily closing prices (₹), time-ordered.
    window       : Rolling window in trading days (default 30).
    trading_days : Number of trading days per year for annualisation.

    Returns
    -------
    pd.Series of rolling volatility values, aligned with *prices* index.
    The first ``window`` values will be NaN (insufficient data).

    Examples
    --------
    >>> vol = calculate_rolling_volatility(prices, window=30)
    >>> latest_vol = vol.iloc[-1]
    """
    log_ret = _log_returns(prices)
    rolling_vol = log_ret.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(
        trading_days
    )
    return rolling_vol


def download_and_calculate_volatility(
    ticker: str = DEFAULT_TICKER,
    window: int = DEFAULT_VOLATILITY_WINDOW,
    reference_date: datetime | None = None,
    history_years: int = VOLATILITY_HISTORY_YEARS,
) -> tuple[float, pd.Series]:
    """
    Download historical closing prices from Yahoo Finance and compute volatility.

    Parameters
    ----------
    ticker         : Yahoo Finance ticker symbol (default ``'RELIANCE.NS'``).
    window         : Rolling volatility window in trading days (default 30).
    reference_date : The date for which volatility is needed. Defaults to today.
                     The function downloads ``history_years`` of data **before**
                     this date.
    history_years  : Years of historical data to fetch (default 1).

    Returns
    -------
    (annualised_volatility, rolling_volatility_series)

    Raises
    ------
    RuntimeError
        If data cannot be downloaded or contains too few rows.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required to download historical prices. "
            "Install it with: pip install yfinance"
        ) from exc

    if reference_date is None:
        reference_date = datetime.today()

    end_date = reference_date
    start_date = end_date - timedelta(days=history_years * 365 + 30)  # small buffer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

    if df.empty or len(df) < window + 5:
        raise RuntimeError(
            f"Insufficient data downloaded for '{ticker}'. "
            f"Got {len(df)} rows; need at least {window + 5}."
        )

    # yfinance may return a MultiIndex when auto_adjust=True
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze().dropna()

    ann_vol = calculate_historical_volatility(close)
    rolling_vol = calculate_rolling_volatility(close, window=window)

    return ann_vol, rolling_vol


def get_volatility_for_date(
    rolling_vol: pd.Series,
    target_date: datetime | pd.Timestamp,
    fallback: float | None = None,
) -> float:
    """
    Return the rolling volatility value on or nearest to *target_date*.

    Parameters
    ----------
    rolling_vol : Rolling volatility series (output of
                  :func:`calculate_rolling_volatility`).
    target_date : The date for which volatility is required.
    fallback    : Value to return if no valid entry is found. If None, raises
                  ``ValueError``.

    Returns
    -------
    float — annualised volatility.
    """
    target_date = pd.Timestamp(target_date)
    # Filter to dates on or before target_date
    subset = rolling_vol.loc[:target_date].dropna()
    if subset.empty:
        if fallback is not None:
            return fallback
        raise ValueError(
            f"No rolling volatility available on or before {target_date}."
        )
    return float(subset.iloc[-1])
