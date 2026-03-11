"""
test_volatility.py — Unit tests for src/volatility.py.

Tests:
  • Log returns calculation
  • Annualised historical volatility from synthetic data
  • Rolling volatility length and NaN structure
  • Fallback behaviour when rolling series has no data
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.volatility import (
    calculate_historical_volatility,
    calculate_rolling_volatility,
    get_volatility_for_date,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def flat_prices() -> pd.Series:
    """Constant price series — volatility should be zero (or near MIN_VOLATILITY)."""
    dates = pd.date_range("2025-01-01", periods=252, freq="B")
    return pd.Series(1390.20, index=dates)


@pytest.fixture()
def random_prices() -> pd.Series:
    """Synthetic geometric Brownian motion prices with known daily σ."""
    rng = np.random.default_rng(0)
    n = 252
    daily_sigma = 0.02          # 2 % daily σ
    log_returns = rng.normal(0, daily_sigma, n)
    prices = 1000 * np.exp(np.cumsum(log_returns))
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates)


# ──────────────────────────────────────────────────────────────────────────────
# Historical volatility
# ──────────────────────────────────────────────────────────────────────────────


class TestHistoricalVolatility:

    def test_positive_output(self, random_prices):
        vol = calculate_historical_volatility(random_prices)
        assert vol > 0, "Volatility must be positive"

    def test_reasonable_range(self, random_prices):
        """Annualised vol should be between 1 % and 200 % for realistic data."""
        vol = calculate_historical_volatility(random_prices)
        assert 0.01 < vol < 2.0, f"Annualised vol {vol:.4f} outside expected range"

    def test_flat_prices_near_zero(self, flat_prices):
        """Constant prices → vol should be at or near MIN_VOLATILITY."""
        from config import MIN_VOLATILITY
        vol = calculate_historical_volatility(flat_prices)
        assert vol <= MIN_VOLATILITY + 1e-12 or vol < 0.001, (
            f"Constant prices should give near-zero vol, got {vol}"
        )

    def test_known_vol_approximation(self):
        """
        GBM with daily σ = 0.02 → annualised ≈ 0.02 × √252 ≈ 0.317.
        Allow ±30 % relative tolerance for sampling variation.
        """
        rng = np.random.default_rng(42)
        daily_sigma = 0.02
        n = 5000  # many samples for precision
        log_returns = rng.normal(0, daily_sigma, n)
        prices = 1000 * np.exp(np.cumsum(log_returns))
        vol = calculate_historical_volatility(pd.Series(prices))
        expected = daily_sigma * np.sqrt(252)
        assert abs(vol - expected) / expected < 0.30, (
            f"Expected vol ≈ {expected:.4f}, got {vol:.4f}"
        )

    def test_insufficient_data_returns_fallback(self):
        """Single price should not raise — returns MIN_VOLATILITY."""
        from config import MIN_VOLATILITY
        vol = calculate_historical_volatility(pd.Series([1390.20]))
        assert vol == MIN_VOLATILITY

    def test_two_prices(self):
        """Two prices → one return → computable volatility."""
        prices = pd.Series([1000.0, 1020.0])
        vol = calculate_historical_volatility(prices)
        assert vol >= 0


# ──────────────────────────────────────────────────────────────────────────────
# Rolling volatility
# ──────────────────────────────────────────────────────────────────────────────


class TestRollingVolatility:

    def test_output_length_matches_input(self, random_prices):
        rolling = calculate_rolling_volatility(random_prices, window=30)
        assert len(rolling) == len(random_prices)

    def test_first_window_minus_one_are_nan(self, random_prices):
        window = 30
        rolling = calculate_rolling_volatility(random_prices, window=window)
        # First (window) values should be NaN — log return at index 0 is NaN,
        # then we need window more returns, so first window values are NaN
        nan_count = rolling.isna().sum()
        assert nan_count >= window, (
            f"Expected ≥ {window} NaN values at start, got {nan_count}"
        )

    def test_values_after_window_are_positive(self, random_prices):
        window = 30
        rolling = calculate_rolling_volatility(random_prices, window=window)
        valid = rolling.dropna()
        assert (valid > 0).all(), "All rolling vol values should be positive"

    def test_window_1_equals_point_std(self, random_prices):
        """Window=1 should give all NaN (need at least 2 points for std with ddof=1)."""
        rolling = calculate_rolling_volatility(random_prices, window=1)
        # With ddof=1 and window=1, std is NaN for all points
        # This just checks it doesn't raise
        assert len(rolling) == len(random_prices)


# ──────────────────────────────────────────────────────────────────────────────
# get_volatility_for_date
# ──────────────────────────────────────────────────────────────────────────────


class TestGetVolatilityForDate:

    @pytest.fixture()
    def sample_rolling(self) -> pd.Series:
        dates = pd.date_range("2025-01-01", periods=100, freq="B")
        data = np.linspace(0.20, 0.30, 100)
        return pd.Series(data, index=dates)

    def test_exact_date_match(self, sample_rolling):
        target = sample_rolling.index[50]
        vol = get_volatility_for_date(sample_rolling, target)
        assert abs(vol - sample_rolling.iloc[50]) < 1e-10

    def test_date_before_series_uses_fallback(self, sample_rolling):
        early_date = pd.Timestamp("2020-01-01")
        vol = get_volatility_for_date(sample_rolling, early_date, fallback=0.25)
        assert vol == 0.25

    def test_date_before_series_no_fallback_raises(self, sample_rolling):
        early_date = pd.Timestamp("2020-01-01")
        with pytest.raises(ValueError):
            get_volatility_for_date(sample_rolling, early_date)

    def test_date_after_series_returns_last(self, sample_rolling):
        future_date = pd.Timestamp("2030-01-01")
        vol = get_volatility_for_date(sample_rolling, future_date)
        assert abs(vol - sample_rolling.iloc[-1]) < 1e-10
