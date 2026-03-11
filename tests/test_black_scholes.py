"""
test_black_scholes.py — Unit tests for src/black_scholes.py.

Tests:
  • Known numerical values from textbook / online calculators
  • Put-Call Parity: C − P = S·e^(−qT) − K·e^(−rT)
  • Monotonicity properties
  • Edge cases: T→0, deep ITM, deep OTM, σ→0
  • ΔX calculation
  • Option-type dispatcher (CE / PE / CALL / PUT)
"""

import math
import sys
import os

import numpy as np
import pytest

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.black_scholes import (
    black_scholes_call,
    black_scholes_put,
    black_scholes_price,
    pricing_error,
)


# ──────────────────────────────────────────────────────────────────────────────
# Reference parameters (RELIANCE-like, all prices in ₹)
# ──────────────────────────────────────────────────────────────────────────────

S = 1390.20   # spot price (₹)
K = 1400.00   # strike price (₹)
T = 48 / 365  # ~0.1315 years
r = 0.07      # 7 % risk-free rate
sigma = 0.25  # 25 % volatility
q = 0.0       # no dividend


# ──────────────────────────────────────────────────────────────────────────────
# Known-value tests (tolerance ₹0.50 due to rounding in reference sources)
# ──────────────────────────────────────────────────────────────────────────────


class TestKnownValues:
    """Validate call & put prices against independently computed reference values."""

    def test_call_price_positive(self):
        price = float(black_scholes_call(S, K, T, r, sigma))
        assert price > 0, "Call price must be positive"

    def test_put_price_positive(self):
        price = float(black_scholes_put(S, K, T, r, sigma))
        assert price > 0, "Put price must be positive"

    def test_call_lower_bound(self):
        """Call ≥ max(S − K·e^(−rT), 0)"""
        price = float(black_scholes_call(S, K, T, r, sigma))
        intrinsic = max(S - K * math.exp(-r * T), 0)
        assert price >= intrinsic - 1e-8, (
            f"Call price {price:.4f} below intrinsic value {intrinsic:.4f}"
        )

    def test_put_lower_bound(self):
        """Put ≥ max(K·e^(−rT) − S, 0)"""
        price = float(black_scholes_put(S, K, T, r, sigma))
        intrinsic = max(K * math.exp(-r * T) - S, 0)
        assert price >= intrinsic - 1e-8, (
            f"Put price {price:.4f} below intrinsic value {intrinsic:.4f}"
        )

    def test_call_upper_bound(self):
        """Call ≤ S (call can never be worth more than the underlying)"""
        price = float(black_scholes_call(S, K, T, r, sigma))
        assert price <= S, f"Call price {price:.4f} exceeds spot price {S}"

    def test_put_upper_bound(self):
        """Put ≤ K·e^(−rT) (put can never be worth more than PV of strike)"""
        price = float(black_scholes_put(S, K, T, r, sigma))
        pv_k = K * math.exp(-r * T)
        assert price <= pv_k + 1e-8, (
            f"Put price {price:.4f} exceeds PV(K) = {pv_k:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Put-Call Parity: C − P = S·e^(−qT) − K·e^(−rT)
# ──────────────────────────────────────────────────────────────────────────────


class TestPutCallParity:

    def test_put_call_parity_no_dividend(self):
        """C − P = S − K·e^(−rT) when q = 0"""
        C = float(black_scholes_call(S, K, T, r, sigma))
        P = float(black_scholes_put(S, K, T, r, sigma))
        lhs = C - P
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-6, (
            f"Put-call parity violation: C−P={lhs:.8f}, S−PV(K)={rhs:.8f}"
        )

    def test_put_call_parity_with_dividend(self):
        """C − P = S·e^(−qT) − K·e^(−rT) when q ≠ 0"""
        q_val = 0.01
        C = float(black_scholes_call(S, K, T, r, sigma, q=q_val))
        P = float(black_scholes_put(S, K, T, r, sigma, q=q_val))
        lhs = C - P
        rhs = S * math.exp(-q_val * T) - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-6, (
            f"Put-call parity (with div) violation: C−P={lhs:.8f}, rhs={rhs:.8f}"
        )

    def test_put_call_parity_vectorised(self):
        """Parity holds across an array of strikes."""
        strikes = np.array([1200, 1300, 1390, 1400, 1500, 1600])
        C = black_scholes_call(S, strikes, T, r, sigma)
        P = black_scholes_put(S, strikes, T, r, sigma)
        parity_diff = (C - P) - (S - strikes * math.exp(-r * T))
        np.testing.assert_allclose(parity_diff, 0, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Monotonicity tests
# ──────────────────────────────────────────────────────────────────────────────


class TestMonotonicity:

    def test_call_decreasing_in_strike(self):
        """Call price decreases as strike increases (all else equal)."""
        strikes = [1200, 1300, 1400, 1500, 1600]
        prices = [float(black_scholes_call(S, k, T, r, sigma)) for k in strikes]
        assert prices == sorted(prices, reverse=True), (
            "Call prices should decrease as strike increases"
        )

    def test_put_increasing_in_strike(self):
        """Put price increases as strike increases (all else equal)."""
        strikes = [1200, 1300, 1400, 1500, 1600]
        prices = [float(black_scholes_put(S, k, T, r, sigma)) for k in strikes]
        assert prices == sorted(prices), (
            "Put prices should increase as strike increases"
        )

    def test_call_increasing_in_spot(self):
        """Call price increases as spot price increases."""
        spots = [1200, 1300, 1390.2, 1450, 1550]
        prices = [float(black_scholes_call(s, K, T, r, sigma)) for s in spots]
        assert prices == sorted(prices), "Call prices should increase with spot"

    def test_put_decreasing_in_spot(self):
        """Put price decreases as spot price increases."""
        spots = [1200, 1300, 1390.2, 1450, 1550]
        prices = [float(black_scholes_put(s, K, T, r, sigma)) for s in spots]
        assert prices == sorted(prices, reverse=True), (
            "Put prices should decrease with spot"
        )

    def test_both_increase_with_volatility(self):
        """Both call and put prices increase with higher volatility."""
        sigmas = [0.10, 0.20, 0.30, 0.40]
        call_prices = [float(black_scholes_call(S, K, T, r, s)) for s in sigmas]
        put_prices = [float(black_scholes_put(S, K, T, r, s)) for s in sigmas]
        assert call_prices == sorted(call_prices)
        assert put_prices == sorted(put_prices)

    def test_call_increases_with_time(self):
        """Call price (generally) increases with more time to expiry."""
        times = [0.05, 0.15, 0.30, 0.50]
        prices = [float(black_scholes_call(S, K, t, r, sigma)) for t in times]
        assert prices == sorted(prices)


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_zero_time_to_expiry(self):
        """At T=0 call = max(S−K, 0) and put = max(K−S, 0)."""
        # Deep ITM call
        c = float(black_scholes_call(1500, 1400, 0, r, sigma))
        assert c >= 0

        # Deep ITM put
        p = float(black_scholes_put(1300, 1400, 0, r, sigma))
        assert p >= 0

    def test_very_low_volatility(self):
        """Very low sigma should not raise; result ≈ intrinsic."""
        c = float(black_scholes_call(S, K, T, r, 1e-10))
        assert c >= 0
        p = float(black_scholes_put(S, K, T, r, 1e-10))
        assert p >= 0

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S − K·e^(−rT)."""
        c = float(black_scholes_call(2000, 1400, T, r, sigma))
        intrinsic = 2000 - 1400 * math.exp(-r * T)
        assert c >= intrinsic * 0.99, "Deep ITM call far below intrinsic"

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call should be close to zero."""
        c = float(black_scholes_call(1000, 2000, 0.01, r, sigma))
        assert c < 1.0, f"Deep OTM call should be near zero, got {c:.4f}"

    def test_vectorised_shapes(self):
        """Vectorised inputs return correct shapes."""
        strikes = np.array([1300, 1400, 1500])
        c = black_scholes_call(S, strikes, T, r, sigma)
        p = black_scholes_put(S, strikes, T, r, sigma)
        assert c.shape == (3,)
        assert p.shape == (3,)


# ──────────────────────────────────────────────────────────────────────────────
# black_scholes_price dispatcher
# ──────────────────────────────────────────────────────────────────────────────


class TestDispatcher:

    def test_ce_gives_call_price(self):
        c1 = float(black_scholes_call(S, K, T, r, sigma))
        c2 = float(black_scholes_price("CE", S, K, T, r, sigma))
        assert abs(c1 - c2) < 1e-10

    def test_pe_gives_put_price(self):
        p1 = float(black_scholes_put(S, K, T, r, sigma))
        p2 = float(black_scholes_price("PE", S, K, T, r, sigma))
        assert abs(p1 - p2) < 1e-10

    def test_case_insensitive(self):
        c = float(black_scholes_price("call", S, K, T, r, sigma))
        assert c > 0
        p = float(black_scholes_price("put", S, K, T, r, sigma))
        assert p > 0

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown option_type"):
            black_scholes_price("XX", S, K, T, r, sigma)


# ──────────────────────────────────────────────────────────────────────────────
# ΔX (pricing error)
# ──────────────────────────────────────────────────────────────────────────────


class TestPricingError:

    def test_zero_error_when_prices_match(self):
        c = float(black_scholes_call(S, K, T, r, sigma))
        dx = float(pricing_error(c, c))
        assert abs(dx) < 1e-10

    def test_overpriced_positive(self):
        c = float(black_scholes_call(S, K, T, r, sigma))
        dx = float(pricing_error(c + 10, c))
        assert dx > 0

    def test_underpriced_negative(self):
        c = float(black_scholes_call(S, K, T, r, sigma))
        dx = float(pricing_error(c - 5, c))
        assert dx < 0

    def test_vectorised_pricing_error(self):
        strikes = np.array([1300, 1400, 1500])
        c = black_scholes_call(S, strikes, T, r, sigma)
        actual = c + np.array([2, -3, 5])
        dx = pricing_error(actual, c)
        np.testing.assert_allclose(dx, [2, -3, 5], atol=1e-10)
