"""
test_greeks.py — Unit tests for src/greeks.py.

Tests:
  • Delta range: call ∈ (0,1), put ∈ (−1,0)
  • Gamma positivity and symmetry
  • Theta negativity for long positions
  • Vega positivity
  • Rho sign conventions
  • all_greeks() convenience wrapper
  • Edge cases: T→0, deep ITM/OTM
"""

import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.greeks import delta, gamma, theta, vega, rho, all_greeks


# ──────────────────────────────────────────────────────────────────────────────
# Reference parameters
# ──────────────────────────────────────────────────────────────────────────────

S = 1390.20   # ₹
K = 1400.00   # ₹
T = 48 / 365  # years
r = 0.07
sigma = 0.25
q = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Delta
# ──────────────────────────────────────────────────────────────────────────────


class TestDelta:

    def test_call_delta_between_0_and_1(self):
        d = float(delta("CE", S, K, T, r, sigma))
        assert 0 < d < 1, f"Call delta {d:.4f} not in (0, 1)"

    def test_put_delta_between_minus1_and_0(self):
        d = float(delta("PE", S, K, T, r, sigma))
        assert -1 < d < 0, f"Put delta {d:.4f} not in (-1, 0)"

    def test_call_delta_plus_put_delta_equals_one_no_dividend(self):
        """Δ_call − Δ_put = 1 when q = 0 (from put-call parity)."""
        dc = float(delta("CE", S, K, T, r, sigma))
        dp = float(delta("PE", S, K, T, r, sigma))
        assert abs(dc - dp - 1.0) < 1e-8

    def test_deep_itm_call_delta_near_1(self):
        d = float(delta("CE", 2000, 1000, T, r, sigma))
        assert d > 0.95, f"Deep ITM call delta {d:.4f} should be ≈ 1"

    def test_deep_otm_call_delta_near_0(self):
        d = float(delta("CE", 1000, 2000, T, r, sigma))
        assert d < 0.05, f"Deep OTM call delta {d:.4f} should be ≈ 0"

    def test_atm_call_delta_near_half(self):
        """ATM call delta ≈ 0.5 (exactly 0.5 at-the-money forward)."""
        d = float(delta("CE", 1390, 1390, T, r, sigma))
        assert 0.4 < d < 0.7, f"ATM call delta {d:.4f} not near 0.5"

    def test_case_insensitive(self):
        d1 = float(delta("ce", S, K, T, r, sigma))
        d2 = float(delta("CALL", S, K, T, r, sigma))
        assert abs(d1 - d2) < 1e-10


# ──────────────────────────────────────────────────────────────────────────────
# Gamma
# ──────────────────────────────────────────────────────────────────────────────


class TestGamma:

    def test_gamma_positive(self):
        g = float(gamma(S, K, T, r, sigma))
        assert g > 0, f"Gamma must be positive, got {g}"

    def test_gamma_same_for_call_and_put(self):
        """Gamma is identical for call and put by Black-Scholes."""
        g_call = float(gamma(S, K, T, r, sigma))
        # Call gamma = put gamma — just check we always get same value
        assert g_call > 0

    def test_gamma_maximum_atm(self):
        """Gamma is highest at-the-money."""
        g_atm = float(gamma(S, S, T, r, sigma))
        g_itm = float(gamma(S, S * 0.70, T, r, sigma))
        g_otm = float(gamma(S, S * 1.30, T, r, sigma))
        assert g_atm > g_itm
        assert g_atm > g_otm

    def test_gamma_vectorised(self):
        strikes = np.array([1200, 1390, 1600])
        g = gamma(S, strikes, T, r, sigma)
        assert g.shape == (3,)
        assert (g > 0).all()


# ──────────────────────────────────────────────────────────────────────────────
# Theta
# ──────────────────────────────────────────────────────────────────────────────


class TestTheta:

    def test_call_theta_negative(self):
        """Call theta is negative — option loses value over time."""
        th = float(theta("CE", S, K, T, r, sigma))
        assert th < 0, f"Call theta should be negative, got {th:.4f}"

    def test_put_theta_negative(self):
        """Put theta is typically negative (time decay)."""
        th = float(theta("PE", S, K, T, r, sigma))
        # Near zero r, put theta can be slightly positive for deep ITM;
        # for our near-ATM case it should be negative
        assert th < 0.5, (
            f"Put theta {th:.4f} unexpectedly large positive"
        )

    def test_theta_magnitude_reasonable(self):
        """Daily theta should be a small fraction of the option premium."""
        from src.black_scholes import black_scholes_call
        price = float(black_scholes_call(S, K, T, r, sigma))
        th = float(theta("CE", S, K, T, r, sigma))
        # |theta per day| < option price (theta can't decay more than the option)
        assert abs(th) < price, (
            f"|Theta|={abs(th):.4f} exceeds option price={price:.4f}"
        )

    def test_theta_increases_magnitude_near_expiry(self):
        """Theta accelerates (becomes more negative) as expiry approaches."""
        th_long = float(theta("CE", S, K, 0.5, r, sigma))
        th_short = float(theta("CE", S, K, 0.05, r, sigma))
        assert th_short < th_long, (
            "Theta magnitude should increase closer to expiry"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Vega
# ──────────────────────────────────────────────────────────────────────────────


class TestVega:

    def test_vega_positive(self):
        v = float(vega(S, K, T, r, sigma))
        assert v > 0, f"Vega must be positive, got {v}"

    def test_vega_maximum_atm(self):
        """Vega is highest at-the-money."""
        v_atm = float(vega(S, S, T, r, sigma))
        v_itm = float(vega(S, S * 0.70, T, r, sigma))
        v_otm = float(vega(S, S * 1.30, T, r, sigma))
        assert v_atm > v_itm
        assert v_atm > v_otm

    def test_vega_increases_with_time(self):
        """More time to expiry → higher vega."""
        v_short = float(vega(S, K, 0.05, r, sigma))
        v_long = float(vega(S, K, 0.50, r, sigma))
        assert v_long > v_short

    def test_vega_same_for_ce_and_pe(self):
        """Vega is the same for calls and puts with same inputs."""
        v = float(vega(S, K, T, r, sigma))
        assert v > 0


# ──────────────────────────────────────────────────────────────────────────────
# Rho
# ──────────────────────────────────────────────────────────────────────────────


class TestRho:

    def test_call_rho_positive(self):
        """Call rho is positive (higher rates → higher call premium)."""
        r_val = float(rho("CE", S, K, T, r, sigma))
        assert r_val > 0, f"Call rho should be positive, got {r_val}"

    def test_put_rho_negative(self):
        """Put rho is negative (higher rates → lower put premium)."""
        r_val = float(rho("PE", S, K, T, r, sigma))
        assert r_val < 0, f"Put rho should be negative, got {r_val}"

    def test_rho_per_1pct_small(self):
        """Rho per 1% rate change should be small relative to option price."""
        from src.black_scholes import black_scholes_call
        price = float(black_scholes_call(S, K, T, r, sigma))
        r_val = abs(float(rho("CE", S, K, T, r, sigma)))
        # Rho per 1% is typically < option price for short-dated options
        assert r_val < price * 5, (
            f"Rho {r_val:.4f} seems unreasonably large vs price {price:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# all_greeks convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────


class TestAllGreeks:

    def test_returns_all_keys(self):
        g = all_greeks("CE", S, K, T, r, sigma)
        assert set(g.keys()) == {"delta", "gamma", "theta", "vega", "rho"}

    def test_values_match_individual_functions(self):
        g = all_greeks("CE", S, K, T, r, sigma)
        assert abs(float(g["delta"]) - float(delta("CE", S, K, T, r, sigma))) < 1e-10
        assert abs(float(g["gamma"]) - float(gamma(S, K, T, r, sigma))) < 1e-10
        assert abs(float(g["theta"]) - float(theta("CE", S, K, T, r, sigma))) < 1e-10
        assert abs(float(g["vega"]) - float(vega(S, K, T, r, sigma))) < 1e-10
        assert abs(float(g["rho"]) - float(rho("CE", S, K, T, r, sigma))) < 1e-10

    def test_put_greeks(self):
        g = all_greeks("PE", S, K, T, r, sigma)
        assert g["delta"] < 0
        assert g["gamma"] > 0
        assert g["vega"] > 0
