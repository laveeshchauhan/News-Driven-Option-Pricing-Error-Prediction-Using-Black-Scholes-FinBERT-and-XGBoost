"""
greeks.py — Option Greeks Calculator.

Computes Delta, Gamma, Theta, Vega, and Rho for European Call and Put options
using the Black-Scholes framework.

All monetary values are in ₹ (INR).  Theta is expressed as daily decay (₹/day)
rather than annual to be more intuitive.

Formulae (continuous dividend yield q; default q = 0):

    d₁ = [ln(S/K) + (r − q + σ²/2) · T] / (σ · √T)
    d₂ = d₁ − σ · √T

    Delta_call  = e^(−qT) · N(d₁)
    Delta_put   = e^(−qT) · (N(d₁) − 1)

    Gamma       = e^(−qT) · n(d₁) / (S · σ · √T)    [same for C & P]

    Theta_call  = − [S · e^(−qT) · n(d₁) · σ] / (2√T)
                  − r · K · e^(−rT) · N(d₂)
                  + q · S · e^(−qT) · N(d₁)

    Theta_put   = − [S · e^(−qT) · n(d₁) · σ] / (2√T)
                  + r · K · e^(−rT) · N(−d₂)
                  − q · S · e^(−qT) · N(−d₁)

    Vega        = S · e^(−qT) · n(d₁) · √T           [same for C & P, per 1 %]

    Rho_call    = K · T · e^(−rT) · N(d₂)
    Rho_put     = −K · T · e^(−rT) · N(−d₂)

Theta and Rho above are *annual*; we divide by 365 to get per-calendar-day.
Vega above is per unit of σ; we divide by 100 to get per 1 % move in σ.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from config import MIN_TIME_TO_EXPIRY, MIN_VOLATILITY


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers (mirrors black_scholes.py — kept local to avoid circular dep)
# ──────────────────────────────────────────────────────────────────────────────


def _safe_array(*arrays: float | np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.atleast_1d(np.asarray(a, dtype=float)) for a in arrays)


def _d1_d2(S, K, T, r, sigma, q):
    T_safe = np.maximum(T, MIN_TIME_TO_EXPIRY)
    sigma_safe = np.maximum(sigma, MIN_VOLATILITY)
    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_safe**2) * T_safe) / (
        sigma_safe * sqrt_T
    )
    d2 = d1 - sigma_safe * sqrt_T
    return d1, d2, T_safe, sigma_safe, sqrt_T


# ──────────────────────────────────────────────────────────────────────────────
# Delta
# ──────────────────────────────────────────────────────────────────────────────


def delta(
    option_type: str,
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """
    Delta (Δ) — sensitivity of option price to a ₹1 move in the underlying.

    Parameters
    ----------
    option_type : ``'CE'``/``'C'`` for call; ``'PE'``/``'P'`` for put.
    S, K, T, r, sigma, q : standard BS inputs.

    Returns
    -------
    numpy array  in range (0, 1) for calls, (−1, 0) for puts.
    """
    S, K, T, r, sigma, q = _safe_array(S, K, T, r, sigma, q)
    d1, _, T_s, _, _ = _d1_d2(S, K, T, r, sigma, q)
    if option_type.strip().upper() in {"CE", "C", "CALL"}:
        result = np.exp(-q * T_s) * norm.cdf(d1)
    else:
        result = np.exp(-q * T_s) * (norm.cdf(d1) - 1)
    return np.squeeze(result)


# ──────────────────────────────────────────────────────────────────────────────
# Gamma
# ──────────────────────────────────────────────────────────────────────────────


def gamma(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """
    Gamma (Γ) — rate of change of Delta per ₹1 move in underlying (same for
    call and put).

    Returns
    -------
    numpy array of gamma values.
    """
    S, K, T, r, sigma, q = _safe_array(S, K, T, r, sigma, q)
    d1, _, T_s, sigma_s, sqrt_T = _d1_d2(S, K, T, r, sigma, q)
    result = np.exp(-q * T_s) * norm.pdf(d1) / (S * sigma_s * sqrt_T)
    return np.squeeze(result)


# ──────────────────────────────────────────────────────────────────────────────
# Theta
# ──────────────────────────────────────────────────────────────────────────────


def theta(
    option_type: str,
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """
    Theta (Θ) — time decay expressed as ₹ change per **calendar day**.

    Negative for long positions (option loses value as time passes).

    Parameters
    ----------
    option_type : ``'CE'``/``'C'`` for call; ``'PE'``/``'P'`` for put.

    Returns
    -------
    numpy array of daily theta values (₹/day).
    """
    S, K, T, r, sigma, q = _safe_array(S, K, T, r, sigma, q)
    d1, d2, T_s, sigma_s, sqrt_T = _d1_d2(S, K, T, r, sigma, q)

    common_term = (
        -(S * np.exp(-q * T_s) * norm.pdf(d1) * sigma_s) / (2 * sqrt_T)
    )

    otype = option_type.strip().upper()
    if otype in {"CE", "C", "CALL"}:
        annual_theta = (
            common_term
            - r * K * np.exp(-r * T_s) * norm.cdf(d2)
            + q * S * np.exp(-q * T_s) * norm.cdf(d1)
        )
    else:
        annual_theta = (
            common_term
            + r * K * np.exp(-r * T_s) * norm.cdf(-d2)
            - q * S * np.exp(-q * T_s) * norm.cdf(-d1)
        )

    # Convert from annual to per-calendar-day
    return np.squeeze(annual_theta / 365)


# ──────────────────────────────────────────────────────────────────────────────
# Vega
# ──────────────────────────────────────────────────────────────────────────────


def vega(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """
    Vega (ν) — sensitivity of option price to a **1 % change** in volatility
    (same for call and put).

    Returns
    -------
    numpy array of vega values (₹ per 1 % change in σ).
    """
    S, K, T, r, sigma, q = _safe_array(S, K, T, r, sigma, q)
    d1, _, T_s, _, sqrt_T = _d1_d2(S, K, T, r, sigma, q)
    # per-unit vega; divide by 100 to express per 1 %
    result = S * np.exp(-q * T_s) * norm.pdf(d1) * sqrt_T / 100
    return np.squeeze(result)


# ──────────────────────────────────────────────────────────────────────────────
# Rho
# ──────────────────────────────────────────────────────────────────────────────


def rho(
    option_type: str,
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """
    Rho (ρ) — sensitivity of option price to a **1 % change** in interest rate.

    Parameters
    ----------
    option_type : ``'CE'``/``'C'`` for call; ``'PE'``/``'P'`` for put.

    Returns
    -------
    numpy array of rho values (₹ per 1 % change in r).
    """
    S, K, T, r, sigma, q = _safe_array(S, K, T, r, sigma, q)
    d1, d2, T_s, _, _ = _d1_d2(S, K, T, r, sigma, q)

    otype = option_type.strip().upper()
    if otype in {"CE", "C", "CALL"}:
        result = K * T_s * np.exp(-r * T_s) * norm.cdf(d2) / 100
    else:
        result = -K * T_s * np.exp(-r * T_s) * norm.cdf(-d2) / 100

    return np.squeeze(result)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: all Greeks at once
# ──────────────────────────────────────────────────────────────────────────────


def all_greeks(
    option_type: str,
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> dict[str, np.ndarray]:
    """
    Return a dictionary with all five Greeks.

    Returns
    -------
    ``{'delta': ..., 'gamma': ..., 'theta': ..., 'vega': ..., 'rho': ...}``
    """
    return {
        "delta": delta(option_type, S, K, T, r, sigma, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "theta": theta(option_type, S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "rho": rho(option_type, S, K, T, r, sigma, q),
    }
