"""
black_scholes.py — Black-Scholes Pricing Engine for European Call & Put Options.

All monetary values are in ₹ (Indian Rupees / INR).

Formulae (without dividend yield, q = 0 by default):

    d₁ = [ln(S/K) + (r + σ²/2) · T] / (σ · √T)
    d₂ = d₁ − σ · √T

    Call:  C = S · N(d₁) − K · e^(−rT) · N(d₂)
    Put:   P = K · e^(−rT) · N(−d₂) − S · N(−d₁)

With optional continuous dividend yield q ≠ 0:

    d₁ = [ln(S/K) + (r − q + σ²/2) · T] / (σ · √T)
    Call:  C = S · e^(−qT) · N(d₁) − K · e^(−rT) · N(d₂)
    Put:   P = K · e^(−rT) · N(−d₂) − S · e^(−qT) · N(−d₁)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from config import MIN_TIME_TO_EXPIRY, MIN_VOLATILITY


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _safe_array(*arrays: float | np.ndarray) -> tuple[np.ndarray, ...]:
    """Convert all inputs to float64 numpy arrays of the same shape."""
    return tuple(np.atleast_1d(np.asarray(a, dtype=float)) for a in arrays)


def _d1_d2(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate d₁ and d₂ for the Black-Scholes formula.

    Parameters
    ----------
    S     : Spot price (₹)
    K     : Strike price (₹)
    T     : Time to expiry in years (clamped to MIN_TIME_TO_EXPIRY)
    r     : Risk-free rate (annualised, e.g. 0.07)
    sigma : Volatility (annualised, e.g. 0.25)
    q     : Continuous dividend yield (default 0)

    Returns
    -------
    (d1, d2) as numpy arrays
    """
    T_safe = np.maximum(T, MIN_TIME_TO_EXPIRY)
    sigma_safe = np.maximum(sigma, MIN_VOLATILITY)

    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_safe**2) * T_safe) / (
        sigma_safe * sqrt_T
    )
    d2 = d1 - sigma_safe * sqrt_T
    return d1, d2


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def black_scholes_call(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """
    Black-Scholes price for a **European Call** option.

    Parameters
    ----------
    S     : Spot / underlying price (₹)
    K     : Strike price (₹)
    T     : Time to expiry in years (e.g. 48/365 ≈ 0.1315)
    r     : Risk-free rate (annualised decimal, e.g. 0.07 for 7 %)
    sigma : Historical / implied volatility (annualised decimal, e.g. 0.25)
    q     : Continuous dividend yield (default 0.0 — not needed for short-term
            RELIANCE options where no ex-dividend date falls within the option
            lifetime)

    Returns
    -------
    numpy array of call prices (₹)

    Examples
    --------
    >>> float(black_scholes_call(1390.20, 1400, 48/365, 0.07, 0.25))
    34.65...  # approximate
    """
    S, K, T, r, sigma, q = _safe_array(S, K, T, r, sigma, q)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return np.squeeze(call)


def black_scholes_put(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """
    Black-Scholes price for a **European Put** option.

    Parameters
    ----------
    S     : Spot / underlying price (₹)
    K     : Strike price (₹)
    T     : Time to expiry in years
    r     : Risk-free rate (annualised decimal)
    sigma : Historical / implied volatility (annualised decimal)
    q     : Continuous dividend yield (default 0.0)

    Returns
    -------
    numpy array of put prices (₹)

    Examples
    --------
    >>> float(black_scholes_put(1390.20, 1400, 48/365, 0.07, 0.25))
    53.46...  # approximate
    """
    S, K, T, r, sigma, q = _safe_array(S, K, T, r, sigma, q)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return np.squeeze(put)


def black_scholes_price(
    option_type: str,
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> np.ndarray:
    """
    Dispatch to Call or Put pricing based on *option_type*.

    Parameters
    ----------
    option_type : ``'CE'`` or ``'C'`` for calls; ``'PE'`` or ``'P'`` for puts
                  (case-insensitive)
    S, K, T, r, sigma, q : see :func:`black_scholes_call`

    Returns
    -------
    numpy array of option prices (₹)

    Raises
    ------
    ValueError
        If *option_type* is not recognised.
    """
    otype = option_type.strip().upper()
    if otype in {"CE", "C", "CALL"}:
        return black_scholes_call(S, K, T, r, sigma, q)
    elif otype in {"PE", "P", "PUT"}:
        return black_scholes_put(S, K, T, r, sigma, q)
    else:
        raise ValueError(
            f"Unknown option_type '{option_type}'. "
            "Use 'CE'/'C'/'CALL' or 'PE'/'P'/'PUT'."
        )


def pricing_error(
    actual_premium: float | np.ndarray,
    bs_price: float | np.ndarray,
) -> np.ndarray:
    """
    Calculate ΔX = Actual Market Premium − Black-Scholes Theoretical Price.

    A positive ΔX means the option is **overpriced** by the market relative
    to the BS model; a negative ΔX means it is **underpriced**.

    This column becomes the **target variable** for Phase 3 (XGBoost).

    Parameters
    ----------
    actual_premium : Market LTP / settle price (₹)
    bs_price       : Theoretical price from Black-Scholes (₹)

    Returns
    -------
    numpy array of pricing errors (₹)
    """
    actual_premium, bs_price = _safe_array(actual_premium, bs_price)
    return np.squeeze(actual_premium - bs_price)
