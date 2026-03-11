#!/usr/bin/env python3
"""
main.py — CLI Entry Point for Phase 1: Black-Scholes Option Pricing for RELIANCE.

Usage examples
--------------

# Run with bundled sample data:
    python main.py --demo

# Run with your own NSE option chain CSV:
    python main.py --input data/raw/nse_option_chain_mar2026.csv

# Run with custom parameters and generate plots:
    python main.py --demo --risk-free-rate 0.072 --volatility-window 60 --plot

# Specify output path:
    python main.py --input data/raw/*.csv --output-csv outputs/my_results.csv
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure project root is on sys.path so `config` and `src` are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_OUTPUT_CSV,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_TICKER,
    DEFAULT_VOLATILITY_WINDOW,
    DEFAULT_DIVIDEND_YIELD,
)
from src.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reliance-option-pricing",
        description=(
            "Phase 1 — Black-Scholes Option Pricing for RELIANCE (NSE India).\n"
            "All prices are in ₹ (Indian Rupees / INR)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input ──────────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input", "-i",
        metavar="PATH",
        help=(
            "Path to one or more NSE option chain CSV file(s). "
            "Supports glob patterns, e.g. 'data/raw/*.csv'."
        ),
    )
    input_group.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run with synthetically generated sample data (no CSV required).",
    )

    # ── Model parameters ───────────────────────────────────────────────────
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        metavar="TICKER",
        help=(
            f"Yahoo Finance ticker symbol for historical volatility download "
            f"(default: {DEFAULT_TICKER})."
        ),
    )
    parser.add_argument(
        "--risk-free-rate", "-r",
        type=float,
        default=DEFAULT_RISK_FREE_RATE,
        metavar="RATE",
        help=(
            f"Annualised risk-free rate as a decimal (default: {DEFAULT_RISK_FREE_RATE} "
            f"= {DEFAULT_RISK_FREE_RATE*100:.1f}%%, Indian 91-day T-Bill rate)."
        ),
    )
    parser.add_argument(
        "--volatility-window", "-w",
        type=int,
        default=DEFAULT_VOLATILITY_WINDOW,
        metavar="DAYS",
        help=(
            f"Rolling window in trading days for historical volatility calculation "
            f"(default: {DEFAULT_VOLATILITY_WINDOW})."
        ),
    )
    parser.add_argument(
        "--dividend-yield", "-q",
        type=float,
        default=DEFAULT_DIVIDEND_YIELD,
        metavar="YIELD",
        help=(
            f"Continuous dividend yield (default: {DEFAULT_DIVIDEND_YIELD}). "
            "For short-term RELIANCE options with no ex-dividend date before expiry, "
            "the correct value is 0."
        ),
    )

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-csv", "-o",
        default=DEFAULT_OUTPUT_CSV,
        metavar="PATH",
        help=f"Output CSV file path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate and save visualisation charts to the outputs/ directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress and summary output.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate risk-free rate
    if not (0.0 <= args.risk_free_rate <= 1.0):
        parser.error(
            f"--risk-free-rate must be between 0 and 1 (got {args.risk_free_rate}). "
            "Example: use 0.07 for 7%, not 7."
        )

    # Validate volatility window
    if args.volatility_window < 5:
        parser.error("--volatility-window must be at least 5 trading days.")

    try:
        run_pipeline(
            input_path=args.input,
            demo=args.demo,
            ticker=args.ticker,
            risk_free_rate=args.risk_free_rate,
            volatility_window=args.volatility_window,
            output_csv=args.output_csv,
            dividend_yield=args.dividend_yield,
            plot=args.plot,
            verbose=not args.quiet,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Tip: Use --demo to run with bundled sample data, or "
            "provide a valid --input path.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
