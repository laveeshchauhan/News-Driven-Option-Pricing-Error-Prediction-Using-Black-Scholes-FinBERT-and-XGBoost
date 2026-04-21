#!/usr/bin/env python3
"""
main.py — CLI Entry Point for RELIANCE Option Pricing (Phases 1 & 2).

Usage examples
--------------

# Phase 1 — Black-Scholes pricing with bundled sample data:
    python main.py --demo

# Phase 1 — with your own NSE option chain CSV:
    python main.py --input data/raw/nse_option_chain_mar2026.csv

# Phase 1 — custom parameters and charts:
    python main.py --demo --risk-free-rate 0.072 --volatility-window 60 --plot

# Phase 2 — FinBERT sentiment analysis (demo, no API key needed):
    python main.py --phase 2 --demo

# Phase 2 — live RSS news + FinBERT:
    python main.py --phase 2

# Phase 2 — with NewsAPI key:
    NEWSAPI_KEY=your_key python main.py --phase 2

# Run both phases end-to-end:
    python main.py --phase all --demo

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
    FINBERT_BATCH_SIZE,
    FINBERT_MODEL_NAME,
    NEWS_LOOKBACK_DAYS,
    SENTIMENT_OUTPUT_CSV,
)
from src.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reliance-option-pricing",
        description=(
            "Black-Scholes Option Pricing + FinBERT Sentiment Analysis\n"
            "for RELIANCE Industries (NSE India).\n"
            "All prices are in ₹ (Indian Rupees / INR)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Phase selection ────────────────────────────────────────────────────
    parser.add_argument(
        "--phase",
        default="1",
        choices=["1", "2", "all"],
        metavar="PHASE",
        help=(
            "Which pipeline phase(s) to run. "
            "  '1'   — Phase 1: Black-Scholes pricing (default). "
            "  '2'   — Phase 2: FinBERT sentiment analysis. "
            "  'all' — Run Phase 1 then Phase 2 and merge outputs."
        ),
    )

    # ── Input ──────────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input", "-i",
        metavar="PATH",
        help=(
            "Path to one or more NSE option chain CSV file(s). "
            "Supports glob patterns, e.g. 'data/raw/*.csv'. "
            "(Phase 1 only)"
        ),
    )
    input_group.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help=(
            "Run with synthetically generated sample data (no CSV or API key "
            "required). Works for both Phase 1 and Phase 2."
        ),
    )

    # ── Phase 1 parameters ─────────────────────────────────────────────────
    p1 = parser.add_argument_group("Phase 1 — Black-Scholes parameters")
    p1.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        metavar="TICKER",
        help=(
            f"Yahoo Finance ticker symbol for historical volatility download "
            f"(default: {DEFAULT_TICKER})."
        ),
    )
    p1.add_argument(
        "--risk-free-rate", "-r",
        type=float,
        default=DEFAULT_RISK_FREE_RATE,
        metavar="RATE",
        help=(
            f"Annualised risk-free rate as a decimal (default: {DEFAULT_RISK_FREE_RATE} "
            f"= {DEFAULT_RISK_FREE_RATE*100:.1f}%%, Indian 91-day T-Bill rate)."
        ),
    )
    p1.add_argument(
        "--volatility-window", "-w",
        type=int,
        default=DEFAULT_VOLATILITY_WINDOW,
        metavar="DAYS",
        help=(
            f"Rolling window in trading days for historical volatility calculation "
            f"(default: {DEFAULT_VOLATILITY_WINDOW})."
        ),
    )
    p1.add_argument(
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
    p1.add_argument(
        "--output-csv", "-o",
        default=DEFAULT_OUTPUT_CSV,
        metavar="PATH",
        help=f"Phase 1 output CSV file path (default: {DEFAULT_OUTPUT_CSV}).",
    )

    # ── Phase 2 parameters ─────────────────────────────────────────────────
    p2 = parser.add_argument_group("Phase 2 — Sentiment analysis parameters")
    p2.add_argument(
        "--newsapi-key",
        default=None,
        metavar="KEY",
        help=(
            "NewsAPI.org developer key. If omitted, reads the NEWSAPI_KEY "
            "environment variable. Falls back to free RSS feeds when not set."
        ),
    )
    p2.add_argument(
        "--lookback-days",
        type=int,
        default=NEWS_LOOKBACK_DAYS,
        metavar="DAYS",
        help=(
            f"Number of calendar days back to fetch news articles "
            f"(default: {NEWS_LOOKBACK_DAYS})."
        ),
    )
    p2.add_argument(
        "--sentiment-model",
        default=FINBERT_MODEL_NAME,
        metavar="MODEL",
        help=(
            f"HuggingFace model name for sentiment classification "
            f"(default: {FINBERT_MODEL_NAME})."
        ),
    )
    p2.add_argument(
        "--sentiment-batch-size",
        type=int,
        default=FINBERT_BATCH_SIZE,
        metavar="N",
        help=f"Batch size for FinBERT inference (default: {FINBERT_BATCH_SIZE}).",
    )
    p2.add_argument(
        "--sentiment-output-csv",
        default=SENTIMENT_OUTPUT_CSV,
        metavar="PATH",
        help=f"Phase 2 output CSV path (default: {SENTIMENT_OUTPUT_CSV}).",
    )
    p2.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable reading/writing news and sentiment caches.",
    )

    # ── Shared output options ──────────────────────────────────────────────
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


def _run_phase1(args, verbose: bool) -> int:
    """Execute Phase 1 pipeline. Returns exit code."""
    if not (0.0 <= args.risk_free_rate <= 1.0):
        print(
            f"ERROR: --risk-free-rate must be between 0 and 1 "
            f"(got {args.risk_free_rate}). Example: use 0.07 for 7%, not 7.",
            file=sys.stderr,
        )
        return 1
    if args.volatility_window < 5:
        print(
            "ERROR: --volatility-window must be at least 5 trading days.",
            file=sys.stderr,
        )
        return 1
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
            verbose=verbose,
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


def _run_phase2(args, verbose: bool) -> int:
    """Execute Phase 2 pipeline. Returns exit code."""
    try:
        from src.sentiment_pipeline import run_sentiment_pipeline
    except ImportError as exc:
        print(
            f"ERROR: Phase 2 dependencies not installed: {exc}\n"
            "Run: pip install transformers torch feedparser textblob",
            file=sys.stderr,
        )
        return 1

    try:
        run_sentiment_pipeline(
            phase1_csv=args.output_csv,
            demo=args.demo,
            newsapi_key=args.newsapi_key,
            lookback_days=args.lookback_days,
            model_name=args.sentiment_model,
            batch_size=args.sentiment_batch_size,
            output_csv=args.sentiment_output_csv,
            use_cache=not args.no_cache,
            plot=args.plot,
            verbose=verbose,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    verbose = not args.quiet

    phase = args.phase.lower()

    if phase == "1":
        return _run_phase1(args, verbose)

    if phase == "2":
        return _run_phase2(args, verbose)

    # "all" — run both phases sequentially
    if phase == "all":
        if verbose:
            print("Running Phase 1 (Black-Scholes)...")
        rc = _run_phase1(args, verbose)
        if rc != 0:
            return rc
        if verbose:
            print("\nRunning Phase 2 (Sentiment Analysis)...")
        return _run_phase2(args, verbose)

    parser.error(f"Unknown --phase value: {args.phase}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
