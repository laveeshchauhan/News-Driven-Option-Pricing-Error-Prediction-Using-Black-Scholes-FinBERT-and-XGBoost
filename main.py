#!/usr/bin/env python3
"""
main.py — CLI Entry Point for RELIANCE Option Pricing (Phases 1, 2 & 3).

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

# Phase 3 — XGBoost ΔX prediction model:
    python main.py --phase 3

# Phase 3 — with charts:
    python main.py --phase 3 --plot

# Run all three phases end-to-end:
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
    ML_INPUT_CSV,
    ML_MODEL_PATH,
    ML_PREDICTIONS_CSV,
    ML_TEST_SIZE,
    NEWS_LOOKBACK_DAYS,
    SENTIMENT_OUTPUT_CSV,
    XGBOOST_COLSAMPLE_BYTREE,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_MIN_CHILD_WEIGHT,
    XGBOOST_N_ESTIMATORS,
    XGBOOST_REG_ALPHA,
    XGBOOST_REG_LAMBDA,
    XGBOOST_SUBSAMPLE,
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
        choices=["1", "2", "3", "all"],
        metavar="PHASE",
        help=(
            "Which pipeline phase(s) to run. "
            "  '1'   — Phase 1: Black-Scholes pricing (default). "
            "  '2'   — Phase 2: FinBERT sentiment analysis. "
            "  '3'   — Phase 3: XGBoost ΔX prediction model. "
            "  'all' — Run Phases 1 → 2 → 3 end-to-end."
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

    # ── Phase 3 parameters ─────────────────────────────────────────────────
    p3 = parser.add_argument_group("Phase 3 — XGBoost ΔX prediction parameters")
    p3.add_argument(
        "--ml-input-csv",
        default=ML_INPUT_CSV,
        metavar="PATH",
        help=(
            f"Input CSV for Phase 3 — use Phase 2 output when available "
            f"(default: {ML_INPUT_CSV})."
        ),
    )
    p3.add_argument(
        "--ml-model-path",
        default=ML_MODEL_PATH,
        metavar="PATH",
        help=f"Where to save the trained XGBoost model (default: {ML_MODEL_PATH}).",
    )
    p3.add_argument(
        "--ml-predictions-csv",
        default=ML_PREDICTIONS_CSV,
        metavar="PATH",
        help=f"Phase 3 predictions output CSV (default: {ML_PREDICTIONS_CSV}).",
    )
    p3.add_argument(
        "--ml-test-size",
        type=float,
        default=ML_TEST_SIZE,
        metavar="FRACTION",
        help=f"Fraction of data held out for testing (default: {ML_TEST_SIZE}).",
    )
    p3.add_argument(
        "--xgb-n-estimators",
        type=int,
        default=XGBOOST_N_ESTIMATORS,
        metavar="N",
        help=f"XGBoost number of boosting rounds (default: {XGBOOST_N_ESTIMATORS}).",
    )
    p3.add_argument(
        "--xgb-max-depth",
        type=int,
        default=XGBOOST_MAX_DEPTH,
        metavar="N",
        help=f"XGBoost max tree depth (default: {XGBOOST_MAX_DEPTH}).",
    )
    p3.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=XGBOOST_LEARNING_RATE,
        metavar="LR",
        help=f"XGBoost learning rate (default: {XGBOOST_LEARNING_RATE}).",
    )
    p3.add_argument(
        "--xgb-subsample",
        type=float,
        default=XGBOOST_SUBSAMPLE,
        metavar="RATIO",
        help=f"XGBoost row sub-sampling ratio (default: {XGBOOST_SUBSAMPLE}).",
    )
    p3.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=XGBOOST_COLSAMPLE_BYTREE,
        metavar="RATIO",
        help=f"XGBoost feature sub-sampling ratio (default: {XGBOOST_COLSAMPLE_BYTREE}).",
    )
    p3.add_argument(
        "--xgb-min-child-weight",
        type=int,
        default=XGBOOST_MIN_CHILD_WEIGHT,
        metavar="N",
        help=f"XGBoost min child weight (default: {XGBOOST_MIN_CHILD_WEIGHT}).",
    )
    p3.add_argument(
        "--xgb-reg-alpha",
        type=float,
        default=XGBOOST_REG_ALPHA,
        metavar="ALPHA",
        help=f"XGBoost L1 regularisation (default: {XGBOOST_REG_ALPHA}).",
    )
    p3.add_argument(
        "--xgb-reg-lambda",
        type=float,
        default=XGBOOST_REG_LAMBDA,
        metavar="LAMBDA",
        help=f"XGBoost L2 regularisation (default: {XGBOOST_REG_LAMBDA}).",
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


def _run_phase3(args, verbose: bool) -> int:
    """Execute Phase 3 pipeline. Returns exit code."""
    try:
        from src.ml_pipeline import run_ml_pipeline
    except ImportError as exc:
        print(
            f"ERROR: Phase 3 dependencies not installed: {exc}\n"
            "Run: pip install xgboost>=2.1.0 scikit-learn>=1.5.0",
            file=sys.stderr,
        )
        return 1

    try:
        run_ml_pipeline(
            input_csv=args.ml_input_csv,
            model_path=args.ml_model_path,
            predictions_csv=args.ml_predictions_csv,
            test_size=args.ml_test_size,
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            min_child_weight=args.xgb_min_child_weight,
            reg_alpha=args.xgb_reg_alpha,
            reg_lambda=args.xgb_reg_lambda,
            plot=args.plot,
            verbose=verbose,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Tip: Run Phase 1 first (python main.py --demo) to generate the input CSV.",
            file=sys.stderr,
        )
        return 1
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

    if phase == "3":
        return _run_phase3(args, verbose)

    # "all" — run all three phases sequentially
    if phase == "all":
        if verbose:
            print("Running Phase 1 (Black-Scholes)...")
        rc = _run_phase1(args, verbose)
        if rc != 0:
            return rc
        if verbose:
            print("\nRunning Phase 2 (Sentiment Analysis)...")
        rc = _run_phase2(args, verbose)
        if rc != 0:
            return rc
        if verbose:
            print("\nRunning Phase 3 (XGBoost ΔX Model)...")
        return _run_phase3(args, verbose)

    parser.error(f"Unknown --phase value: {args.phase}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
