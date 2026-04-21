"""
Configuration constants for the Reliance Option Pricing project.

All monetary values are in ₹ (Indian Rupees / INR).
"""

# ─────────────────────────────────────────────
# Market / Financial Constants
# ─────────────────────────────────────────────

# Default risk-free rate — Indian 91-day T-Bill rate (annualized)
DEFAULT_RISK_FREE_RATE: float = 0.07

# Number of trading days per year used for volatility annualisation
TRADING_DAYS_PER_YEAR: int = 252

# Number of calendar days per year used for time-to-expiry calculation
CALENDAR_DAYS_PER_YEAR: int = 365

# Default rolling window (in trading days) for historical volatility
DEFAULT_VOLATILITY_WINDOW: int = 30

# Default dividend yield — zero for short-term options where no ex-div date
# falls within the option's lifetime
DEFAULT_DIVIDEND_YIELD: float = 0.0

# ─────────────────────────────────────────────
# Ticker / Asset Configuration
# ─────────────────────────────────────────────

DEFAULT_TICKER: str = "RELIANCE.NS"
DEFAULT_SPOT_PRICE: float = 1390.20   # ₹ — from NSE screenshot dated 11-Mar-2026

# ─────────────────────────────────────────────
# NSE Option Chain Column Mappings
# ─────────────────────────────────────────────
# Maps raw NSE CSV headers → internal (snake_case) column names

NSE_COLUMN_MAP: dict = {
    "SYMBOL": "symbol",
    "DATE": "date",
    "EXPIRY": "expiry",
    "OPTION TYPE": "option_type",
    "STRIKE PRICE": "strike_price",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "LTP": "ltp",
    "SETTLE PRICE": "settle_price",
    "NO. OF CONTRACTS": "contracts",
    "TURNOVER IN ₹ LAKHS": "turnover_lakhs",
    "PREMIUM TURNOVER IN ₹ LAKHS": "premium_turnover_lakhs",
    "OPEN INT": "open_interest",
    "CHANGE IN OI": "change_in_oi",
    "UNDERLYING VALUE": "underlying_value",
}

# The column in the dataset that holds the actual market premium to compare
# against the Black-Scholes theoretical price (used to compute ΔX)
ACTUAL_PREMIUM_COLUMN: str = "ltp"

# ─────────────────────────────────────────────
# File / Directory Paths
# ─────────────────────────────────────────────

DATA_DIR: str = "data"
SAMPLE_DATA_DIR: str = "data/sample"
SAMPLE_OPTION_CHAIN_PATH: str = "data/sample/sample_option_chain.csv"
OUTPUT_DIR: str = "outputs"
DEFAULT_OUTPUT_CSV: str = "outputs/results.csv"

# ─────────────────────────────────────────────
# Pipeline Behaviour
# ─────────────────────────────────────────────

# Minimum time-to-expiry (years) to avoid division-by-zero in BS formulae
MIN_TIME_TO_EXPIRY: float = 1e-6

# Minimum volatility to avoid division-by-zero
MIN_VOLATILITY: float = 1e-6

# Period (in years) of historical data to download for volatility estimation
VOLATILITY_HISTORY_YEARS: int = 1

# ─────────────────────────────────────────────
# Phase 2 — NLP Sentiment Analysis
# ─────────────────────────────────────────────

# HuggingFace model name for financial sentiment classification
FINBERT_MODEL_NAME: str = "ProsusAI/finbert"

# Keywords used to filter / search news articles about RELIANCE
NEWS_KEYWORDS: list = [
    "Reliance Industries",
    "RELIANCE.NS",
    "RIL",
    "Jio",
    "Mukesh Ambani",
]

# Number of calendar days to look back when fetching news
NEWS_LOOKBACK_DAYS: int = 90

# RSS feed URLs (free, no API key required) — used as the default news source
NEWS_RSS_FEEDS: list = [
    "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "https://feeds.feedburner.com/ndtvprofit-latest",
]

# Paths for Phase 2 data/output
NEWS_DATA_DIR: str = "data/news"
RAW_NEWS_CACHE_PATH: str = "data/news/raw_news.csv"
SENTIMENT_CACHE_PATH: str = "data/news/sentiment_scores.csv"
SENTIMENT_OUTPUT_CSV: str = "outputs/results_with_sentiment.csv"

# Threshold (absolute) for ΔX-based option classification — also used in Phase 2
SENTIMENT_NEUTRAL_BAND: float = 0.05  # scores within ±0.05 of zero → "Neutral"

# Maximum number of articles to process per run (0 = no limit)
NEWS_MAX_ARTICLES: int = 0

# Batch size for FinBERT inference
FINBERT_BATCH_SIZE: int = 16

# ─────────────────────────────────────────────
# Phase 3 — XGBoost ΔX Prediction Model
# ─────────────────────────────────────────────

# Input CSV for Phase 3 (Phase 2 enriched output; falls back to Phase 1 output)
ML_INPUT_CSV: str = "outputs/results_with_sentiment.csv"

# Path to persist the trained XGBoost model
ML_MODEL_PATH: str = "outputs/xgb_model.json"

# Path for Phase 3 predictions CSV
ML_PREDICTIONS_CSV: str = "outputs/predictions.csv"

# Fraction of data held out for the test set
ML_TEST_SIZE: float = 0.2

# Random seed for reproducibility
ML_RANDOM_STATE: int = 42

# XGBoost hyper-parameters (sensible defaults; tunable via CLI)
XGBOOST_N_ESTIMATORS: int = 300
XGBOOST_MAX_DEPTH: int = 6
XGBOOST_LEARNING_RATE: float = 0.05
XGBOOST_SUBSAMPLE: float = 0.8
XGBOOST_COLSAMPLE_BYTREE: float = 0.8
XGBOOST_MIN_CHILD_WEIGHT: int = 3
XGBOOST_REG_ALPHA: float = 0.1    # L1 regularisation
XGBOOST_REG_LAMBDA: float = 1.0   # L2 regularisation

# Feature columns used for XGBoost training
# option_type_enc is derived (CE→1, PE→0) and moneyness = underlying_value / strike_price
ML_FEATURE_COLUMNS: list = [
    "strike_price",
    "underlying_value",
    "time_to_expiry",
    "volatility",
    "risk_free_rate",
    "bs_price",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "moneyness",            # engineered: S / K
    "option_type_enc",      # engineered: CE→1, PE→0
    "daily_sentiment_score",
    "daily_pos_mean",
    "daily_neg_mean",
    "daily_article_count",
]

# Target column
ML_TARGET_COLUMN: str = "delta_x"

# ─────────────────────────────────────────────
# Phase 4 — Live Inference Pipeline
# ─────────────────────────────────────────────

# Output CSV for Phase 4 live predictions
INFERENCE_OUTPUT_CSV: str = "outputs/live_predictions.csv"

# Trading signal thresholds (₹).
# BUY  signal when predicted ΔX < -INFERENCE_SIGNAL_BUY_THRESHOLD  (underpriced)
# SELL signal when predicted ΔX >  INFERENCE_SIGNAL_SELL_THRESHOLD (overpriced)
# HOLD otherwise (fairly priced)
INFERENCE_SIGNAL_BUY_THRESHOLD: float = 2.0
INFERENCE_SIGNAL_SELL_THRESHOLD: float = 2.0
