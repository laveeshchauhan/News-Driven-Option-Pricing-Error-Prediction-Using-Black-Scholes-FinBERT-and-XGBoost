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
