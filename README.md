# Reliance Option Pricing

**Black-Scholes Option Pricing & Sentiment-Driven ML for RELIANCE Industries (NSE India)**

> All prices are in **₹ (Indian Rupees / INR)** — this is NSE India market data, NOT USD.

---

## Project Overview

This project is a **5-phase quantitative finance system** for pricing **Reliance Industries (RELIANCE.NS)** options listed on NSE India:

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Black-Scholes Pricing Engine + Greeks | ✅ Complete |
| **Phase 2** | FinBERT NLP Sentiment Analysis | ✅ Complete |
| **Phase 3** | XGBoost ML Model (ΔX prediction) | ✅ Complete |
| **Phase 4** | Live Inference Pipeline | ✅ Complete |
| **Phase 5** | MLOps & Monitoring | ✅ Complete |

---

## Phase 5 — MLOps & Monitoring

Phase 5 adds lightweight, self-contained MLOps capabilities to the pipeline:
model performance tracking, feature drift detection, run logging, and
automated health reporting — no external MLOps service required.

### How It Works

```
Phase 4 live_predictions.csv   (or demo data)
        ↓
  monitor.py  →  detect_data_drift (PSI + KS test per feature)
        ↓
  monitor.py  →  monitor_predictions (RMSE, MAE, R², signal accuracy)
        ↓
  monitor.py  →  check_alerts (critical / warning thresholds)
        ↓
  monitor.py  →  log_run  →  outputs/run_log.jsonl
        ↓
  monitor.py  →  generate_report  →  outputs/monitoring_report.json
        ↓
  monitoring_pipeline.py  →  console summary + optional drift chart
```

### Drift Detection

Two complementary statistical tests are run per feature:

| Method | What it measures |
|--------|-----------------|
| **PSI** (Population Stability Index) | Bucket-level distribution shift between baseline and new data |
| **KS test** (Kolmogorov–Smirnov two-sample) | Overall distributional difference |

PSI interpretation:

| PSI range | Status | Action |
|-----------|--------|--------|
| < 0.10 | ✅ OK | No action needed |
| 0.10 – 0.20 | ⚠️ Warning | Monitor closely |
| > 0.20 | 🔴 Alert | Consider retraining |

### Outputs

| File | Description |
|------|-------------|
| `outputs/monitoring_baseline.json` | Per-feature statistics from training set (reference for drift) |
| `outputs/run_log.jsonl` | Append-only log — one JSON record per pipeline run |
| `outputs/monitoring_report.json` | Latest full monitoring report (drift + perf + alerts) |
| `outputs/monitoring_drift.png` | Feature drift PSI bar chart (with `--plot`) |

### Quick Start — Phase 5

```bash
# Demo mode — no CSV or trained model required
python main.py --phase 5 --demo

# With existing Phase 4 predictions
python main.py --phase 5

# With drift chart
python main.py --phase 5 --demo --plot

# Force re-build of the monitoring baseline
python main.py --phase 5 --demo --refresh-baseline

# Run all five phases end-to-end
python main.py --phase all --demo --plot
```

### CLI Options (Phase 5)

| Option | Default | Description |
|--------|---------|-------------|
| `--phase 5` | — | Run Phase 5 MLOps & Monitoring pipeline |
| `--monitoring-predictions-csv PATH` | outputs/live_predictions.csv | Phase 4 predictions to analyse |
| `--monitoring-training-csv PATH` | outputs/results.csv | Training data for baseline creation |
| `--monitoring-baseline-path PATH` | outputs/monitoring_baseline.json | Baseline JSON path |
| `--monitoring-run-log PATH` | outputs/run_log.jsonl | JSONL run log path |
| `--monitoring-report PATH` | outputs/monitoring_report.json | Monitoring report output path |
| `--refresh-baseline` | False | Force re-creation of the baseline |

### Monitoring Report Structure

```json
{
  "report_generated_at": "2026-04-21T21:00:00+00:00",
  "overall_health": "ok",
  "summary": {
    "drift_status": "ok",
    "n_critical_alerts": 0,
    "n_warning_alerts": 0,
    "n_features_drifted": 0
  },
  "drift": { "feature_drift": { … }, "drifted_features": [] },
  "performance": { "rmse": 2.31, "mae": 1.82, "r2": 0.91, … },
  "alerts": [],
  "run_history": [ … ]
}
```

---

## Phase 2 — FinBERT Sentiment Analysis

Phase 2 adds a financial news sentiment layer on top of the Phase 1 foundation.
Sentiment from Reliance-related headlines is a known contributor to option pricing
anomalies (ΔX) — this phase mines and scores that signal so Phase 3's XGBoost model
can use it as a feature.

### How It Works

```
NewsAPI / RSS feeds (ET Markets, Moneycontrol, NDTV Profit)
        ↓
  news_fetcher.py  →  data/news/raw_news.csv
        ↓
   sentiment.py    →  data/news/sentiment_scores.csv
        ↓
sentiment_pipeline.py
        ↓  merges on date
Phase 1 results  (delta_x, greeks, bs_price, …)
        ↓
outputs/results_with_sentiment.csv
        ↓
[Phase 3 XGBoost — features: delta_x + sentiment + greeks + moneyness + …]
```

### Sentiment Score

FinBERT (`ProsusAI/finbert`) outputs three softmax probabilities for each headline:

```
score = P(positive) − P(negative)   ∈ [−1, +1]
```

Articles are aggregated per calendar day:

| Column | Description |
|--------|-------------|
| `daily_sentiment_score` | Mean `P(pos) − P(neg)` across all articles that day |
| `daily_sentiment_label` | Positive / Negative / Neutral |
| `daily_article_count` | Number of Reliance articles that day |
| `daily_pos_mean` | Mean P(positive) |
| `daily_neg_mean` | Mean P(negative) |

### Quick Start — Phase 2

```bash
# Demo (no API key or internet required)
python main.py --phase 2 --demo

# Live RSS feeds (free, no API key)
python main.py --phase 2

# With NewsAPI key (30-day history, higher article count)
NEWSAPI_KEY=your_key python main.py --phase 2

# Run both phases end-to-end
python main.py --phase all --demo --plot
```

### CLI Options (Phase 2)

| Option | Default | Description |
|--------|---------|-------------|
| `--phase 2` | — | Run Phase 2 sentiment pipeline |
| `--phase all` | — | Run Phase 1 then Phase 2 |
| `--newsapi-key KEY` | env `NEWSAPI_KEY` | NewsAPI.org developer key |
| `--lookback-days N` | 90 | Days of news history to fetch |
| `--sentiment-model MODEL` | ProsusAI/finbert | HuggingFace model name |
| `--sentiment-batch-size N` | 16 | FinBERT inference batch size |
| `--sentiment-output-csv PATH` | outputs/results_with_sentiment.csv | Phase 2 output |
| `--no-cache` | False | Disable news + sentiment caching |

### Fallback Chain

If FinBERT is not installed (no `transformers`/`torch`), the module degrades:

1. **FinBERT** (`ProsusAI/finbert`) — primary, financial-domain BERT
2. **TextBlob** — NLTK pattern-based polarity
3. **VADER** — rule-based financial lexicon
4. **Neutral zeros** — if all else fails

---

## Phase 3 — XGBoost ΔX Prediction Model

Phase 3 trains an **XGBoost regression model** to predict option mispricing (ΔX) using the
enriched feature set from Phases 1 & 2 as inputs.

### How It Works

```
outputs/results_with_sentiment.csv   (Phase 2 output)
  — or —
outputs/results.csv                  (Phase 1 fallback)
        ↓
  Feature engineering
    • moneyness   = S / K
    • option_type_enc  (CE→1, PE→0)
    • fill missing sentiment columns → 0
        ↓
  Train / test split  (80 % / 20 %)
        ↓
  XGBoost regression  →  predict ΔX
        ↓
  Evaluate  (RMSE, MAE, R²)
        ↓
  outputs/xgb_model.json         (saved model)
  outputs/predictions.csv        (test-set predictions)
  outputs/xgb_feature_importance.png
  outputs/xgb_actual_vs_predicted.png
  outputs/xgb_residuals.png
```

### Features Used

| Feature | Source |
|---------|--------|
| `strike_price` | Phase 1 |
| `underlying_value` | Phase 1 |
| `time_to_expiry` | Phase 1 |
| `volatility` | Phase 1 |
| `risk_free_rate` | Phase 1 |
| `bs_price` | Phase 1 |
| `delta`, `gamma`, `theta`, `vega`, `rho` | Phase 1 Greeks |
| `moneyness` (S/K) | Engineered |
| `option_type_enc` (CE→1, PE→0) | Engineered |
| `daily_sentiment_score` | Phase 2 |
| `daily_pos_mean`, `daily_neg_mean` | Phase 2 |
| `daily_article_count` | Phase 2 |

### Quick Start — Phase 3

```bash
# Train on Phase 2 enriched data (recommended)
python main.py --phase 3

# Train with charts
python main.py --phase 3 --plot

# Run all three phases end-to-end
python main.py --phase all --demo --plot

# Custom XGBoost hyper-parameters
python main.py --phase 3 \
    --xgb-n-estimators 500 \
    --xgb-max-depth 4 \
    --xgb-learning-rate 0.02
```

### CLI Options (Phase 3)

| Option | Default | Description |
|--------|---------|-------------|
| `--phase 3` | — | Run Phase 3 XGBoost model |
| `--ml-input-csv PATH` | outputs/results_with_sentiment.csv | Input data (Phase 2 output) |
| `--ml-model-path PATH` | outputs/xgb_model.json | Where to save trained model |
| `--ml-predictions-csv PATH` | outputs/predictions.csv | Test-set predictions CSV |
| `--ml-test-size FRACTION` | 0.2 | Fraction held out for testing |
| `--xgb-n-estimators N` | 300 | Boosting rounds |
| `--xgb-max-depth N` | 6 | Max tree depth |
| `--xgb-learning-rate LR` | 0.05 | Learning rate |
| `--xgb-subsample RATIO` | 0.8 | Row sub-sampling |
| `--xgb-colsample-bytree RATIO` | 0.8 | Feature sub-sampling |
| `--xgb-reg-alpha ALPHA` | 0.1 | L1 regularisation |
| `--xgb-reg-lambda LAMBDA` | 1.0 | L2 regularisation |

---

## Phase 4 — Live Inference Pipeline

Phase 4 loads the trained Phase 3 XGBoost model and runs it on **new, unseen
option chain data** — no retraining required.  For each option it produces a
**predicted ΔX** and an actionable **trading signal**.

### How It Works

```
New option chain CSV  (or demo data)
        ↓
  Phase 1 pipeline  →  BS price, Greeks, ΔX
        ↓  (optional)
  Phase 2 sentiment  →  merge daily_sentiment_score
        ↓
  Feature engineering (moneyness, option_type_enc, …)
        ↓
  Load outputs/xgb_model.json  (Phase 3 trained model)
        ↓
  XGBoost inference  →  predicted_delta_x
        ↓
  Trading signals:
    BUY  (long)  — predicted ΔX < −₹2  (option underpriced)
    SELL (short) — predicted ΔX >  ₹2  (option overpriced)
    HOLD         — |predicted ΔX| ≤ ₹2 (fairly priced)
        ↓
  Rank by signal priority + |ΔX|
        ↓
  outputs/live_predictions.csv
  outputs/live_inference_signals.png   (with --plot)
```

### Output Columns

| Column | Description |
|--------|-------------|
| `predicted_delta_x` | XGBoost-predicted ΔX (₹) |
| `signal` | `BUY`, `SELL`, or `HOLD` |
| `confidence` | `\|predicted_delta_x\|` normalised to [0, 1] within batch |

### Quick Start — Phase 4

```bash
# Demo mode — no CSV or trained model required*
python main.py --phase 4 --demo

# With your own NSE option chain CSV (Phase 3 model must exist)
python main.py --phase 4 --input data/raw/nse_option_chain.csv

# With charts
python main.py --phase 4 --demo --plot

# Run all four phases end-to-end
python main.py --phase all --demo --plot

# Custom signal thresholds
python main.py --phase 4 --demo --buy-threshold 5.0 --sell-threshold 5.0
```

> \* Demo mode auto-trains a temporary model if `outputs/xgb_model.json` is
> absent, so no prior Phase 3 run is needed.

### CLI Options (Phase 4)

| Option | Default | Description |
|--------|---------|-------------|
| `--phase 4` | — | Run Phase 4 live inference pipeline |
| `--inference-output-csv PATH` | outputs/live_predictions.csv | Predictions output CSV |
| `--buy-threshold ₹` | 2.0 | Predicted ΔX below −threshold → BUY |
| `--sell-threshold ₹` | 2.0 | Predicted ΔX above +threshold → SELL |
| `--ml-model-path PATH` | outputs/xgb_model.json | Trained model to load |

---

## Project Structure (Phase 1 + 2)

### Black-Scholes Inputs

| Parameter | Symbol | Value / Source |
|-----------|--------|----------------|
| Spot Price | S | ₹1,390.20 (UNDERLYING VALUE column) |
| Strike Price | K | Multiple strikes from option chain |
| Time to Expiry | T | (expiry − trade date) / 365 |
| Risk-Free Rate | r | ~7% (Indian 91-day T-Bill rate) |
| Volatility | σ | Calculated from 1-year daily closing prices |
| Dividend Yield | q | 0 (no ex-div date before April 2026 expiry) |

### Formulae

**d₁ and d₂:**

```
d₁ = [ln(S/K) + (r + σ²/2) · T] / (σ · √T)
d₂ = d₁ − σ · √T
```

**European Call (CE):**

```
C = S · N(d₁) − K · e^(−rT) · N(d₂)
```

**European Put (PE):**

```
P = K · e^(−rT) · N(−d₂) − S · N(−d₁)
```

**Pricing Error (ΔX) — target for Phase 3 XGBoost:**

```
ΔX = Actual Market Premium (LTP) − BS Theoretical Price
```

### Option Greeks

| Greek | Symbol | Formula (Call) | Formula (Put) |
|-------|--------|----------------|---------------|
| Delta | Δ | N(d₁) | N(d₁) − 1 |
| Gamma | Γ | n(d₁) / (S·σ·√T) | Same as call |
| Theta | Θ | −[S·n(d₁)·σ/(2√T)] − r·K·e^(−rT)·N(d₂) | −[S·n(d₁)·σ/(2√T)] + r·K·e^(−rT)·N(−d₂) |
| Vega | ν | S·n(d₁)·√T / 100 | Same as call |
| Rho | ρ | K·T·e^(−rT)·N(d₂) / 100 | −K·T·e^(−rT)·N(−d₂) / 100 |

---

## Phase 1 — Black-Scholes Foundation

## Project Structure (Phase 1 + 2 + 3 + 4 + 5)

```
reliance-option-pricing/
│
├── data/
│   ├── sample/
│   │   └── sample_option_chain.csv    # Sample NSE option chain (11-Mar-2026)
│   └── news/                          # Phase 2: cached news & sentiment CSVs
│       ├── raw_news.csv               # Raw article cache
│       └── sentiment_scores.csv       # Daily sentiment score cache
│
├── src/
│   ├── __init__.py
│   ├── black_scholes.py               # BS pricing engine — Call & Put
│   ├── volatility.py                  # Historical volatility (full + rolling)
│   ├── greeks.py                      # Delta, Gamma, Theta, Vega, Rho
│   ├── data_loader.py                 # Load NSE CSV + yfinance downloader
│   ├── pipeline.py                    # Phase 1 end-to-end orchestrator
│   ├── news_fetcher.py                # Phase 2: RSS / NewsAPI downloader
│   ├── sentiment.py                   # Phase 2: FinBERT scoring + aggregation
│   ├── sentiment_pipeline.py          # Phase 2 end-to-end orchestrator
│   ├── model.py                       # Phase 3: XGBoost DeltaXModel wrapper
│   ├── ml_pipeline.py                 # Phase 3 end-to-end orchestrator
│   ├── inference.py                   # Phase 4: inference engine + signal generator
│   ├── live_pipeline.py               # Phase 4 end-to-end orchestrator
│   ├── monitor.py                     # Phase 5: drift detection, run logging, reports
│   └── monitoring_pipeline.py         # Phase 5 end-to-end orchestrator
│
├── tests/
│   ├── __init__.py
│   ├── test_black_scholes.py          # BS engine unit tests
│   ├── test_volatility.py             # Volatility unit tests
│   ├── test_greeks.py                 # Greeks unit tests
│   ├── test_sentiment.py              # Phase 2 sentiment unit tests
│   ├── test_model.py                  # Phase 3 XGBoost unit tests (40 tests)
│   ├── test_inference.py              # Phase 4 inference unit tests (38 tests)
│   └── test_monitor.py                # Phase 5 monitoring unit tests (68 tests)
│
├── outputs/                           # Generated results (git-ignored)
│   └── .gitkeep
│
├── main.py                            # CLI entry point (Phases 1–5)
├── config.py                          # Configuration constants
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/laveeshchauhan/reliance-option-pricing.git
cd reliance-option-pricing

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Quick Demo (no data needed)

```bash
python main.py --demo
```

### With your NSE option chain CSV

```bash
python main.py --input data/sample/sample_option_chain.csv
```

### With multiple CSV files (glob)

```bash
python main.py --input "data/raw/*.csv"
```

### Custom parameters + charts

```bash
python main.py --demo \
    --risk-free-rate 0.072 \
    --volatility-window 60 \
    --output-csv outputs/results_mar2026.csv \
    --plot
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input PATH` | — | Path to NSE CSV file(s) or glob pattern |
| `--demo` | False | Run with synthetic sample data |
| `--ticker TICKER` | RELIANCE.NS | Yahoo Finance ticker |
| `--risk-free-rate RATE` | 0.07 | Annualised risk-free rate (decimal) |
| `--volatility-window DAYS` | 30 | Rolling volatility window (trading days) |
| `--dividend-yield YIELD` | 0.0 | Continuous dividend yield |
| `--output-csv PATH` | outputs/results.csv | Results output path |
| `--plot` | False | Generate and save charts |
| `--quiet` | False | Suppress terminal output |

---

## NSE Data Format

The expected CSV format matches the NSE F&O option chain download:

| Column | Description |
|--------|-------------|
| SYMBOL | RELIANCE |
| DATE | 11-Mar-2026 |
| EXPIRY | 28-Apr-2026 |
| OPTION TYPE | CE / PE |
| STRIKE PRICE | e.g. 1400 |
| OPEN / HIGH / LOW / CLOSE / LTP | Option price data (₹) |
| SETTLE PRICE | Settlement price (₹) |
| NO. OF CONTRACTS | Volume |
| TURNOVER IN ₹ LAKHS | Turnover |
| PREMIUM TURNOVER IN ₹ LAKHS | Premium turnover |
| OPEN INT | Open interest |
| CHANGE IN OI | Change in OI |
| UNDERLYING VALUE | Spot price (₹) |

### How to Download from NSE

1. Visit [nseindia.com](https://www.nseindia.com/report-detail/fo_eq_security)
2. Select **Symbol → RELIANCE**
3. Select your date range (e.g. 01-Dec-2025 to 11-Mar-2026)
4. Download CSV

---

## Output

The pipeline produces:

### `outputs/results.csv`

Each row contains all inputs plus:

| New Column | Description |
|------------|-------------|
| `time_to_expiry` | T = (expiry − date) / 365 |
| `volatility` | Rolling historical σ |
| `bs_price` | Black-Scholes theoretical price (₹) |
| `actual_premium` | Market LTP (₹) |
| `delta_x` | ΔX = actual − BS (₹) |
| `classification` | Overpriced / Underpriced / Fair |
| `delta` | Option delta Δ |
| `gamma` | Option gamma Γ |
| `theta` | Daily theta Θ (₹/day) |
| `vega` | Vega per 1% σ change |
| `rho` | Rho per 1% rate change |

### Terminal Summary

```
═══════════════════════════════════════════════════════════════════
  PHASE 1 — BLACK-SCHOLES PRICING RESULTS SUMMARY
  All prices in ₹ (INR)  |  RELIANCE  |  NSE India
═══════════════════════════════════════════════════════════════════
  Total option rows processed  :       30
  Call options (CE)            :       16
  Put options  (PE)            :       14
═══════════════════════════════════════════════════════════════════
  Mean ΔX (Actual − BS)        : ₹    -2.3451
  Std  ΔX                      : ₹    15.2310
  RMSE                         : ₹    15.4100
═══════════════════════════════════════════════════════════════════
  Overpriced  (ΔX >  ₹1)      :       12
  Underpriced (ΔX < −₹1)      :       14
  Fair        (|ΔX| ≤ ₹1)     :        4
═══════════════════════════════════════════════════════════════════
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

Edit `config.py` to change defaults:

```python
DEFAULT_RISK_FREE_RATE = 0.07      # 7% — Indian 91-day T-Bill
TRADING_DAYS_PER_YEAR = 252        # For volatility annualisation
CALENDAR_DAYS_PER_YEAR = 365       # For time-to-expiry
DEFAULT_VOLATILITY_WINDOW = 30     # Rolling window (trading days)
DEFAULT_DIVIDEND_YIELD = 0.0       # q = 0 for short-term RELIANCE options
```

---

## Key Design Decisions

1. **q = 0**: Reliance's last ex-dividend was 03-Mar-2026 (already passed). The next ex-dividend (~May 2026) falls **after** the April 2026 expiry, so dividend yield is irrelevant for this analysis.

2. **T = calendar days / 365**: NSE uses calendar days for time-to-expiry (not trading days).

3. **σ from rolling window**: Uses a configurable rolling window (default 30 days) so each row gets the volatility as it would have been known on that trade date.

4. **ΔX is the ML target**: The pricing error column is critical — it becomes the target variable for Phase 3's XGBoost model to predict option mispricing.

---

## License

MIT License
