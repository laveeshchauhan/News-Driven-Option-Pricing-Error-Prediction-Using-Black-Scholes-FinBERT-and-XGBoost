# Reliance Option Pricing

**Black-Scholes Option Pricing & Sentiment-Driven ML for RELIANCE Industries (NSE India)**

> All prices are in **₹ (Indian Rupees / INR)** — this is NSE India market data, NOT USD.

---

## Project Overview

This project is a **5-phase quantitative finance system** for pricing **Reliance Industries (RELIANCE.NS)** options listed on NSE India:

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Black-Scholes Pricing Engine + Greeks | ✅ Complete |
| Phase 2 | FinBERT NLP Sentiment Analysis | 🔜 Planned |
| Phase 3 | XGBoost ML Model (ΔX prediction) | 🔜 Planned |
| Phase 4 | Live Inference Pipeline | 🔜 Planned |
| Phase 5 | MLOps & Monitoring | 🔜 Planned |

---

## Phase 1 — Black-Scholes Foundation

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

## Project Structure

```
reliance-option-pricing/
│
├── data/
│   └── sample/
│       └── sample_option_chain.csv    # Sample NSE option chain (11-Mar-2026)
│
├── src/
│   ├── __init__.py
│   ├── black_scholes.py               # BS pricing engine — Call & Put
│   ├── volatility.py                  # Historical volatility (full + rolling)
│   ├── greeks.py                      # Delta, Gamma, Theta, Vega, Rho
│   ├── data_loader.py                 # Load NSE CSV + yfinance downloader
│   └── pipeline.py                    # Phase 1 end-to-end orchestrator
│
├── tests/
│   ├── __init__.py
│   ├── test_black_scholes.py          # BS engine unit tests
│   ├── test_volatility.py             # Volatility unit tests
│   └── test_greeks.py                 # Greeks unit tests
│
├── outputs/                           # Generated results (git-ignored)
│   └── .gitkeep
│
├── main.py                            # CLI entry point
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
