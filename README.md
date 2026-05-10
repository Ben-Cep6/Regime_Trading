# Regime-Aware Machine Learning Trading System

A quantitative trading system that combines **unsupervised** and **supervised** machine learning to make risk-aware trading decisions on equity markets. Built as a final project for Introduction to AI by Ben Concepcion.

---

## Overview

Traditional buy-and-hold strategies expose investors to full market downturns. This system addresses that by first detecting the current market environment (regime), then using a predictive ML model to decide when to enter and exit positions — optimising for **risk-adjusted returns** rather than raw returns.

**Asset:** SPY (S&P 500 ETF)  
**Period:** January 2010 – December 2023 (14 years, 3,522 trading days)  
**Approach:** KMeans regime detection + Random Forest signal prediction

---

## Project Structure

```
project_root/
├── data/                   # Raw and processed data (auto-generated)
│   ├── SPY_raw.csv         # Cached raw OHLCV download
│   ├── raw.csv             # Pipeline copy of raw data
│   ├── features.csv        # Engineered features
│   └── regimes.csv         # Features + regime labels
├── src/
│   ├── data_loader.py      # Step 1: Data collection from Yahoo Finance
│   ├── features.py         # Step 2: Feature engineering (19 indicators)
│   ├── regime.py           # Step 3: KMeans regime detection
│   ├── model.py            # Step 4: Random Forest prediction model
│   ├── strategy.py         # Step 5: Position sizing logic
│   └── backtest.py         # Step 6/7: Backtesting engine + metrics
├── results/                # Output charts and metrics (auto-generated)
│   ├── regimes.png         # Price chart with regime shading
│   ├── feature_importance.png
│   └── backtest.png        # 4-panel results dashboard
├── main.py                 # Pipeline runner — run this
└── README.md
```

---

## Setup and Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Regime_Trading

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install pandas numpy scikit-learn matplotlib yfinance xgboost
```

---

## Running the Pipeline

```bash
python main.py
```

The pipeline runs all 7 steps automatically and saves results to `data/` and `results/`. Configuration (ticker, date range, model type) is controlled via the `CONFIG` dict at the top of `main.py`.

```python
CONFIG = {
    "ticker":         "SPY",
    "start":          "2010-01-01",
    "end":            "2024-01-01",
    "model_type":     "rf",        # "rf" for Random Forest, "xgb" for XGBoost
    "allow_short":    False,
    "bear_exit":      True,
    "cost_per_trade": 0.001,
}
```

---

## Pipeline

### Step 1 — Data Collection (`data_loader.py`)
Downloads daily OHLCV data from Yahoo Finance via `yfinance`. Implements a local CSV cache so repeated runs load instantly. Uses `auto_adjust=True` to correct all price columns for dividends and splits.

### Step 2 — Feature Engineering (`features.py`)
Transforms raw prices into 19 technical features across four categories:

| Category | Features |
|---|---|
| Returns | 1-day, 5-day, 21-day cumulative returns |
| Volatility | 21-day, 63-day rolling std, volatility ratio |
| Momentum | RSI-14, MACD histogram, Rate of Change (10-day) |
| Trend | SMA 50/200 ratio, price vs SMA50, volume z-score |

204 warm-up rows are dropped to eliminate NaNs from long rolling windows.

### Step 3 — Regime Detection (`regime.py`)
Applies **KMeans clustering (k=3)** to partition the market into three regimes. Features are standardised before clustering. Regimes are re-labelled by mean return rank so that Bear=0, Sideways=1, Bull=2 regardless of KMeans assignment.

| Regime | Days | Mean Return/Day | Approx Sharpe |
|---|---|---|---|
| Bear | 898 (27%) | -0.68% | -0.50 |
| Sideways | 1,979 (60%) | +0.22% | +0.39 |
| Bull | 441 (13%) | +0.78% | +0.54 |

**Silhouette score: 0.354** — acceptable separation for financial time series data.

### Step 4 — ML Prediction (`model.py`)
Trains a **Random Forest classifier** (200 trees) to predict the next 5-day return direction. Target variable is ternary: +1 (Buy), 0 (Hold), -1 (Sell), with a ±0.2% dead-band.

Uses **TimeSeriesSplit** (walk-forward) cross-validation to prevent look-ahead bias — always trains on the past and tests on the future.

| Fold | CV Accuracy |
|---|---|
| Fold 1 | 41.67% |
| Fold 2 | 43.84% |
| Fold 3 | 36.41% |
| Fold 4 | 35.69% |
| Fold 5 | 47.46% |
| **Mean** | **41.01% ± 4.46%** |

Random guessing baseline with 3 classes = 33.3%. The model achieves a real but modest predictive edge.

**Top features by importance:** RSI-14 (12.5%), price_to_sma50 (9.7%), vol_63d (9.3%), roc_10 (8.8%), return_21d (8.8%).

### Step 5 — Strategy Logic (`strategy.py`)
Combines the regime label and ML signal into a final position size. The regime sets the allocation ceiling; the ML signal sets the direction.

| Regime | ML Signal | Position |
|---|---|---|
| Bull | Buy (+1) | 1.0 (100% long) |
| Sideways | Buy (+1) | 0.5 (50% long) |
| Bear | Any | 0.0 (flat — hard override) |

A 2-bar signal persistence filter reduces whipsawing. Transaction cost of 0.1% applied on every position change.

### Step 6 & 7 — Backtesting and Evaluation (`backtest.py`)
Vectorised backtest engine. Positions are shifted by 1 bar before multiplying by returns to ensure no look-ahead bias. Computes full performance metrics and generates the 4-panel results dashboard.

---

## Results

| Metric | Strategy | Buy & Hold |
|---|---|---|
| Total Return | 171.1% | 320.3% |
| Annualised Return | 7.88% | 11.54% |
| **Sharpe Ratio** | **0.706** | **0.487** |
| **Max Drawdown** | **-5.63%** | **-35.75%** |
| Calmar Ratio | 1.401 | ~0.32 |
| Annualised Volatility | 5.27% | ~18% |

The strategy underperforms buy-and-hold in raw returns but significantly outperforms on every risk-adjusted metric. The Sharpe ratio improved by **45%** and maximum drawdown was reduced by **84%** — the strategy was largely flat during the COVID crash (March 2020) and the 2022 bear market while passive investors suffered major drawdowns.

---

## Key Findings

- **Risk-adjusted outperformance is real.** A Sharpe of 0.706 vs 0.487 over 14 years is a meaningful, consistent edge.
- **Regime detection works at the strategy layer.** The regime feature scored near-zero importance in the Random Forest — its contribution came entirely from the allocation rules, not the model's predictions.
- **RSI is the strongest predictor.** Momentum-based overbought/oversold signals dominated feature importance, consistent with academic literature.
- **Transaction costs are the biggest drag.** 25.3% total cost over the backtest period due to frequent regime switching — the primary limitation.
- **Some overfitting is present.** The gap between 60% in-sample and 41% CV accuracy is expected for tree-based models on financial data. The out-of-sample edge is real.

---

## Limitations

- Regime switching is too frequent (near-daily in some periods), inflating transaction costs
- The strategy missed the 2010–2019 bull run by over-classifying as Sideways/Bear
- Tested on a single asset (SPY) — may not generalise to individual stocks
- Fixed hyperparameters (k=3, 5-day horizon, 0.2% threshold) not formally optimised

---

## Future Improvements

- Add a 5-day regime persistence filter to prevent daily switching
- Train separate models per regime (Bull model, Bear model, Sideways model)
- Add macroeconomic features: VIX, yield curve slope, credit spreads
- Extend to multiple assets and asset classes
- Formal hyperparameter optimisation via Bayesian search

---

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
yfinance
xgboost (optional)
```

---

## License

MIT License — free to use and modify for educational purposes.