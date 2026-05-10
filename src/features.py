"""
features.py
-----------
Pipeline Step 2: Feature Engineering

Transforms raw OHLCV data into ML-ready features.
Called after data_loader.py, feeds into regime.py and model.py.

Feature Groups:
    - Returns      : log returns over multiple horizons
    - Volatility   : rolling standard deviation of returns
    - Momentum     : RSI, rate of change
    - Trend        : moving average ratios, MACD
    - Volume       : volume z-score
    - Target       : forward return label for supervised learning
"""

import pandas as pd
import numpy as np


# ── Configuration ──────────────────────────────────────────────────────────────

# Forward return horizon for the prediction target (in trading days)
TARGET_HORIZON = 5   # predict 5-day ahead return


# ── Main Entry Point ───────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, target_horizon: int = TARGET_HORIZON) -> pd.DataFrame:
    """
    Master function — runs all feature engineering steps in order.

    Parameters
    ----------
    df             : raw OHLCV DataFrame from data_loader.get_data()
    target_horizon : how many days ahead to predict (default: 5)

    Returns
    -------
    pd.DataFrame with all features + target column, NaNs dropped.

    Usage
    -----
    from src.data_loader import get_data
    from src.features import build_features

    df_raw = get_data()
    df_features = build_features(df_raw)
    """
    feat = df.copy()

    feat = _add_returns(feat)
    feat = _add_volatility(feat)
    feat = _add_momentum(feat)
    feat = _add_trend(feat)
    feat = _add_volume_features(feat)
    feat = _add_target(feat, target_horizon)

    # Drop rows with NaN — caused by rolling windows at the start
    n_before = len(feat)
    feat = feat.dropna()
    n_dropped = n_before - len(feat)

    print(f"[features] ✓ Built {feat.shape[1]} features over {len(feat)} rows "
          f"(dropped {n_dropped} warm-up rows)")

    return feat


# ── Feature Groups ─────────────────────────────────────────────────────────────

def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns over multiple horizons.

    Why log returns?
    - Additive across time (arithmetic returns are not)
    - More normally distributed — better for ML models
    - Standard in quantitative finance
    
    Formula: log(P_t / P_{t-n})
    """
    close = df["Close"]

    df["return_1d"]  = np.log(close / close.shift(1))   # daily return
    df["return_5d"]  = np.log(close / close.shift(5))   # weekly
    df["return_21d"] = np.log(close / close.shift(21))  # monthly

    return df


def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling volatility — key input for regime detection.

    High vol  → likely bear or crisis regime
    Low vol   → likely bull or sideways regime

    Annualised by multiplying by sqrt(252) — the number of trading days/year.
    This makes the number interpretable (e.g. 0.15 = 15% annualised vol).
    """
    daily_ret = df["return_1d"]

    df["vol_21d"]  = daily_ret.rolling(21).std()  * np.sqrt(252)
    df["vol_63d"]  = daily_ret.rolling(63).std()  * np.sqrt(252)

    # Vol ratio: short-term vs long-term vol — spikes signal regime shifts
    df["vol_ratio"] = df["vol_21d"] / df["vol_63d"]

    return df


def _add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum indicators — capture trend strength and speed.
    """
    # ── RSI (Relative Strength Index) ─────────────────────────────────────────
    # Measures whether an asset is overbought (>70) or oversold (<30)
    # Range: 0 to 100
    df["rsi_14"] = _compute_rsi(df["Close"], period=14)

    # ── Rate of Change ────────────────────────────────────────────────────────
    # Percentage change over N days — pure momentum signal
    # ROC > 0 means price is higher than N days ago
    df["roc_10"] = df["Close"].pct_change(10) * 100  # 10-day ROC

    return df


def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend indicators — where is price relative to its moving averages?

    SMA ratio > 1  → price is above its average  (uptrend)
    SMA ratio < 1  → price is below its average  (downtrend)
    """
    close = df["Close"]

    sma_50  = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()

    # Price relative to moving averages
    df["sma_ratio_50_200"] = sma_50 / sma_200  # "Golden/Death Cross" signal
    df["price_to_sma50"]   = close / sma_50    # How far above/below 50-day MA

    # ── MACD ──────────────────────────────────────────────────────────────────
    # Moving Average Convergence Divergence
    # MACD line = 12-day EMA minus 26-day EMA
    # Signal    = 9-day EMA of MACD line
    # Histogram = MACD - Signal  (what we use as a feature)
    ema_12   = close.ewm(span=12, adjust=False).mean()
    ema_26   = close.ewm(span=26, adjust=False).mean()
    macd     = ema_12 - ema_26
    signal   = macd.ewm(span=9, adjust=False).mean()

    df["macd_hist"] = macd - signal

    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume z-score — is today's volume unusually high or low?

    A large volume spike often precedes or confirms a regime shift.
    We use z-score to normalise across time (volume levels drift over years).

    Z-score = (today's volume - 21d mean) / 21d std
    """
    vol_mean = df["Volume"].rolling(21).mean()
    vol_std  = df["Volume"].rolling(21).std()

    df["volume_zscore"] = (df["Volume"] - vol_mean) / vol_std

    return df


def _add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Prediction target for supervised learning.

    target_return  : continuous forward log return (for regression)
    target_label   : binary label — 1 if return > 0 else 0 (for classification)

    We use classification (target_label) as our primary target because:
    - Predicting direction is more actionable than predicting exact magnitude
    - Classification models are easier to evaluate (accuracy, precision/recall)
    - Less sensitive to outliers than regression

    IMPORTANT: We shift by -horizon so each row's target is the
    *future* return, not the current one. This is the correct setup
    to avoid look-ahead bias.
    """
    close = df["Close"]
    forward_return = np.log(close.shift(-horizon) / close)

    df["target_return"] = forward_return
    df["target_label"]  = (forward_return > 0).astype(int)  # 1 = up, 0 = down

    return df


# ── Technical Indicator Helpers ────────────────────────────────────────────────

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Standard Wilder RSI implementation.

    Steps:
    1. Compute daily change
    2. Separate gains (positive changes) and losses (negative changes)
    3. Smooth both with an exponential moving average
    4. RSI = 100 - (100 / (1 + avg_gain / avg_loss))
    """
    delta = series.diff()

    gain = delta.clip(lower=0)          # keep only positive changes
    loss = -delta.clip(upper=0)         # keep only negative changes (make positive)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss           # relative strength ratio
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ── Utility ────────────────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Returns only the feature column names (excludes raw OHLCV and targets).
    Use this when passing features into a model to avoid accidentally
    including the target or raw price columns.

    Usage
    -----
    feature_cols = get_feature_columns(df_features)
    X = df_features[feature_cols]
    y = df_features["target_label"]
    """
    exclude = {"Open", "High", "Low", "Close", "Volume",
               "target_return", "target_label"}
    return [col for col in df.columns if col not in exclude]