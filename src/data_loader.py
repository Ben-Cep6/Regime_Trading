"""
data_loader.py
--------------
Pipeline Step 1: Data Collection

Fetches daily OHLCV data from Yahoo Finance via yfinance.
Caches results locally so subsequent runs are instant.

Usage (from other modules):
    from src.data_loader import get_data
    df = get_data()  # returns cleaned DataFrame
"""

import os
import pandas as pd
import yfinance as yf



# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_TICKER = "SPY"
DEFAULT_START  = "2010-01-01"
DEFAULT_END    = "2024-01-01"

# One level up from /src → project root, then /data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Main Entry Point ───────────────────────────────────────────────────────────

def get_data(
    ticker: str          = DEFAULT_TICKER,
    start: str           = DEFAULT_START,
    end: str             = DEFAULT_END,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Smart loader — returns cached CSV if available, else downloads from Yahoo.

    Parameters
    ----------
    ticker         : Stock/ETF symbol (default: "SPY")
    start          : Start date string "YYYY-MM-DD"
    end            : End date string   "YYYY-MM-DD"
    force_download : If True, ignore cache and re-download

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
                 indexed by Date (DatetimeIndex, ascending)
    """
    cache_path = _cache_path(ticker)

    if not force_download and os.path.exists(cache_path):
        print(f"[data_loader] Cache hit — loading {ticker} from disk.")
        return _load_csv(cache_path, ticker)

    return _download(ticker, start, end, save=True)


# ── Internal Functions ─────────────────────────────────────────────────────────

def _download(ticker: str, start: str, end: str, save: bool) -> pd.DataFrame:
    """Download from Yahoo Finance via yfinance."""
    print(f"[data_loader] Downloading {ticker} ({start} → {end}) ...")

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts for splits/dividends automatically
        progress=False,     # suppress yfinance progress bar
    )

    if raw.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. "
            f"Check the ticker symbol and date range."
        )

    # yfinance may return MultiIndex columns — flatten them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only standard OHLCV columns
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Ensure ascending date order
    df = df.sort_index(ascending=True)
    df.index.name = "Date"

    # Drop any rows with NaN
    n_before  = len(df)
    df        = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[data_loader] Dropped {n_dropped} row(s) with NaN values.")

    df["Close"] = df["Close"].clip(lower=df["Low"], upper=df["High"])
    print(
        f"[data_loader] ✓ {len(df)} trading days loaded  "
        f"({df.index[0].date()} → {df.index[-1].date()})"
    )

    if save:
        _save_csv(df, ticker)

    return df


def _load_csv(path: str, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    print(
        f"[data_loader] ✓ {len(df)} rows loaded from cache  "
        f"({df.index[0].date()} → {df.index[-1].date()})"
    )
    return df


def _save_csv(df: pd.DataFrame, ticker: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _cache_path(ticker)
    df.to_csv(path)
    print(f"[data_loader] ✓ Saved to {path}")


def _cache_path(ticker: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker}_raw.csv")