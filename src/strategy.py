"""
src/strategy.py
---------------
Combine regime labels + ML model signals into final position sizing.

Pipeline position: Step 5 — runs after model.py
Input:  DataFrame with 'regime', 'regime_label', and 'signal' columns
Output: Same DataFrame with 'position' column (fraction of capital to deploy)
"""

import numpy as np
import pandas as pd


# ── Strategy configuration ─────────────────────────────────────────────────────

# How much capital to deploy per regime (0.0 – 1.0)
# Rationale: be more aggressive in bull, cautious in sideways, exit in bear.
REGIME_ALLOCATION = {
    "Bull":     1.0,
    "Sideways": 0.5,
    "Bear":     0.0,    # flat in bear regime — no short selling
}

# Allow shorting (position = -1) when signal == -1 in certain regimes
ALLOW_SHORT = False

# Minimum consecutive days a signal must persist before acting (noise filter)
SIGNAL_CONFIRMATION_BARS = 2


# ── Position generation ────────────────────────────────────────────────────────

def generate_positions(df: pd.DataFrame,
                       regime_allocation: dict = None,
                       allow_short: bool = ALLOW_SHORT,
                       confirmation_bars: int = SIGNAL_CONFIRMATION_BARS) -> pd.DataFrame:
    """
    Translate ML signals + regime labels into scalar positions.

    Logic
    -----
    1. Start from the ML signal: +1, 0, or -1.
    2. Apply a persistence filter: only act if the signal has held for
       `confirmation_bars` consecutive periods (reduces whipsaws).
    3. Scale by the regime allocation fraction:
         position = confirmed_signal × regime_factor
    4. If allow_short=False, clip position to [0, 1].

    Parameters
    ----------
    df                 : DataFrame with 'signal' and 'regime_label' columns.
    regime_allocation  : Dict mapping regime label → capital fraction.
                         Defaults to REGIME_ALLOCATION.
    allow_short        : Whether negative positions are permitted.
    confirmation_bars  : How many consecutive matching signals before entry.

    Returns
    -------
    df with new columns:
      'confirmed_signal' — signal after persistence filter
      'regime_factor'    — allocation multiplier for the current regime
      'position'         — final position size [-1, 1]
    """
    if regime_allocation is None:
        regime_allocation = REGIME_ALLOCATION

    df = df.copy()

    # ── 1. Persistence filter ──────────────────────────────────────────────────
    # A signal is "confirmed" only when it equals its value for the past N bars.
    signal = df["signal"].ffill()   # forward-fill NaNs from warm-up period

    confirmed = signal.copy()
    for i in range(1, confirmation_bars):
        confirmed = confirmed.where(confirmed == signal.shift(i), 0)

    df["confirmed_signal"] = confirmed

    # ── 2. Regime factor ──────────────────────────────────────────────────────
    df["regime_factor"] = df["regime_label"].map(regime_allocation).fillna(0.0)

    # ── 3. Combine ────────────────────────────────────────────────────────────
    df["position"] = df["confirmed_signal"] * df["regime_factor"]

    # ── 4. Short clipping ─────────────────────────────────────────────────────
    if not allow_short:
        df["position"] = df["position"].clip(lower=0)

    df["position"] = df["position"].fillna(0)

    # Diagnostics
    pos_counts = df["position"].value_counts().sort_index()
    print("[strategy] Position distribution:")
    print(pos_counts.to_string(), "\n")

    return df


# ── Regime-specific override ───────────────────────────────────────────────────

def apply_regime_override(df: pd.DataFrame, bear_exit: bool = True) -> pd.DataFrame:
    """
    Hard override: force position = 0 whenever regime == 'Bear'.

    This is a safety valve independent of the ML signal — useful when the model
    is uncertain but the macro regime screams risk-off.

    Parameters
    ----------
    df        : DataFrame with 'position' and 'regime_label' columns.
    bear_exit : If True, zero out all positions in Bear regime.

    Returns
    -------
    df with updated 'position'.
    """
    df = df.copy()
    if bear_exit:
        bear_mask = df["regime_label"] == "Bear"
        df.loc[bear_mask, "position"] = 0
        n_overridden = bear_mask.sum()
        print(f"[strategy] Bear-regime override applied to {n_overridden} bars.")
    return df


# ── Transaction cost simulation ────────────────────────────────────────────────

def add_transaction_costs(df: pd.DataFrame,
                          cost_per_trade: float = 0.001) -> pd.DataFrame:
    """
    Deduct a round-trip cost whenever the position changes.

    Parameters
    ----------
    df             : DataFrame with 'position' column.
    cost_per_trade : Fractional cost per position change (default 0.1 %).

    Returns
    -------
    df with new 'trade_cost' column (cumulative cost is handled in backtest.py).
    """
    df = df.copy()
    position_change  = df["position"].diff().abs()
    df["trade_cost"] = position_change * cost_per_trade
    df["trade_cost"] = df["trade_cost"].fillna(0)
    total_cost = df["trade_cost"].sum()
    print(f"[strategy] Total estimated transaction costs: {total_cost:.4f} ({total_cost * 100:.2f}%)")
    return df