"""
src/backtest.py
---------------
Event-driven vectorised backtest engine.

Pipeline position: Step 6 — runs after strategy.py
Input:  DataFrame with 'position', 'returns', and optionally 'trade_cost'
Output: Equity curve, performance metrics dict, charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Constants ──────────────────────────────────────────────────────────────────

TRADING_DAYS  = 252          # annualisation factor
RISK_FREE_RATE = 0.04        # annual risk-free rate (update to current T-bill)
INITIAL_CAPITAL = 10_000     # starting portfolio value (cosmetic — ratios are unaffected)


# ── Core engine ────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame,
                 return_col: str   = "return_1d",
                 position_col: str = "position",
                 cost_col: str     = "trade_cost") -> pd.DataFrame:
    """
    Vectorised backtest: multiply position × next-bar return.

    Parameters
    ----------
    df           : DataFrame with position and return columns.
    return_col   : Column of per-bar asset returns (decimal).
    position_col : Column of positions (scalar, typically 0 or 1).
    cost_col     : Column of per-bar transaction costs (optional).

    Returns
    -------
    df with new columns:
      'strategy_returns' — per-bar P&L of the strategy
      'equity_curve'     — cumulative portfolio value (starts at INITIAL_CAPITAL)
      'buy_hold_curve'   — passive benchmark equity curve
    """
    df = df.copy()

    # Strategy return = position decided on bar t × return realised on bar t+1
    # We shift position by 1 to avoid look-ahead bias.
    df["strategy_returns"] = df[position_col].shift(1) * df[return_col]

    # Subtract transaction costs if available
    if cost_col in df.columns:
        df["strategy_returns"] -= df[cost_col].shift(1).fillna(0)

    df["strategy_returns"] = df["strategy_returns"].fillna(0)

    # Equity curves
    df["equity_curve"]   = INITIAL_CAPITAL * (1 + df["strategy_returns"]).cumprod()
    df["buy_hold_curve"] = INITIAL_CAPITAL * (1 + df[return_col].fillna(0)).cumprod()

    print(f"[backtest] Backtest complete. {len(df)} bars processed.\n")
    return df


# ── Performance metrics ────────────────────────────────────────────────────────

def _annualised_return(returns: pd.Series) -> float:
    total = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS
    return total ** (1 / n_years) - 1 if n_years > 0 else 0.0


def _sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    daily_rf = rf / TRADING_DAYS
    excess   = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return np.sqrt(TRADING_DAYS) * excess.mean() / excess.std()


def _max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    return drawdown.min()


def _calmar_ratio(returns: pd.Series, equity: pd.Series) -> float:
    ann_ret = _annualised_return(returns)
    mdd     = abs(_max_drawdown(equity))
    return ann_ret / mdd if mdd != 0 else np.nan


def _win_rate(returns: pd.Series) -> float:
    active = returns[returns != 0]
    if len(active) == 0:
        return np.nan
    return (active > 0).sum() / len(active)


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute a standard set of performance metrics.

    Parameters
    ----------
    df : Output of run_backtest() (must contain 'strategy_returns',
         'equity_curve', 'returns', 'buy_hold_curve').

    Returns
    -------
    metrics : Dict with strategy and buy-and-hold metrics side by side.
    """
    s_ret = df["strategy_returns"].dropna()
    bh_ret = df["return_1d"].dropna()

    s_equity  = df["equity_curve"].dropna()
    bh_equity = df["buy_hold_curve"].dropna()

    metrics = {
        # ── Strategy ──────────────────────────────────────────────────────────
        "Strategy | Total Return (%)":      round(100 * (s_equity.iloc[-1] / INITIAL_CAPITAL - 1), 2),
        "Strategy | Ann. Return (%)":       round(100 * _annualised_return(s_ret), 2),
        "Strategy | Sharpe Ratio":          round(_sharpe_ratio(s_ret), 3),
        "Strategy | Calmar Ratio":          round(_calmar_ratio(s_ret, s_equity), 3),
        "Strategy | Max Drawdown (%)":      round(100 * _max_drawdown(s_equity), 2),
        "Strategy | Win Rate (%)":          round(100 * _win_rate(s_ret), 2),
        "Strategy | Ann. Volatility (%)":   round(100 * s_ret.std() * np.sqrt(TRADING_DAYS), 2),

        # ── Buy & Hold ────────────────────────────────────────────────────────
        "Buy&Hold | Total Return (%)":      round(100 * (bh_equity.iloc[-1] / INITIAL_CAPITAL - 1), 2),
        "Buy&Hold | Ann. Return (%)":       round(100 * _annualised_return(bh_ret), 2),
        "Buy&Hold | Sharpe Ratio":          round(_sharpe_ratio(bh_ret), 3),
        "Buy&Hold | Max Drawdown (%)":      round(100 * _max_drawdown(bh_equity), 2),
    }

    # Pretty print
    print("=" * 52)
    print("  PERFORMANCE METRICS")
    print("=" * 52)
    width = max(len(k) for k in metrics) + 2
    for k, v in metrics.items():
        print(f"  {k:<{width}} {v}")
    print("=" * 52, "\n")

    return metrics


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, save_path: str = None) -> None:
    """
    4-panel results dashboard:
      1. Equity curve vs buy-and-hold
      2. Daily strategy returns
      3. Drawdown
      4. Rolling Sharpe (63-day)
    """
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45)

    # Panel 1 — Equity curves
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df["equity_curve"],   label="Strategy",     color="#2ecc71",  lw=1.5)
    ax1.plot(df.index, df["buy_hold_curve"], label="Buy & Hold",   color="#3498db",  lw=1.2, ls="--")
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # Panel 2 — Daily returns
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(df.index, df["strategy_returns"], color="#2ecc71", alpha=0.6, width=1)
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_title("Daily Strategy Returns")
    ax2.set_ylabel("Return")
    ax2.grid(alpha=0.3)

    # Panel 3 — Drawdown
    rolling_max = df["equity_curve"].cummax()
    drawdown    = (df["equity_curve"] - rolling_max) / rolling_max * 100

    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(df.index, drawdown, 0, color="#e74c3c", alpha=0.6, label="Drawdown")
    ax3.set_title("Drawdown (%)")
    ax3.set_ylabel("%")
    ax3.grid(alpha=0.3)

    # Panel 4 — Rolling Sharpe (63-day ≈ 1 quarter)
    window = 63
    roll_sharpe = (
        df["strategy_returns"].rolling(window).mean() /
        df["strategy_returns"].rolling(window).std()
    ) * np.sqrt(TRADING_DAYS)

    ax4 = fig.add_subplot(gs[3])
    ax4.plot(df.index, roll_sharpe, color="#9b59b6", lw=1.2, label="63d Rolling Sharpe")
    ax4.axhline(0, color="black", lw=0.7, ls="--")
    ax4.axhline(1, color="#2ecc71", lw=0.7, ls=":", alpha=0.7)
    ax4.set_title("Rolling Sharpe Ratio (63-day)")
    ax4.set_ylabel("Sharpe")
    ax4.legend(loc="upper left")
    ax4.grid(alpha=0.3)

    plt.suptitle("Regime-Aware ML Strategy — Backtest Results", fontsize=13, y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[backtest] Results chart saved → {save_path}")

    plt.show()