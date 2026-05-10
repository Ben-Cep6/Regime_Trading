"""
main.py
-------
End-to-end pipeline runner for the Regime-Aware ML Trading System.

Run:  python main.py
"""

import os
import pandas as pd

# ── Internal modules ───────────────────────────────────────────────────────────
from src.data_loader import get_data          # you already have this
from src.features    import build_features  # you already have this
from src.regime      import detect_regimes, plot_regimes, regime_return_stats
from src.model       import make_labels, train_model, generate_predictions, plot_feature_importance
from src.strategy    import generate_positions, apply_regime_override, add_transaction_costs
from src.backtest    import run_backtest, compute_metrics, plot_results


# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    "ticker":       "SPY",
    "start":        "2015-01-01",
    "end":          "2024-01-01",
    "model_type":   "rf",         # "rf" or "xgb"
    "allow_short":  False,
    "bear_exit":    True,
    "cost_per_trade": 0.001,      # 0.1% round-trip

    # Paths
    "data_dir":     "data",
    "results_dir":  "results",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def ensure_dirs(config: dict) -> None:
    for key in ("data_dir", "results_dir"):
        os.makedirs(config[key], exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path)
    print(f"[main] Saved → {path}")


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(config: dict = CONFIG) -> dict:
    ensure_dirs(config)
    print("\n" + "=" * 60)
    print("  REGIME-AWARE ML TRADING SYSTEM")
    print("=" * 60 + "\n")

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    print("── Step 1: Loading data ──────────────────────────────────────")
    df_raw = get_data(
        ticker=config["ticker"],
        start=config["start"],
        end=config["end"],
    )
    save_csv(df_raw, os.path.join(config["data_dir"], "raw.csv"))

    # ── Step 2: Feature engineering ────────────────────────────────────────────
    print("\n── Step 2: Engineering features ──────────────────────────────")
    df_features = build_features(df_raw)
    save_csv(df_features, os.path.join(config["data_dir"], "features.csv"))

    # ── Step 3: Regime detection ───────────────────────────────────────────────
    print("\n── Step 3: Detecting market regimes ──────────────────────────")
    df_regimes, kmeans, regime_scaler, regime_features = detect_regimes(df_features)

    regime_return_stats(df_regimes)
    plot_regimes(
        df_regimes,
        save_path=os.path.join(config["results_dir"], "regimes.png"),
    )
    save_csv(df_regimes, os.path.join(config["data_dir"], "regimes.csv"))

    # ── Step 4: ML model ───────────────────────────────────────────────────────
    print("\n── Step 4: Training prediction model ─────────────────────────")
    df_labelled = make_labels(df_regimes)

    model, model_scaler, model_features, cv_scores = train_model(
        df_labelled,
        model_type=config["model_type"],
    )
    plot_feature_importance(
        model,
        model_features,
        save_path=os.path.join(config["results_dir"], "feature_importance.png"),
    )

    df_signals = generate_predictions(df_labelled, model, model_scaler, model_features)

    # ── Step 5: Strategy ───────────────────────────────────────────────────────
    print("\n── Step 5: Generating positions ──────────────────────────────")
    df_positions = generate_positions(df_signals, allow_short=config["allow_short"])

    if config["bear_exit"]:
        df_positions = apply_regime_override(df_positions)

    df_positions = add_transaction_costs(df_positions, cost_per_trade=config["cost_per_trade"])

    # ── Step 6 & 7: Backtest + Evaluation ─────────────────────────────────────
    print("\n── Step 6 & 7: Backtesting & Evaluation ──────────────────────")
    df_results  = run_backtest(df_positions)
    metrics     = compute_metrics(df_results)
    plot_results(
        df_results,
        save_path=os.path.join(config["results_dir"], "backtest.png"),
    )

    save_csv(df_results, os.path.join(config["results_dir"], "backtest_results.csv"))

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Pipeline complete ─────────────────────────────────────────")
    print(f"  Ticker:   {config['ticker']}")
    print(f"  Period:   {config['start']} → {config['end']}")
    print(f"  Model:    {config['model_type'].upper()}")
    print(f"  Sharpe:   {metrics.get('Strategy | Sharpe Ratio', 'N/A')}")
    print(f"  Max DD:   {metrics.get('Strategy | Max Drawdown (%)', 'N/A')}%")
    print(f"  Results saved in: {config['results_dir']}/\n")

    return {
        "df":      df_results,
        "metrics": metrics,
        "model":   model,
        "kmeans":  kmeans,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting pipeline...")
    results = run_pipeline(CONFIG)