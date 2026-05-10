"""
src/regime.py
-------------
Unsupervised regime detection using KMeans clustering.

Pipeline position: Step 3 — runs after features.py
Input:  DataFrame with engineered features (returns, volatility, etc.)
Output: Same DataFrame with a 'regime' column added (0, 1, 2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ── Constants ──────────────────────────────────────────────────────────────────

REGIME_FEATURES = ["return_1d", "vol_21d", "rsi_14", "macd_hist"]
N_REGIMES       = 3   # bull / sideways / bear
RANDOM_STATE    = 42

# Human-readable labels assigned AFTER clustering (mapped by mean return rank)
REGIME_LABELS   = {0: "Bear", 1: "Sideways", 2: "Bull"}
REGIME_COLORS   = {"Bear": "#e74c3c", "Sideways": "#f39c12", "Bull": "#2ecc71"}


# ── Core functions ─────────────────────────────────────────────────────────────

def fit_regime_model(df: pd.DataFrame, features: list[str] = None, n_regimes: int = N_REGIMES) -> tuple:
    """
    Fit a KMeans model on the selected feature columns.

    Parameters
    ----------
    df        : DataFrame with engineered features (no NaNs in selected columns).
    features  : List of column names to cluster on. Defaults to REGIME_FEATURES.
    n_regimes : Number of clusters (market regimes).

    Returns
    -------
    kmeans    : Fitted KMeans object.
    scaler    : Fitted StandardScaler (needed to transform new data consistently).
    features  : The feature list actually used.
    """
    if features is None:
        features = [f for f in REGIME_FEATURES if f in df.columns]

    if not features:
        raise ValueError(
            f"None of the default regime features {REGIME_FEATURES} were found in the DataFrame. "
            "Pass an explicit `features` list or update REGIME_FEATURES."
        )

    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_regimes, random_state=RANDOM_STATE, n_init="auto")
    kmeans.fit(X_scaled)

    score = silhouette_score(X_scaled, kmeans.labels_)
    print(f"[regime] KMeans fitted | k={n_regimes} | silhouette score={score:.4f}")

    return kmeans, scaler, features


def assign_regimes(df: pd.DataFrame, kmeans: KMeans, scaler: StandardScaler, features: list[str]) -> pd.DataFrame:
    """
    Assign a regime label to every row in df.

    Clusters are re-labelled so that:
      - 0 → Bear   (lowest mean return cluster)
      - 1 → Sideways
      - 2 → Bull   (highest mean return cluster)

    Parameters
    ----------
    df     : DataFrame (may contain NaNs — they receive NaN regime).
    kmeans : Fitted KMeans object.
    scaler : Fitted StandardScaler.
    features : Feature columns used during fitting.

    Returns
    -------
    df with new column 'regime' (int 0/1/2) and 'regime_label' (str).
    """
    df = df.copy()

    valid_mask = df[features].notna().all(axis=1)
    X_scaled   = scaler.transform(df.loc[valid_mask, features])
    raw_labels = kmeans.predict(X_scaled)

    # Map raw cluster IDs → ordered regime IDs (by mean return, ascending)
    temp         = df.loc[valid_mask].copy()
    temp["_raw"] = raw_labels

    if "returns" in features:
        mean_returns  = temp.groupby("_raw")["returns"].mean().sort_values()
    else:
        # Fall back: use first feature if returns not available
        mean_returns = temp.groupby("_raw")[features[0]].mean().sort_values()

    raw_to_ordered = {raw: ordered for ordered, raw in enumerate(mean_returns.index)}
    ordered_labels = np.vectorize(raw_to_ordered.get)(raw_labels)

    df["regime"]       = pd.NA
    df["regime_label"] = None

    df.loc[valid_mask, "regime"]       = ordered_labels
    df.loc[valid_mask, "regime_label"] = [REGIME_LABELS[r] for r in ordered_labels]

    df["regime"] = df["regime"].astype("Int64")  # nullable integer

    counts = df["regime_label"].value_counts()
    print(f"[regime] Regime distribution:\n{counts.to_string()}\n")

    return df


def detect_regimes(df: pd.DataFrame, features: list[str] = None, n_regimes: int = N_REGIMES) -> tuple:
    """
    Convenience wrapper: fit + assign in one call.

    Returns
    -------
    df_with_regimes, kmeans, scaler, features_used
    """
    kmeans, scaler, features_used = fit_regime_model(df, features, n_regimes)
    df_out = assign_regimes(df, kmeans, scaler, features_used)
    return df_out, kmeans, scaler, features_used


# ── Diagnostics ────────────────────────────────────────────────────────────────

def plot_regimes(df: pd.DataFrame, price_col: str = "Close", save_path: str = None) -> None:
    """
    Plot price series with regime background shading.

    Parameters
    ----------
    df         : DataFrame with 'regime_label' column and a price column.
    price_col  : Column to plot (default 'close').
    save_path  : If given, saves the figure to this path.
    """
    if "regime_label" not in df.columns:
        raise ValueError("Run detect_regimes() first to add a 'regime_label' column.")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df[price_col], color="#2c3e50", linewidth=1.0, label=price_col.title())

    # Shade background by regime
    prev_label = None
    start_idx  = df.index[0]

    for idx, label in df["regime_label"].items():
        if label != prev_label:
            if prev_label is not None:
                color = REGIME_COLORS.get(prev_label, "gray")
                ax.axvspan(start_idx, idx, alpha=0.15, color=color, linewidth=0)
            start_idx  = idx
            prev_label = label

    # Final segment
    if prev_label is not None:
        color = REGIME_COLORS.get(prev_label, "gray")
        ax.axvspan(start_idx, df.index[-1], alpha=0.15, color=color, linewidth=0)

    # Legend
    patches = [
        mpatches.Patch(color=REGIME_COLORS[label], label=label, alpha=0.5)
        for label in REGIME_COLORS
    ]
    ax.legend(handles=patches + ax.get_lines(), loc="upper left")
    ax.set_title("Price with Market Regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[regime] Chart saved → {save_path}")

    plt.show()


def regime_return_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return descriptive return statistics per regime.
    Useful for verifying that regimes are economically meaningful.
    """
    if "regime_label" not in df.columns or "return_1d" not in df.columns:
        raise ValueError("DataFrame must have 'regime_label' and 'return_1d' columns.")

    stats = (
        df.groupby("regime_label")["return_1d"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "Mean Return", "std": "Volatility", "count": "Days"})
        .sort_values("Mean Return")
    )
    stats["Sharpe (approx)"] = stats["Mean Return"] / stats["Volatility"]
    print("\n[regime] Return stats by regime:")
    print(stats.to_string())
    return stats