"""
src/model.py
------------
Supervised ML model to predict next-period price direction.

Pipeline position: Step 4 — runs after regime.py
Input:  DataFrame with features + 'regime' column
Output: DataFrame with 'signal' column (1 = buy, -1 = sell, 0 = hold)
        + trained model objects
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[model] XGBoost not found — using RandomForest only.")


# ── Constants ──────────────────────────────────────────────────────────────────

# Features fed to the supervised model (must exist in your DataFrame)
MODEL_FEATURES = [
    "return_1d", "return_5d", "return_21d",
    "vol_21d", "vol_63d", "vol_ratio",
    "rsi_14", "roc_10", "macd_hist",
    "sma_ratio_50_200", "price_to_sma50",
    "volume_zscore", "regime",
]

LABEL_HORIZON   = 5      # predict return over next N bars
LABEL_THRESHOLD = 0.002  # ±0.2 % dead-band → label 0 (hold)
RANDOM_STATE    = 42
N_SPLITS        = 5      # TimeSeriesSplit folds


# ── Label engineering ──────────────────────────────────────────────────────────

def make_labels(df: pd.DataFrame, horizon: int = LABEL_HORIZON,
                threshold: float = LABEL_THRESHOLD) -> pd.DataFrame:
    """
    Create a ternary target variable:
      +1  → forward return > +threshold  (Buy)
       0  → |forward return| ≤ threshold  (Hold)
      -1  → forward return < -threshold  (Sell)

    Parameters
    ----------
    df        : DataFrame with a 'close' or 'returns' column.
    horizon   : Number of bars ahead to compute the forward return.
    threshold : Dead-band size (in decimal, e.g. 0.002 = 0.2 %).

    Returns
    -------
    df with a new 'target' column (drops the last `horizon` rows where
    the forward return cannot be computed).
    """
    df = df.copy()

    if "Close" in df.columns:
        fwd_return = df["Close"].pct_change(horizon).shift(-horizon)
    elif "returns" in df.columns:
        fwd_return = df["returns"].rolling(horizon).sum().shift(-horizon)
    else:
        raise ValueError("DataFrame must contain a 'close' or 'returns' column.")

    df["fwd_return"] = fwd_return
    df["target"] = 0
    df.loc[fwd_return >  threshold, "target"] =  1
    df.loc[fwd_return < -threshold, "target"] = -1

    df = df.dropna(subset=["target", "fwd_return"])
    df["target"] = df["target"].astype(int)

    dist = df["target"].value_counts().sort_index()
    print(f"[model] Target distribution (horizon={horizon}, threshold={threshold}):")
    print(dist.to_string(), "\n")

    return df


# ── Model training ─────────────────────────────────────────────────────────────

def _get_model(model_type: str = "rf") -> object:
    """Return an untrained classifier instance."""
    if model_type == "xgb":
        if not XGBOOST_AVAILABLE:
            print("[model] XGBoost unavailable, falling back to RandomForest.")
            model_type = "rf"
        else:
            return XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
            )

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",   # handles imbalanced buy/sell/hold
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_model(df: pd.DataFrame, features: list[str] = None,
                model_type: str = "rf") -> tuple:
    """
    Train a classifier using walk-forward (TimeSeriesSplit) cross-validation,
    then refit on the full training set.

    Parameters
    ----------
    df         : DataFrame with features and a 'target' column.
    features   : Feature column names. Defaults to MODEL_FEATURES (filtered to
                 columns that actually exist in df).
    model_type : 'rf' (RandomForest) or 'xgb' (XGBoost).

    Returns
    -------
    model      : Fitted classifier.
    scaler     : Fitted StandardScaler.
    features   : Feature list actually used.
    cv_scores  : List of per-fold accuracy scores.
    """
    if features is None:
        features = [f for f in MODEL_FEATURES if f in df.columns]

    if not features:
        raise ValueError("No valid model features found in DataFrame.")

    df_clean = df[features + ["target"]].dropna()

    X = df_clean[features].values
    y = df_clean["target"].values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Walk-forward cross-validation
    tscv      = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_scores = []

    print(f"[model] Running {N_SPLITS}-fold walk-forward CV with {model_type.upper()}...")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        clf = _get_model(model_type)
        clf.fit(X_scaled[train_idx], y[train_idx])
        preds      = clf.predict(X_scaled[test_idx])
        fold_score = accuracy_score(y[test_idx], preds)
        cv_scores.append(fold_score)
        print(f"  Fold {fold + 1}: accuracy = {fold_score:.4f}")

    print(f"[model] Mean CV accuracy = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n")

    # Final model — trained on ALL data (will be used for live prediction)
    model = _get_model(model_type)
    model.fit(X_scaled, y)

    # Full-sample report
    preds_full = model.predict(X_scaled)
    print("[model] Classification report (full sample — informational only):")
    print(classification_report(y, preds_full, target_names=["Sell(-1)", "Hold(0)", "Buy(+1)"]))

    return model, scaler, features, cv_scores


# ── Prediction ─────────────────────────────────────────────────────────────────

def generate_predictions(df: pd.DataFrame, model, scaler: StandardScaler,
                         features: list[str]) -> pd.DataFrame:
    """
    Run the trained model on the full DataFrame to produce a 'signal' column.

    Parameters
    ----------
    df      : DataFrame with feature columns (may contain NaNs).
    model   : Fitted classifier.
    scaler  : Fitted StandardScaler.
    features: Feature list used during training.

    Returns
    -------
    df with 'signal' column: +1 (Buy), 0 (Hold), -1 (Sell).
    """
    df = df.copy()
    valid_mask = df[features].notna().all(axis=1)

    X_scaled = scaler.transform(df.loc[valid_mask, features])
    preds    = model.predict(X_scaled)

    df["signal"] = np.nan
    df.loc[valid_mask, "signal"] = preds
    df["signal"] = df["signal"].astype("Int64")

    return df


# ── Diagnostics ────────────────────────────────────────────────────────────────

def plot_feature_importance(model, features: list[str],
                            top_n: int = 15, save_path: str = None) -> None:
    """
    Bar chart of feature importances (RandomForest or XGBoost).
    """
    if not hasattr(model, "feature_importances_"):
        print("[model] Model does not expose feature_importances_ — skipping plot.")
        return

    importances = pd.Series(model.feature_importances_, index=features)
    importances = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="#3498db")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[model] Feature importance chart saved → {save_path}")

    plt.show()