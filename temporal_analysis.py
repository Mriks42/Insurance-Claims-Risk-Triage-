"""
Temporal Analysis Module
=========================
Simulates how model performance changes over time by splitting
the dataset by Month column and evaluating the model on each
month's claims independently.

Shows:
- PR-AUC per month
- Fraud rate per month
- Risk score distribution shift
- Model decay visualization

Usage:
    python temporal_analysis.py
    from temporal_analysis import run_temporal_analysis
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, roc_auc_score
from config import OUTPUTS_DIR, PCT_SIU, PCT_MANUAL

TEMPORAL_DIR = os.path.join(OUTPUTS_DIR, "temporal")
os.makedirs(TEMPORAL_DIR, exist_ok=True)

MONTH_ORDER = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


# ============================================================
# Split data by month
# ============================================================
def split_by_month(df_raw, risk_scores, y_true):
    """
    Group claims by accident month.
    Returns list of month dicts sorted chronologically.
    """
    df = df_raw.copy().reset_index(drop=True)
    df["_risk_score"] = risk_scores
    df["_actual"]     = np.array(y_true)

    if "Month" not in df.columns:
        print("  'Month' column not found — using row-order batches instead")
        return _split_by_order(df)

    df["_month_num"] = df["Month"].map(MONTH_ORDER).fillna(0)
    months = []

    for month_name, month_num in sorted(MONTH_ORDER.items(), key=lambda x: x[1]):
        mask = df["Month"] == month_name
        sub  = df[mask]
        if len(sub) < 5:
            continue
        months.append({
            "month":       month_name,
            "month_num":   month_num,
            "df":          sub,
            "risk_scores": sub["_risk_score"].values,
            "y_true":      sub["_actual"].values,
            "n":           len(sub),
        })

    return months


def _split_by_order(df, n_splits=12):
    """Fallback: split by row order into n_splits chunks."""
    chunk = len(df) // n_splits
    splits = []
    for i in range(n_splits):
        start = i * chunk
        end   = start + chunk if i < n_splits - 1 else len(df)
        sub   = df.iloc[start:end]
        splits.append({
            "month":       f"Period {i+1}",
            "month_num":   i + 1,
            "df":          sub,
            "risk_scores": sub["_risk_score"].values,
            "y_true":      sub["_actual"].values,
            "n":           len(sub),
        })
    return splits


# ============================================================
# Per-month metrics
# ============================================================
def month_metrics(month_data):
    """Compute performance metrics for a single month."""
    scores = month_data["risk_scores"]
    y      = month_data["y_true"]
    n      = len(y)

    if n < 10 or y.sum() == 0:
        return None

    # PR-AUC and ROC-AUC
    try:
        pr_auc  = float(average_precision_score(y, scores))
        roc_auc = float(roc_auc_score(y, scores))
    except Exception:
        pr_auc  = 0.0
        roc_auc = 0.0

    # Triage buckets
    siu_thresh    = np.percentile(scores, 100 * (1 - PCT_SIU))
    manual_thresh = np.percentile(scores, 100 * (1 - PCT_SIU - PCT_MANUAL))
    siu_mask      = scores >= siu_thresh
    manual_mask   = (scores >= manual_thresh) & ~siu_mask

    siu_fraud_rate = float(y[siu_mask].mean()) if siu_mask.sum() > 0 else 0.0

    # Precision@5%
    k = max(1, int(0.05 * n))
    top_k = np.argsort(scores)[::-1][:k]
    prec_5 = float(y[top_k].mean())

    return {
        "month":          month_data["month"],
        "month_num":      month_data["month_num"],
        "n":              n,
        "fraud_rate":     round(float(y.mean()), 4),
        "pr_auc":         round(pr_auc, 4),
        "roc_auc":        round(roc_auc, 4),
        "precision_5pct": round(prec_5, 4),
        "avg_risk_score": round(float(scores.mean()), 4),
        "siu_fraud_rate": round(siu_fraud_rate, 4),
    }


# ============================================================
# Full temporal analysis
# ============================================================
def run_temporal_analysis(df_raw, risk_scores, y_true):
    """
    Run month-by-month performance analysis.

    Returns:
        DataFrame of monthly metrics
    """
    print("Running temporal analysis...")

    months = split_by_month(df_raw, risk_scores, y_true)
    print(f"  Found {len(months)} months of data")

    rows = []
    for m in months:
        metrics = month_metrics(m)
        if metrics:
            rows.append(metrics)
            print(f"  {metrics['month']:>3}: n={metrics['n']:>4} | "
                  f"fraud={metrics['fraud_rate']:.3f} | "
                  f"PR-AUC={metrics['pr_auc']:.4f} | "
                  f"Prec@5%={metrics['precision_5pct']:.4f}")

    df_metrics = pd.DataFrame(rows).sort_values("month_num").reset_index(drop=True)
    df_metrics.to_csv(os.path.join(TEMPORAL_DIR, "monthly_metrics.csv"), index=False)
    print(f"\n  Saved: {os.path.join(TEMPORAL_DIR, 'monthly_metrics.csv')}")

    return df_metrics


# ============================================================
# Visualizations
# ============================================================
def plot_temporal_charts(df_metrics, save_dir=TEMPORAL_DIR):
    """Generate temporal performance charts."""
    if df_metrics.empty:
        print("  No data to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Temporal Analysis — Model Performance Over Time",
                 fontsize=14, fontweight="bold")

    months = df_metrics["month"].tolist()
    x      = range(len(months))

    # 1. PR-AUC over time
    ax = axes[0, 0]
    ax.plot(x, df_metrics["pr_auc"], "o-", color="#2563eb", linewidth=2, markersize=8)
    overall_mean = df_metrics["pr_auc"].mean()
    ax.axhline(y=overall_mean, color="#9ca3af", linestyle="--",
               label=f"Mean PR-AUC ({overall_mean:.4f})")
    ax.set_title("PR-AUC by Month")
    ax.set_ylabel("PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Fraud rate over time
    ax = axes[0, 1]
    ax.bar(x, df_metrics["fraud_rate"], color="#e74c3c", alpha=0.8)
    ax.axhline(y=df_metrics["fraud_rate"].mean(), color="#9ca3af",
               linestyle="--", label=f"Mean ({df_metrics['fraud_rate'].mean():.3f})")
    ax.set_title("Actual Fraud Rate by Month")
    ax.set_ylabel("Fraud Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Precision@5% over time
    ax = axes[1, 0]
    ax.plot(x, df_metrics["precision_5pct"], "s-", color="#27ae60",
            linewidth=2, markersize=8)
    ax.axhline(y=df_metrics["precision_5pct"].mean(), color="#9ca3af",
               linestyle="--",
               label=f"Mean ({df_metrics['precision_5pct'].mean():.4f})")
    ax.set_title("Precision@5% by Month")
    ax.set_ylabel("Precision@5%")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Average risk score over time
    ax = axes[1, 1]
    ax.plot(x, df_metrics["avg_risk_score"], "D-", color="#f39c12",
            linewidth=2, markersize=8)
    ax.set_title("Average Risk Score by Month")
    ax.set_ylabel("Avg Risk Score")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, "temporal_performance.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def get_temporal_summary(df_metrics):
    """Return a summary dict for dashboard display."""
    if df_metrics.empty:
        return {}

    best_month  = df_metrics.loc[df_metrics["pr_auc"].idxmax(), "month"]
    worst_month = df_metrics.loc[df_metrics["pr_auc"].idxmin(), "month"]

    return {
        "n_months":        len(df_metrics),
        "mean_pr_auc":     round(float(df_metrics["pr_auc"].mean()), 4),
        "std_pr_auc":      round(float(df_metrics["pr_auc"].std()), 4),
        "best_month":      best_month,
        "best_pr_auc":     round(float(df_metrics["pr_auc"].max()), 4),
        "worst_month":     worst_month,
        "worst_pr_auc":    round(float(df_metrics["pr_auc"].min()), 4),
        "mean_fraud_rate": round(float(df_metrics["fraud_rate"].mean()), 4),
        "fraud_rate_std":  round(float(df_metrics["fraud_rate"].std()), 4),
    }


# ============================================================
# Standalone run
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Temporal Analysis")
    print("=" * 60)

    import joblib
    from config import DATA_PATH, TARGET, COLS_TO_DROP, RANDOM_STATE
    from feature_engineering import engineer_features, REPLACED_CATEGORICALS
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df_raw = pd.read_csv(DATA_PATH)
    df_eng = engineer_features(df_raw)

    X = df_eng.drop(columns=[TARGET] + COLS_TO_DROP)
    y = df_eng[TARGET]
    cols_to_remove = [c for c in REPLACED_CATEGORICALS if c in X.columns]
    X_xgb = X.drop(columns=cols_to_remove)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_xgb, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant",
                                                     fill_value="Unknown")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ], remainder="drop"
    )
    preprocessor.fit(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = joblib.load(
        os.path.join(OUTPUTS_DIR, "improvement", "best_model_improved.joblib")
    )
    risk_scores = model.predict_proba(X_test_t)[:, 1]
    raw_test    = df_raw.iloc[y_test.index].reset_index(drop=True)

    df_metrics = run_temporal_analysis(raw_test, risk_scores, y_test.values)
    plot_temporal_charts(df_metrics)

    summary = get_temporal_summary(df_metrics)
    print(f"\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\n[OK] Temporal analysis complete. Outputs in: {TEMPORAL_DIR}")
