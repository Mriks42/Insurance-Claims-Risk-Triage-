"""
Seasonality Analysis Module
============================
Evaluates the model separately on each calendar month's claims.

This is a SEASONALITY view, not model decay over time. Claims are grouped by
month-of-year and POOLED ACROSS 1994-1996 — December here means every December
in the dataset, not a point on a timeline. The dataset is a single static
historical collection split randomly, so there is no chronological axis along
which this model could decay; monitoring.py covers the drift question.

Month-level samples are small (~130 claims and 3-13 fraud cases per month in the
test split), so ranking metrics are suppressed for months with fewer than
MIN_FRAUD_FOR_METRICS fraud cases.

Shows:
- PR-AUC per month (where the sample supports it)
- Fraud rate per month
- Risk score distribution by month

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

# Minimum fraud cases in a month before ranking metrics (PR-AUC, Precision@5%)
# are considered meaningful. Below this they are reported as None.
MIN_FRAUD_FOR_METRICS = 5

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
    """
    Compute performance metrics for a single month.

    Months with fewer than MIN_FRAUD_FOR_METRICS fraud cases return the row with
    pr_auc / precision_5pct set to None rather than a number. In the test split
    the smallest months hold only 3-4 fraud cases out of ~130 claims, and
    Precision@5% is computed over k=5-7 claims — a PR-AUC on 3 positives is
    noise, and plotting it as if it were a seasonal signal invites the wrong
    conclusion. The row is still returned so the sample size stays visible.
    """
    scores = month_data["risk_scores"]
    y      = month_data["y_true"]
    n      = len(y)

    if n < 10 or y.sum() == 0:
        return None

    n_fraud    = int(y.sum())
    sufficient = n_fraud >= MIN_FRAUD_FOR_METRICS

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
        "n_fraud":        n_fraud,
        "sufficient":     sufficient,
        "fraud_rate":     round(float(y.mean()), 4),
        "pr_auc":         round(pr_auc, 4)  if sufficient else None,
        "roc_auc":        round(roc_auc, 4) if sufficient else None,
        "precision_5pct": round(prec_5, 4)  if sufficient else None,
        "avg_risk_score": round(float(scores.mean()), 4),
        "siu_fraud_rate": round(siu_fraud_rate, 4),
    }


# ============================================================
# Full temporal analysis
# ============================================================
def run_temporal_analysis(df_raw, risk_scores, y_true, write_outputs=True):
    """
    Run per-month (seasonality) performance analysis.

    Args:
        write_outputs: Write the metrics CSV. False when called from the
                       dashboard, so rendering a page has no side effects.

    Returns:
        DataFrame of monthly metrics
    """
    print("Running seasonality analysis...")

    months = split_by_month(df_raw, risk_scores, y_true)
    print(f"  Found {len(months)} months of data")

    rows = []
    for m in months:
        metrics = month_metrics(m)
        if metrics:
            rows.append(metrics)
            if metrics["sufficient"]:
                print(f"  {metrics['month']:>3}: n={metrics['n']:>4} "
                      f"fraud={metrics['n_fraud']:>3} | "
                      f"PR-AUC={metrics['pr_auc']:.4f} | "
                      f"Prec@5%={metrics['precision_5pct']:.4f}")
            else:
                print(f"  {metrics['month']:>3}: n={metrics['n']:>4} "
                      f"fraud={metrics['n_fraud']:>3} | metrics suppressed "
                      f"(< {MIN_FRAUD_FOR_METRICS} fraud cases)")

    df_metrics = pd.DataFrame(rows).sort_values("month_num").reset_index(drop=True)
    if write_outputs:
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
    """
    Return a summary dict for dashboard display.

    Best/worst month are taken only from months with enough fraud cases for the
    ranking metric to mean anything — otherwise the "worst month" is simply
    whichever small month happened to draw an unlucky handful of claims.
    """
    if df_metrics.empty:
        return {}

    scored = df_metrics[df_metrics["pr_auc"].notna()]
    if scored.empty:
        return {
            "n_months":        len(df_metrics),
            "n_months_scored": 0,
            "mean_fraud_rate": round(float(df_metrics["fraud_rate"].mean()), 4),
            "fraud_rate_std":  round(float(df_metrics["fraud_rate"].std()), 4),
        }

    best_month  = scored.loc[scored["pr_auc"].idxmax(), "month"]
    worst_month = scored.loc[scored["pr_auc"].idxmin(), "month"]

    return {
        "n_months":        len(df_metrics),
        "n_months_scored": len(scored),
        "mean_pr_auc":     round(float(scored["pr_auc"].mean()), 4),
        "std_pr_auc":      round(float(scored["pr_auc"].std()), 4),
        "best_month":      best_month,
        "best_pr_auc":     round(float(scored["pr_auc"].max()), 4),
        "worst_month":     worst_month,
        "worst_pr_auc":    round(float(scored["pr_auc"].min()), 4),
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

    from data_pipeline import build_model_dataset, load_best_model, raw_rows_for

    data  = build_model_dataset()
    model = load_best_model()

    risk_scores = model.predict_proba(data["X_test_t"])[:, 1]
    raw_test    = raw_rows_for(data, "test")

    df_metrics = run_temporal_analysis(raw_test, risk_scores, data["y_test"].values)
    plot_temporal_charts(df_metrics)

    summary = get_temporal_summary(df_metrics)
    print(f"\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\n[OK] Temporal analysis complete. Outputs in: {TEMPORAL_DIR}")
