"""
Model Monitoring & Drift Detection Module
==========================================
Simulates production monitoring by splitting the test set into
time-ordered batches and checking for data/prediction drift
using two-sample KS tests with Benjamini-Hochberg correction.

What it detects:
- Data drift: input feature distributions shifting over time
- Prediction drift: model output distribution changing
- Target drift: actual fraud rate changing

Usage:
    python monitoring.py
    from monitoring import compute_drift_report, get_drift_summary
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUTS_DIR

MONITORING_DIR = os.path.join(OUTPUTS_DIR, "monitoring")
os.makedirs(MONITORING_DIR, exist_ok=True)

N_BATCHES = 6   # split test set into 6 batches (simulates 6 weeks/months)


# ============================================================
# Split data into time-ordered batches
# ============================================================
def create_batches(df_raw, risk_scores, y_true, n_batches=N_BATCHES):
    """
    Split data into n_batches in chronological order. Simulates claims arriving
    over time.

    Ordering is (Year, Month), not Month alone. This dataset spans 1994-1996 and
    every calendar month contains claims from all three years, so sorting by
    month name alone interleaved 1994/1995/1996 within each batch — the batches
    were labelled "time-ordered" but weren't, and the Year feature then shifted
    across batches purely as an artifact of the ordering.

    Returns list of batch dicts.
    """
    df = df_raw.copy().reset_index(drop=True)
    df["_risk_score"] = risk_scores
    df["_actual"]     = np.array(y_true)

    month_order = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    if "Month" in df.columns:
        df["_month_num"] = df["Month"].map(month_order).fillna(0)
        sort_keys = (["Year", "_month_num"] if "Year" in df.columns
                     else ["_month_num"])
        df = df.sort_values(sort_keys).reset_index(drop=True)

    batch_size = len(df) // n_batches
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end   = start + batch_size if i < n_batches - 1 else len(df)
        batch_df = df.iloc[start:end].copy()
        batches.append({
            "batch_id":    i + 1,
            "label":       f"Batch {i+1}",
            "df":          batch_df,
            "risk_scores": batch_df["_risk_score"].values,
            "y_true":      batch_df["_actual"].values,
            "n":           len(batch_df),
        })

    return batches


# ============================================================
# Per-batch statistics
# ============================================================
def batch_statistics(batch):
    """
    Compute key statistics for a single batch.

    Buckets are assigned by RANK, matching compute_triage_summary(), the API and
    the Summary dashboard. np.percentile interpolates between neighbouring
    scores, which put 13 claims in a "top 5%" bucket of 257 where the rank cut
    takes 12 — a small discrepancy, but the whole point of standardising on rank
    was that one definition drives every surface.
    """
    scores = batch["risk_scores"]
    y      = batch["y_true"]

    from config import PCT_SIU, PCT_MANUAL
    n          = len(scores)
    siu_cut    = int(PCT_SIU * n)
    manual_cut = int((PCT_SIU + PCT_MANUAL) * n)

    rank_of = np.empty(n, dtype=int)
    rank_of[np.argsort(scores)[::-1]] = np.arange(n)
    siu_mask    = rank_of < siu_cut
    manual_mask = (rank_of >= siu_cut) & (rank_of < manual_cut)

    ordered       = np.sort(scores)[::-1]
    siu_thresh    = float(ordered[max(0, siu_cut - 1)])
    manual_thresh = float(ordered[max(0, manual_cut - 1)])

    return {
        "batch_id":       batch["batch_id"],
        "label":          batch["label"],
        "n":              batch["n"],
        "fraud_rate":     float(y.mean()),
        "avg_risk_score": float(scores.mean()),
        "std_risk_score": float(scores.std()),
        "siu_count":      int(siu_mask.sum()),
        "manual_count":   int(manual_mask.sum()),
        "siu_fraud_rate": float(y[siu_mask].mean()) if siu_mask.sum() > 0 else 0,
        "p25_score":      float(np.percentile(scores, 25)),
        "p50_score":      float(np.percentile(scores, 50)),
        "p75_score":      float(np.percentile(scores, 75)),
        "p95_score":      float(np.percentile(scores, 95)),
    }


# ============================================================
# Drift detection (KS test + Benjamini-Hochberg)
# ============================================================
def compute_drift_report(reference_df, current_df, feature_cols,
                          batch_label="batch"):
    """
    Compute data drift between reference (training) and current (batch) data.

    One implementation, one answer: KS tests with Benjamini-Hochberg correction.

    This used to prefer Evidently and fall back to KS only when Evidently was
    absent. Since requirements.txt pinned no upper bound and Evidently 0.7
    removed `evidently.report`, the branch taken depended on whichever version
    pip happened to resolve — so the deployed Space and a local machine produced
    different Drift Detection tables from identical code, with no error to
    indicate it. Evidently applies no multiple-comparison correction and uses
    its own dataset-level threshold, so the two paths could not agree even in
    principle.

    Evidently has been removed rather than demoted to HTML rendering: it was no
    longer load-bearing, and an unused dependency that can silently change
    behaviour is worse than no dependency.

    Returns drift summary dict.
    """
    return _manual_drift_check(reference_df, current_df, feature_cols, batch_label)


def _manual_drift_check(reference_df, current_df, feature_cols, batch_label,
                         alpha=0.05):
    """
    Drift detection via two-sample KS tests, with Benjamini-Hochberg correction.

    Correction matters here: with 7 features across 5 batches this runs 35
    hypothesis tests, so at alpha=0.05 roughly 1.75 features are expected to
    look "drifted" by chance alone. The old version flagged the whole dataset as
    drifted if ANY single raw p-value fell below 0.05, which made the warning
    close to guaranteed regardless of the data. `dataset_drift` is now based on
    the BH-adjusted p-values; raw counts are still reported for comparison.
    """
    from scipy import stats

    shared_cols = [c for c in feature_cols
                   if c in reference_df.columns and c in current_df.columns]
    num_cols = reference_df[shared_cols].select_dtypes(include=[np.number]).columns.tolist()

    tested, p_values = [], []
    for col in num_cols:
        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values
        if len(ref_vals) > 10 and len(cur_vals) > 10:
            _, p_val = stats.ks_2samp(ref_vals, cur_vals)
            tested.append(col)
            p_values.append(float(p_val))

    drifted_raw = [c for c, p in zip(tested, p_values) if p < alpha]

    # Benjamini-Hochberg. scipy.stats.false_discovery_control landed in 1.11;
    # fall back to uncorrected p-values on older installs rather than failing.
    if p_values:
        try:
            adjusted = list(stats.false_discovery_control(p_values, method="bh"))
        except AttributeError:
            adjusted = p_values
    else:
        adjusted = []
    drifted = [c for c, p in zip(tested, adjusted) if p < alpha]

    return {
        "batch_label":            batch_label,
        "dataset_drift":          len(drifted) > 0,
        "drift_share":            len(drifted) / max(1, len(tested)),
        "n_drifted_features":     len(drifted),
        "n_drifted_uncorrected":  len(drifted_raw),
        "n_features":             len(tested),
        "drifted_features":       drifted,
        "tested_features":        tested,
        "html_report":            None,
    }


# ============================================================
# Full monitoring pipeline
# ============================================================
# Features excluded from drift testing.
#   Year      — batches are ordered BY year, so testing it is circular: it is
#               guaranteed to "drift" as a direct consequence of the batching.
#   RepNumber — a claims-rep identifier, not a measured quantity; a shift in it
#               reflects staffing, not the data distribution the model sees.
DRIFT_EXCLUDE = ["Year", "RepNumber"]


def run_monitoring(df_raw, risk_scores, y_true, reference_df=None,
                   drift_exclude=None, write_outputs=True):
    """
    Run the full monitoring pipeline.

    Args:
        df_raw:        Raw test dataframe
        risk_scores:   Model predictions on the test set
        y_true:        Actual labels
        reference_df:  Reference dataframe for drift comparison. Pass the RAW
                       TRAINING ROWS — that is what "has the incoming data
                       drifted from what the model was trained on?" actually
                       means. Falling back to batch 1 (the old default) only
                       compares held-out data against itself.
        drift_exclude: Feature names to skip (defaults to DRIFT_EXCLUDE).
        write_outputs: Write CSVs to outputs/monitoring/. False when called from
                       the dashboard, so rendering a page has no side effects.

    Returns:
        dict with batch_stats, drift_results, batches and reference_kind
    """
    print("Running monitoring pipeline...")

    batches = create_batches(df_raw, risk_scores, y_true)
    print(f"  Created {len(batches)} batches of ~{batches[0]['n']} claims each")

    batch_stats = [batch_statistics(b) for b in batches]
    stats_df = pd.DataFrame(batch_stats)
    if write_outputs:
        stats_df.to_csv(os.path.join(MONITORING_DIR, "batch_statistics.csv"), index=False)

    if reference_df is not None:
        ref_df, reference_kind = reference_df, "training split"
    else:
        ref_df, reference_kind = batches[0]["df"], "batch 1 (no reference supplied)"
    print(f"  Drift reference: {reference_kind}")

    exclude = DRIFT_EXCLUDE if drift_exclude is None else drift_exclude
    num_feature_cols = [c for c in df_raw.columns
                        if df_raw[c].dtype in [np.float64, np.int64, float, int]
                        and c not in ["FraudFound_P", "PolicyNumber"]
                        and c not in exclude]
    print(f"  Drift features ({len(num_feature_cols)}): {num_feature_cols}")
    if exclude:
        print(f"  Excluded: {exclude}")

    drift_results = []
    for batch in batches:   # every batch is compared against the reference
        drift = compute_drift_report(
            ref_df, batch["df"],
            feature_cols=num_feature_cols,
            batch_label=batch["label"].replace(" ", "_").lower(),
        )
        drift_results.append(drift)
        status = "⚠️ DRIFT" if drift["dataset_drift"] else "✅ Stable"
        print(f"  {batch['label']}: {status} "
              f"({drift['n_drifted_features']}/{drift['n_features']} features drifted"
              f", {drift.get('n_drifted_uncorrected', '?')} before BH correction)")

    if write_outputs:
        drift_df = pd.DataFrame([{k: v for k, v in d.items() if k != "html_report"}
                                  for d in drift_results])
        drift_df.to_csv(os.path.join(MONITORING_DIR, "drift_summary.csv"), index=False)

    return {
        "batch_stats":    stats_df,
        "drift_results":  drift_results,
        "batches":        batches,
        "reference_kind": reference_kind,
        "drift_features": num_feature_cols,
        "excluded":       exclude,
    }


# ============================================================
# Visualizations
# ============================================================
def plot_monitoring_charts(batch_stats_df, save_dir=MONITORING_DIR):
    """Generate monitoring trend charts."""
    df = batch_stats_df

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Monitoring — Batch Trends Over Time", fontsize=14, fontweight="bold")

    # 1. Fraud rate over time
    ax = axes[0, 0]
    ax.plot(df["label"], df["fraud_rate"], "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax.set_title("Actual Fraud Rate per Batch")
    ax.set_ylabel("Fraud Rate")
    ax.set_ylim(0, max(df["fraud_rate"].max() * 1.5, 0.15))
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30)

    # 2. Average risk score over time
    ax = axes[0, 1]
    ax.plot(df["label"], df["avg_risk_score"], "o-", color="#2563eb", linewidth=2, markersize=8)
    ax.fill_between(range(len(df)),
                    df["avg_risk_score"] - df["std_risk_score"],
                    df["avg_risk_score"] + df["std_risk_score"],
                    alpha=0.2, color="#2563eb")
    ax.set_title("Average Risk Score per Batch (± std)")
    ax.set_ylabel("Risk Score")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=30)
    ax.grid(True, alpha=0.3)

    # 3. SIU count over time
    ax = axes[1, 0]
    ax.bar(df["label"], df["siu_count"], color="#e74c3c", alpha=0.8, label="SIU")
    ax.bar(df["label"], df["manual_count"], bottom=df["siu_count"],
           color="#f39c12", alpha=0.8, label="Manual Review")
    ax.set_title("Flagged Claims per Batch")
    ax.set_ylabel("Count")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30)
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Risk score distribution (box plot style using percentiles)
    ax = axes[1, 1]
    x = range(len(df))
    ax.fill_between(x, df["p25_score"], df["p75_score"],
                    alpha=0.3, color="#2563eb", label="IQR (25-75%)")
    ax.plot(x, df["p50_score"], "o-", color="#2563eb", linewidth=2,
            markersize=6, label="Median")
    ax.plot(x, df["p95_score"], "^--", color="#e74c3c", linewidth=1.5,
            markersize=6, label="95th percentile")
    ax.set_title("Risk Score Distribution per Batch")
    ax.set_ylabel("Risk Score")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=30)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, "monitoring_trends.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def get_drift_summary(drift_results):
    """Return a simple summary dict for dashboard display."""
    if not drift_results:
        return {"any_drift": False, "n_drifted_batches": 0, "details": []}

    n_drifted = sum(1 for d in drift_results if d.get("dataset_drift", False))
    return {
        "any_drift":        n_drifted > 0,
        "n_drifted_batches": n_drifted,
        "total_batches":    len(drift_results),
        "details":          drift_results,
    }


# ============================================================
# Standalone run
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Model Monitoring & Drift Detection")
    print("=" * 60)

    from data_pipeline import build_model_dataset, load_best_model, raw_rows_for

    data  = build_model_dataset()
    model = load_best_model()

    risk_scores = model.predict_proba(data["X_test_t"])[:, 1]
    raw_test    = raw_rows_for(data, "test")
    raw_train   = raw_rows_for(data, "train")   # drift reference

    results = run_monitoring(
        raw_test, risk_scores, data["y_test"].values,
        reference_df=raw_train,
    )

    plot_monitoring_charts(results["batch_stats"])

    print(f"\n[OK] Monitoring complete. Outputs in: {MONITORING_DIR}")
