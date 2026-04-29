"""
Model Monitoring & Drift Detection Module
==========================================
Simulates production monitoring by splitting the test set into
time-ordered batches and checking for data/prediction drift
using the Evidently library.

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
    Split data into n_batches ordered by Month column (if available),
    otherwise by row order. Simulates claims arriving over time.

    Returns list of batch dicts.
    """
    df = df_raw.copy().reset_index(drop=True)
    df["_risk_score"] = risk_scores
    df["_actual"]     = np.array(y_true)

    # Sort by Month if available for realistic temporal ordering
    month_order = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    if "Month" in df.columns:
        df["_month_num"] = df["Month"].map(month_order).fillna(0)
        df = df.sort_values("_month_num").reset_index(drop=True)

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
    """Compute key statistics for a single batch."""
    scores = batch["risk_scores"]
    y      = batch["y_true"]

    from config import PCT_SIU, PCT_MANUAL
    siu_thresh    = np.percentile(scores, 100 * (1 - PCT_SIU))
    manual_thresh = np.percentile(scores, 100 * (1 - PCT_SIU - PCT_MANUAL))

    siu_mask    = scores >= siu_thresh
    manual_mask = (scores >= manual_thresh) & ~siu_mask

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
# Drift detection (Evidently)
# ============================================================
def compute_drift_report(reference_df, current_df, feature_cols,
                          save_dir=MONITORING_DIR, batch_label="batch"):
    """
    Compute data drift between reference (training) and current (batch) data.
    Uses Evidently's DataDriftPreset.

    Returns drift summary dict.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import DatasetDriftMetric

        # Keep only shared feature columns
        shared_cols = [c for c in feature_cols
                       if c in reference_df.columns and c in current_df.columns]
        ref = reference_df[shared_cols].copy()
        cur = current_df[shared_cols].copy()

        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftPreset(),
        ])
        report.run(reference_data=ref, current_data=cur)

        # Save HTML report
        html_path = os.path.join(save_dir, f"drift_report_{batch_label}.html")
        report.save_html(html_path)

        # Extract summary
        result = report.as_dict()
        drift_metric = result["metrics"][0]["result"]

        return {
            "batch_label":       batch_label,
            "dataset_drift":     drift_metric.get("dataset_drift", False),
            "drift_share":       drift_metric.get("share_of_drifted_columns", 0),
            "n_drifted_features": drift_metric.get("number_of_drifted_columns", 0),
            "n_features":        drift_metric.get("number_of_columns", len(shared_cols)),
            "html_report":       html_path,
        }

    except ImportError:
        # Evidently not installed — compute basic drift manually
        return _manual_drift_check(reference_df, current_df, feature_cols, batch_label)
    except Exception as e:
        print(f"  Evidently error: {e}. Using manual drift check.")
        return _manual_drift_check(reference_df, current_df, feature_cols, batch_label)


def _manual_drift_check(reference_df, current_df, feature_cols, batch_label):
    """
    Fallback drift detection using KS test (no Evidently needed).
    """
    from scipy import stats

    shared_cols = [c for c in feature_cols
                   if c in reference_df.columns and c in current_df.columns]
    num_cols = reference_df[shared_cols].select_dtypes(include=[np.number]).columns.tolist()

    drifted = []
    for col in num_cols:
        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values
        if len(ref_vals) > 10 and len(cur_vals) > 10:
            stat, p_val = stats.ks_2samp(ref_vals, cur_vals)
            if p_val < 0.05:
                drifted.append(col)

    return {
        "batch_label":        batch_label,
        "dataset_drift":      len(drifted) > 0,
        "drift_share":        len(drifted) / max(1, len(num_cols)),
        "n_drifted_features": len(drifted),
        "n_features":         len(num_cols),
        "drifted_features":   drifted,
        "html_report":        None,
    }


# ============================================================
# Full monitoring pipeline
# ============================================================
def run_monitoring(df_raw, risk_scores, y_true, reference_df=None):
    """
    Run the full monitoring pipeline.

    Args:
        df_raw:       Raw test dataframe
        risk_scores:  Model predictions on test set
        y_true:       Actual labels
        reference_df: Reference (training) dataframe for drift comparison.
                      If None, uses first batch as reference.

    Returns:
        dict with batch_stats and drift_results
    """
    print("Running monitoring pipeline...")

    # Create batches
    batches = create_batches(df_raw, risk_scores, y_true)
    print(f"  Created {len(batches)} batches of ~{batches[0]['n']} claims each")

    # Compute per-batch statistics
    batch_stats = [batch_statistics(b) for b in batches]
    stats_df = pd.DataFrame(batch_stats)
    stats_df.to_csv(os.path.join(MONITORING_DIR, "batch_statistics.csv"), index=False)

    # Drift detection (use first batch as reference if no training data provided)
    drift_results = []
    ref_df = reference_df if reference_df is not None else batches[0]["df"]
    num_feature_cols = [c for c in df_raw.columns
                        if df_raw[c].dtype in [np.float64, np.int64, float, int]
                        and c not in ["FraudFound_P", "PolicyNumber"]]

    for batch in batches[1:]:   # compare each batch against reference
        drift = compute_drift_report(
            ref_df, batch["df"],
            feature_cols=num_feature_cols,
            batch_label=batch["label"].replace(" ", "_").lower(),
        )
        drift_results.append(drift)
        status = "⚠️ DRIFT" if drift["dataset_drift"] else "✅ Stable"
        print(f"  {batch['label']}: {status} "
              f"({drift['n_drifted_features']}/{drift['n_features']} features drifted)")

    # Save drift summary
    drift_df = pd.DataFrame([{k: v for k, v in d.items() if k != "html_report"}
                              for d in drift_results])
    drift_df.to_csv(os.path.join(MONITORING_DIR, "drift_summary.csv"), index=False)

    return {
        "batch_stats":   stats_df,
        "drift_results": drift_results,
        "batches":       batches,
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

    model = joblib.load(os.path.join(OUTPUTS_DIR, "improvement", "best_model_improved.joblib"))
    risk_scores = model.predict_proba(X_test_t)[:, 1]

    raw_test = df_raw.iloc[y_test.index].reset_index(drop=True)
    results  = run_monitoring(raw_test, risk_scores, y_test.values)

    plot_monitoring_charts(results["batch_stats"])

    print(f"\n[OK] Monitoring complete. Outputs in: {MONITORING_DIR}")
