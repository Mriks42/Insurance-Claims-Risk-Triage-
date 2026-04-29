"""
Fairness Analysis Module
=========================
Checks whether the fraud triage model treats different demographic
groups equitably. Insurance ML is regulated — models cannot
discriminate based on protected attributes.

Analyzes:
- Age groups
- Sex
- Marital Status

Metrics per group:
- Flag rate (% of group flagged as SIU or Manual Review)
- Fraud rate (actual fraud % in group)
- False positive rate (flagged but not fraud)
- Precision (flagged and actually fraud)
- Disparate impact ratio (vs. majority group)

Usage:
    python fairness_analysis.py
    from fairness_analysis import compute_fairness_report
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUTS_DIR, PCT_SIU, PCT_MANUAL

FAIRNESS_DIR = os.path.join(OUTPUTS_DIR, "fairness")
os.makedirs(FAIRNESS_DIR, exist_ok=True)

# Disparate impact threshold (80% rule — standard in US employment/insurance law)
DISPARATE_IMPACT_THRESHOLD = 0.80


# ============================================================
# Age group binning
# ============================================================
def bin_age(age_series):
    """Bin continuous age into groups."""
    bins   = [0, 25, 35, 50, 65, 120]
    labels = ["16-25", "26-35", "36-50", "51-65", "65+"]
    return pd.cut(age_series, bins=bins, labels=labels, right=True)


# ============================================================
# Core fairness metrics per group
# ============================================================
def group_metrics(df, group_col, score_col="risk_score",
                  label_col="actual_fraud", flagged_col="flagged"):
    """
    Compute fairness metrics for each value of group_col.

    Returns a DataFrame with one row per group.
    """
    rows = []
    groups = df[group_col].dropna().unique()

    for g in sorted(groups, key=str):
        mask = df[group_col] == g
        sub  = df[mask]

        n          = len(sub)
        n_fraud    = int(sub[label_col].sum())
        n_flagged  = int(sub[flagged_col].sum())
        n_fp       = int(((sub[flagged_col] == 1) & (sub[label_col] == 0)).sum())
        n_tp       = int(((sub[flagged_col] == 1) & (sub[label_col] == 1)).sum())

        fraud_rate  = n_fraud / n if n > 0 else 0
        flag_rate   = n_flagged / n if n > 0 else 0
        fpr         = n_fp / max(1, (sub[label_col] == 0).sum())
        precision   = n_tp / max(1, n_flagged)
        avg_score   = float(sub[score_col].mean())

        rows.append({
            "group":        str(g),
            "n":            n,
            "n_fraud":      n_fraud,
            "n_flagged":    n_flagged,
            "fraud_rate":   round(fraud_rate, 4),
            "flag_rate":    round(flag_rate, 4),
            "false_pos_rate": round(fpr, 4),
            "precision":    round(precision, 4),
            "avg_risk_score": round(avg_score, 4),
        })

    result = pd.DataFrame(rows)

    # Disparate impact ratio (relative to group with lowest flag rate)
    if len(result) > 0:
        min_flag = result["flag_rate"].min()
        result["disparate_impact"] = result["flag_rate"].apply(
            lambda x: round(min_flag / x, 4) if x > 0 else 1.0
        )
        result["di_flag"] = result["disparate_impact"].apply(
            lambda x: "⚠️ Concern" if x < DISPARATE_IMPACT_THRESHOLD else "✅ OK"
        )

    return result


# ============================================================
# Full fairness report
# ============================================================
def compute_fairness_report(df_raw, risk_scores, y_true):
    """
    Compute fairness metrics across all protected attributes.

    Args:
        df_raw:      Raw dataframe (with Age, Sex, MaritalStatus columns)
        risk_scores: Array of model risk scores (same order as df_raw rows used)
        y_true:      Array of actual fraud labels

    Returns:
        dict of {attribute: metrics_dataframe}
    """
    # Build analysis dataframe
    analysis = pd.DataFrame({
        "risk_score":    risk_scores,
        "actual_fraud":  np.array(y_true),
    })

    # Assign triage buckets
    siu_thresh    = np.percentile(risk_scores, 100 * (1 - PCT_SIU))
    manual_thresh = np.percentile(risk_scores, 100 * (1 - PCT_SIU - PCT_MANUAL))
    analysis["bucket"] = np.where(
        risk_scores >= siu_thresh, "SIU",
        np.where(risk_scores >= manual_thresh, "Manual Review", "Approve")
    )
    analysis["flagged"] = (analysis["bucket"].isin(["SIU", "Manual Review"])).astype(int)

    # Add demographic columns from raw data
    raw_reset = df_raw.reset_index(drop=True)
    if "Age" in raw_reset.columns:
        analysis["Age"]          = raw_reset["Age"].values
        analysis["AgeGroup"]     = bin_age(raw_reset["Age"]).astype(str)
    if "Sex" in raw_reset.columns:
        analysis["Sex"]          = raw_reset["Sex"].values
    if "MaritalStatus" in raw_reset.columns:
        analysis["MaritalStatus"] = raw_reset["MaritalStatus"].values

    reports = {}

    # Age group analysis
    if "AgeGroup" in analysis.columns:
        reports["Age Group"] = group_metrics(analysis, "AgeGroup")

    # Sex analysis
    if "Sex" in analysis.columns:
        reports["Sex"] = group_metrics(analysis, "Sex")

    # Marital status analysis
    if "MaritalStatus" in analysis.columns:
        reports["Marital Status"] = group_metrics(analysis, "MaritalStatus")

    return reports, analysis


# ============================================================
# Visualizations
# ============================================================
def plot_fairness_charts(reports, save_dir=FAIRNESS_DIR):
    """Generate and save fairness bar charts for each attribute."""
    for attr, df in reports.items():
        if df.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Fairness Analysis — {attr}", fontsize=14, fontweight="bold")

        metrics = [
            ("flag_rate",     "Flag Rate (SIU + Manual)", "#e74c3c"),
            ("fraud_rate",    "Actual Fraud Rate",        "#2563eb"),
            ("false_pos_rate","False Positive Rate",      "#f39c12"),
        ]

        for ax, (metric, title, color) in zip(axes, metrics):
            bars = ax.bar(df["group"], df[metric], color=color, alpha=0.8)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(attr)
            ax.set_ylabel("Rate")
            ax.set_ylim(0, min(1.0, df[metric].max() * 1.4 + 0.05))
            for bar, val in zip(bars, df[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        fname = os.path.join(save_dir,
                             f"fairness_{attr.lower().replace(' ', '_')}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


def plot_disparate_impact(reports, save_dir=FAIRNESS_DIR):
    """Plot disparate impact ratios with the 80% threshold line."""
    for attr, df in reports.items():
        if df.empty or "disparate_impact" not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e74c3c" if di < DISPARATE_IMPACT_THRESHOLD else "#27ae60"
                  for di in df["disparate_impact"]]
        bars = ax.bar(df["group"], df["disparate_impact"], color=colors, alpha=0.85)
        ax.axhline(y=DISPARATE_IMPACT_THRESHOLD, color="black", linestyle="--",
                   linewidth=1.5, label=f"80% rule threshold ({DISPARATE_IMPACT_THRESHOLD})")
        ax.set_title(f"Disparate Impact Ratio — {attr}\n"
                     f"(Below {DISPARATE_IMPACT_THRESHOLD} = potential bias concern)",
                     fontsize=12)
        ax.set_xlabel(attr)
        ax.set_ylabel("Disparate Impact Ratio")
        ax.set_ylim(0, 1.2)
        ax.legend()

        for bar, val in zip(bars, df["disparate_impact"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        fname = os.path.join(save_dir,
                             f"disparate_impact_{attr.lower().replace(' ', '_')}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


# ============================================================
# Summary
# ============================================================
def print_fairness_summary(reports):
    """Print a human-readable fairness summary."""
    print("\n" + "=" * 60)
    print("FAIRNESS ANALYSIS SUMMARY")
    print("=" * 60)

    any_concern = False
    for attr, df in reports.items():
        print(f"\n── {attr} ──")
        cols = ["group", "n", "fraud_rate", "flag_rate",
                "false_pos_rate", "precision", "disparate_impact", "di_flag"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))

        if "di_flag" in df.columns:
            concerns = df[df["di_flag"].str.contains("Concern")]
            if not concerns.empty:
                any_concern = True
                print(f"\n  ⚠️  Disparate impact concern for: "
                      f"{', '.join(concerns['group'].tolist())}")

    if any_concern:
        print("\n[WARNING] Some groups show potential disparate impact (below 80% rule).")
        print("  Consider reviewing model features for proxy discrimination.")
    else:
        print("\n[OK] No disparate impact concerns detected.")


# ============================================================
# Standalone run
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Fairness Analysis")
    print("=" * 60)

    import joblib
    from config import DATA_PATH, TARGET, COLS_TO_DROP, RANDOM_STATE
    from feature_engineering import engineer_features, REPLACED_CATEGORICALS
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Load data
    df_raw = pd.read_csv(DATA_PATH)
    df_eng = engineer_features(df_raw)

    X = df_eng.drop(columns=[TARGET] + COLS_TO_DROP)
    y = df_eng[TARGET]

    cols_to_remove = [c for c in REPLACED_CATEGORICALS if c in X.columns]
    X_xgb = X.drop(columns=cols_to_remove)

    _, X_temp, _, y_temp = train_test_split(
        X_xgb, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Rebuild preprocessor
    X_train, _, _, _ = train_test_split(
        X_xgb, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
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

    # Load model
    model_path = os.path.join(OUTPUTS_DIR, "improvement", "best_model_improved.joblib")
    model = joblib.load(model_path)
    risk_scores = model.predict_proba(X_test_t)[:, 1]

    # Get raw test rows
    raw_test = df_raw.iloc[y_test.index].reset_index(drop=True)

    # Compute fairness
    reports, analysis = compute_fairness_report(raw_test, risk_scores, y_test.values)

    # Save reports
    for attr, df in reports.items():
        fname = os.path.join(FAIRNESS_DIR,
                             f"fairness_{attr.lower().replace(' ', '_')}.csv")
        df.to_csv(fname, index=False)
        print(f"Saved: {fname}")

    # Print summary
    print_fairness_summary(reports)

    # Save charts
    print("\nGenerating charts...")
    plot_fairness_charts(reports)
    plot_disparate_impact(reports)

    print(f"\n[OK] Fairness analysis complete. Outputs in: {FAIRNESS_DIR}")
