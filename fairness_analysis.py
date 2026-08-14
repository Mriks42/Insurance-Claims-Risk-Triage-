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
- Disparate impact ratio (least-flagged group's rate / this group's rate;
  see group_metrics() for why the comparison runs in that direction)

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
AGE_UNKNOWN_LABEL = "Unknown (age not recorded)"


def bin_age(age_series):
    """
    Bin continuous age into groups.

    320 rows in fraud_oracle.csv carry Age == 0, which is not a real age — the
    minimum genuine value is 16. Those rows fall outside the (0, 25] bin, so
    pd.cut returns NaN and the dashboard used to render a demographic group
    literally labelled "nan" (n=34 on the test split). They are now labelled
    explicitly and excluded from the disparate impact calculation, the same way
    small groups are.

    Note this affects reporting only. Age is left untouched in the model's
    feature path — changing it would alter the trained feature space.
    """
    bins   = [0, 25, 35, 50, 65, 120]
    labels = ["16-25", "26-35", "36-50", "51-65", "65+"]
    binned = pd.cut(age_series, bins=bins, labels=labels, right=True)
    return binned.cat.add_categories([AGE_UNKNOWN_LABEL]).fillna(AGE_UNKNOWN_LABEL)


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

    # Disparate impact ratio.
    #
    #   DI(group) = (lowest flag rate across groups) / (this group's flag rate)
    #
    # So the LEAST-flagged group scores 1.0, and a group scores below 0.80 when
    # it is flagged at more than 1.25x the least-flagged group's rate. Being
    # flagged here is a burden (investigation), not a benefit, so the group the
    # rule surfaces is the most-flagged one — which is the opposite of the
    # classic hiring-selection framing of the 80% rule.
    if len(result) > 0:
        # The reference must be the least-flagged group WITH A NON-ZERO flag
        # rate. On the test split the 65+ group (n=48) is never flagged at all,
        # so a plain .min() made the reference 0.0 and every other group scored
        # 0/x = 0.000 — the entire column read "Concern" and carried no
        # information. A group that is never flagged cannot serve as a ratio
        # denominator; it is reported separately instead.
        nonzero = result.loc[result["flag_rate"] > 0, "flag_rate"]
        min_flag = nonzero.min() if len(nonzero) else 0.0

        def _di(x):
            if x <= 0:
                return None          # never flagged — ratio undefined
            if min_flag <= 0:
                return None          # no valid reference group
            return round(min_flag / x, 4)

        result["disparate_impact"] = result["flag_rate"].apply(_di)
        # pd.isna, not `is None`: assigning None into a float column stores NaN,
        # so an identity check silently falls through to the "OK" branch and
        # labels a never-flagged group as passing.
        result["di_flag"] = result["disparate_impact"].apply(
            lambda x: "n/a (never flagged)" if pd.isna(x)
            else ("⚠️ Concern" if x < DISPARATE_IMPACT_THRESHOLD else "✅ OK")
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

        # Groups with no defined ratio (never flagged) are dropped rather than
        # drawn as an empty bar labelled "nan"; they are named in the subtitle.
        undefined = df.loc[df["disparate_impact"].isna(), "group"].tolist()
        df = df[df["disparate_impact"].notna()]
        if df.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e74c3c" if di < DISPARATE_IMPACT_THRESHOLD else "#27ae60"
                  for di in df["disparate_impact"]]
        bars = ax.bar(df["group"], df["disparate_impact"], color=colors, alpha=0.85)
        ax.axhline(y=DISPARATE_IMPACT_THRESHOLD, color="black", linestyle="--",
                   linewidth=1.5, label=f"80% rule threshold ({DISPARATE_IMPACT_THRESHOLD})")
        subtitle = f"(Below {DISPARATE_IMPACT_THRESHOLD} = potential bias concern)"
        if undefined:
            subtitle += f"\nNot shown — never flagged: {', '.join(undefined)}"
        ax.set_title(f"Disparate Impact Ratio — {attr}\n{subtitle}", fontsize=12)
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

    from data_pipeline import build_model_dataset, load_best_model, raw_rows_for

    data   = build_model_dataset()
    model  = load_best_model()
    y_test = data["y_test"]

    risk_scores = model.predict_proba(data["X_test_t"])[:, 1]
    raw_test    = raw_rows_for(data, "test")

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
