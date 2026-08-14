"""
SHAP Explainability Module
===========================
Handles: computing SHAP values, global feature importance, per-claim
explanations, and human-readable reason codes.

Phase 4 (Weeks 7-8) of the FSE 570 Capstone project.

Usage:
    python shap_explainability.py                 # standalone (runs full pipeline first)
    from shap_explainability import ...           # import into Streamlit app
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    N_REASON_CODES, PCT_SIU, PCT_MANUAL,
    SHAP_DIR, METRICS_DIR,
)


# ============================================================
# SHAP computation
# ============================================================
def compute_shap_values(model, X_data):
    """
    Compute SHAP values using TreeExplainer.
    Returns (explainer, shap_values).
    """
    try:
        import shap
    except ImportError:
        os.system(f"{sys.executable} -m pip install shap --quiet")
        import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)

    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Base value (expected value): {explainer.expected_value:.4f}")

    return explainer, shap_values


# ============================================================
# Global feature importance
# ============================================================
def global_feature_importance(shap_values, feature_names, save_dir=SHAP_DIR, top_n=20):
    """Compute and save global feature importance (mean |SHAP|)."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance.to_csv(os.path.join(save_dir, "global_feature_importance.csv"), index=False)

    print(f"\n=== TOP {top_n} GLOBAL FEATURE IMPORTANCES ===")
    print(importance.head(top_n).to_string(index=False))

    # Bar chart
    top_features = importance.head(top_n)
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_features["mean_abs_shap"].values[::-1])
    plt.yticks(range(top_n), top_features["feature"].values[::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {top_n} Global Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "global_feature_importance.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(save_dir, 'global_feature_importance.png')}")

    return importance


# ============================================================
# SHAP summary (beeswarm) plot
# ============================================================
def shap_summary_plot(shap_values, X_data, feature_names, save_dir=SHAP_DIR):
    """Generate and save the SHAP beeswarm summary plot."""
    import shap

    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        features=X_data,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_summary_beeswarm.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(save_dir, 'shap_summary_beeswarm.png')}")


# ============================================================
# Per-claim explanations
# ============================================================
def explain_top_claims(shap_values, y_prob, y_true, feature_names,
                       n_claims=None, n_drivers=N_REASON_CODES, save_dir=SHAP_DIR,
                       X_data=None):
    """
    Generate per-claim SHAP explanations for the top high-risk claims.
    Returns a list of explanation dicts.
    """
    if n_claims is None:
        n_claims = max(1, int(PCT_SIU * len(y_prob)))

    sorted_indices = np.argsort(y_prob)[::-1]
    explanations = []

    for rank, idx in enumerate(sorted_indices[:n_claims]):
        claim_shap = shap_values[idx]
        top_fi = np.argsort(np.abs(claim_shap))[::-1][:n_drivers]

        if X_data is not None:
            row = X_data[idx]
            row = row.toarray().ravel() if hasattr(row, "toarray") else np.asarray(row).ravel()
        else:
            row = None

        drivers = []
        for fi in top_fi:
            drivers.append({
                "feature": feature_names[fi],
                "shap_value": round(float(claim_shap[fi]), 4),
                "direction": "increases" if claim_shap[fi] > 0 else "decreases",
                # value of this feature for THIS claim — lets the reason code say
                # "is NOT 'Liability'" when the one-hot dummy is 0
                "feature_value": (None if row is None else float(row[fi])),
            })

        explanations.append({
            "rank": rank + 1,
            "val_index": int(idx),
            "risk_score": round(float(y_prob[idx]), 4),
            "actual_fraud": int(y_true[idx]),
            "drivers": drivers,
        })

    # Save detailed CSV
    rows = []
    for claim in explanations:
        for d in claim["drivers"]:
            rows.append({
                "rank": claim["rank"],
                "val_index": claim["val_index"],
                "risk_score": claim["risk_score"],
                "actual_fraud": claim["actual_fraud"],
                "feature": d["feature"],
                "shap_value": d["shap_value"],
                "direction": d["direction"],
            })

    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "siu_claim_explanations.csv"), index=False)
    print(f"\nGenerated explanations for {n_claims} SIU-tier claims")
    print(f"Saved: {os.path.join(save_dir, 'siu_claim_explanations.csv')}")

    # Print example
    if explanations:
        ex = explanations[0]
        print(f"\n=== EXAMPLE CLAIM EXPLANATION (Rank 1) ===")
        print(f"Risk Score: {ex['risk_score']}")
        print(f"Actual Fraud: {'Yes' if ex['actual_fraud'] else 'No'}")
        print(f"Top {n_drivers} Drivers:")
        for d in ex["drivers"]:
            print(f"  - {d['feature']}: SHAP={d['shap_value']:+.4f} ({d['direction']} fraud risk)")

    return explanations


# ============================================================
# Human-readable reason codes
# ============================================================
def make_reason_code(feature_name, shap_value, cat_cols=None, feature_value=None):
    """
    Convert a SHAP driver into a human-readable reason code.

    Correctly handles OHE features whose original column names contain
    underscores (e.g. AddressChange_Claim_2 to 3 years) by matching
    the longest column-name prefix first.

    feature_value is the claim's value for that one-hot column. A dummy can
    carry a large SHAP value while being 0 — meaning "this claim is NOT that
    category, and that matters". Without it the code asserts the claim IS the
    category, which contradicts the claim's own attributes. When it is None the
    older, ambiguous phrasing is kept so existing callers do not change meaning.
    """
    direction = "High risk" if shap_value > 0 else "Low risk"
    strength  = abs(shap_value)

    if strength > 0.5:
        intensity = "STRONG"
    elif strength > 0.2:
        intensity = "MODERATE"
    else:
        intensity = "MILD"

    # Match longest prefix first to handle columns with underscores
    from feature_engineering import humanize_feature

    # Plain English first, raw feature name in brackets. Investigators read the
    # sentence; the bracketed name keeps the row auditable back to the model.
    human = humanize_feature(feature_name, cat_cols, feature_value)
    return (f"[{intensity}] {direction}: {human} "
            f"[{feature_name}] (SHAP: {shap_value:+.4f})")


def generate_reason_codes(explanations, cat_cols=None, save_dir=SHAP_DIR):
    """Generate human-readable reason codes for all claim explanations."""
    rows = []
    for claim in explanations:
        for d in claim["drivers"]:
            rc = make_reason_code(d["feature"], d["shap_value"], cat_cols,
                                  d.get("feature_value"))
            rows.append({
                "rank": claim["rank"],
                "risk_score": claim["risk_score"],
                "actual_fraud": claim["actual_fraud"],
                "reason_code": rc,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "siu_reason_codes.csv"), index=False)
    print(f"\nSaved: {os.path.join(save_dir, 'siu_reason_codes.csv')}")

    # Print examples
    print(f"\n=== EXAMPLE REASON CODES (Top 3 claims) ===")
    for claim in explanations[:3]:
        fraud_str = "Yes" if claim["actual_fraud"] else "No"
        print(f"\nClaim Rank {claim['rank']} | Risk Score: {claim['risk_score']} | Fraud: {fraud_str}")
        for d in claim["drivers"]:
            rc = make_reason_code(d["feature"], d["shap_value"], cat_cols,
                                  d.get("feature_value"))
            print(f"  {rc}")

    return df


# ============================================================
# Main: run full SHAP analysis
# ============================================================
def run_shap_analysis(data=None):
    """
    Run the full SHAP analysis pipeline.
    If data is None, runs the modeling pipeline first to get data + model.
    Uses calibrated risk scores for triage ranking when available.
    """
    print("=" * 60)
    print("SHAP Explainability Analysis")
    print("=" * 60)

    if data is None:
        from modeling import run_full_pipeline
        data = run_full_pipeline()

    model        = data["best_model"]
    X_val_t      = data["X_val_t"]
    y_val        = np.array(data["y_val"])
    feature_names = data["feature_names"]
    cat_cols     = data["cat_cols"]

    # Use calibrated probabilities for triage ranking if available,
    # raw probabilities for SHAP (TreeExplainer needs the raw model)
    if "cal_val_prob" in data:
        y_prob = data["cal_val_prob"]
        print("  Using calibrated risk scores for triage ranking.")
    else:
        y_prob = model.predict_proba(X_val_t)[:, 1]

    # Compute SHAP on raw model (TreeExplainer requires the base estimator)
    print("\nComputing SHAP values...")
    explainer, shap_values = compute_shap_values(model, X_val_t)

    # Global importance
    print("\nComputing global feature importance...")
    importance = global_feature_importance(shap_values, feature_names)

    # Summary plot
    print("\nGenerating SHAP summary plot...")
    shap_summary_plot(shap_values, X_val_t, feature_names)

    # Per-claim explanations
    print("\nGenerating per-claim explanations...")
    explanations = explain_top_claims(shap_values, y_prob, y_val, feature_names,
                                      X_data=X_val_t)

    # Reason codes
    print("\nGenerating human-readable reason codes...")
    reason_codes = generate_reason_codes(explanations, cat_cols)

    print("\n" + "=" * 60)
    print("[OK] SHAP Analysis Complete!")
    print("=" * 60)
    print(f"\nAll outputs saved to: {SHAP_DIR}")

    return {
        "explainer":    explainer,
        "shap_values":  shap_values,
        "importance":   importance,
        "explanations": explanations,
        "reason_codes": reason_codes,
    }


if __name__ == "__main__":
    run_shap_analysis()
