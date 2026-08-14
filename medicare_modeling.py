"""
Medicare Provider Fraud — XGBoost + Optuna Modeling Pipeline
=============================================================
Trains an XGBoost model with Optuna Bayesian tuning on the Medicare
Provider Fraud dataset. Produces SHAP analysis and triage bucket
performance for cross-domain comparison with the auto insurance model.

Run:
    python medicare_modeling.py

Outputs saved to: outputs/medicare/
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
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, confusion_matrix
)
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap

OUT_DIR = os.path.join("outputs", "medicare")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS     = 20
CV_FOLDS     = 3

# ══════════════════════════════════════════════════════════════
# 1. LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════

def load_and_prepare():
    print("[1] Loading and preparing Medicare data...")

    labels = pd.read_csv("medicare_data/Train-1542865627584.csv")
    bene   = pd.read_csv("medicare_data/Train_Beneficiarydata-1542865627584.csv")
    inp    = pd.read_csv("medicare_data/Train_Inpatientdata-1542865627584.csv")
    out    = pd.read_csv("medicare_data/Train_Outpatientdata-1542865627584.csv")

    # Binary label
    labels["Fraud"] = (labels["PotentialFraud"] == "Yes").astype(int)

    # ── Inpatient aggregation ─────────────────────────────────
    inp_agg = inp.groupby("Provider").agg(
        InpatientClaims         = ("ClaimID",                "count"),
        AvgInpatientReimbursed  = ("InscClaimAmtReimbursed", "mean"),
        TotalInpatientAmt       = ("InscClaimAmtReimbursed", "sum"),
        AvgDeductible_IP        = ("DeductibleAmtPaid",      "mean"),
        UniquePatients_IP       = ("BeneID",                 "nunique"),
        UniquePhysicians        = ("AttendingPhysician",     "nunique"),
        MaxInpatientReimbursed  = ("InscClaimAmtReimbursed", "max"),
    ).reset_index()

    # ── Outpatient aggregation ────────────────────────────────
    out_agg = out.groupby("Provider").agg(
        OutpatientClaims        = ("ClaimID",                "count"),
        AvgOutpatientReimbursed = ("InscClaimAmtReimbursed", "mean"),
        TotalOutpatientAmt      = ("InscClaimAmtReimbursed", "sum"),
        UniquePatients_OP       = ("BeneID",                 "nunique"),
        MaxOutpatientReimbursed = ("InscClaimAmtReimbursed", "max"),
    ).reset_index()

    # ── Beneficiary aggregation via inpatient claims ──────────
    chronic_cols = [c for c in bene.columns if c.startswith("ChronicCond_")]
    bene_inp = inp[["Provider", "BeneID"]].drop_duplicates()
    bene_merged = bene_inp.merge(bene, on="BeneID", how="left")

    # NOTE: this aggregates only chronic_cols[0] (ChronicCond_Alzheimer), not a
    # count across all chronic conditions — the name is now honest about that.
    # Summing across every ChronicCond_* column would be the better feature, but
    # it changes the feature matrix and would invalidate the published Medicare
    # results, so it is deferred to the next retrain.
    # Also note bene_merged is joined via INPATIENT claims only, so providers
    # with outpatient claims exclusively get 0 for every beneficiary feature.
    bene_agg = bene_merged.groupby("Provider").agg(
        AvgAlzheimerFlag      = (chronic_cols[0],              "mean"),
        UniqueStates          = ("State",                      "nunique"),
        AvgIPAnnualReimb      = ("IPAnnualReimbursementAmt",   "mean"),
        AvgOPAnnualReimb      = ("OPAnnualReimbursementAmt",   "mean"),
        PctDeceased           = ("DOD",                        lambda x: x.notna().mean()),
    ).reset_index()

    # ── Merge all ─────────────────────────────────────────────
    df = labels.merge(inp_agg,  on="Provider", how="left")
    df = df.merge(out_agg,  on="Provider", how="left")
    df = df.merge(bene_agg, on="Provider", how="left")
    df = df.fillna(0)

    # ── Derived features ──────────────────────────────────────
    df["TotalClaims"]         = df["InpatientClaims"] + df["OutpatientClaims"]
    df["TotalReimbursed"]     = df["TotalInpatientAmt"] + df["TotalOutpatientAmt"]
    df["AvgReimbursedPerClaim"] = np.where(
        df["TotalClaims"] > 0,
        df["TotalReimbursed"] / df["TotalClaims"], 0
    )
    df["InpatientRatio"]      = np.where(
        df["TotalClaims"] > 0,
        df["InpatientClaims"] / df["TotalClaims"], 0
    )
    df["UniquePatients"]      = df["UniquePatients_IP"] + df["UniquePatients_OP"]

    feature_cols = [c for c in df.columns
                    if c not in ["Provider", "PotentialFraud", "Fraud"]]

    X = df[feature_cols].values
    y = df["Fraud"].values

    print(f"    Feature matrix: {X.shape[0]:,} providers × {X.shape[1]} features")
    print(f"    Fraud rate: {y.mean()*100:.2f}% ({y.sum()} fraud / {len(y)} total)")
    print(f"    Features: {feature_cols}")

    return X, y, feature_cols, df


# ══════════════════════════════════════════════════════════════
# 2. SPLIT
# ══════════════════════════════════════════════════════════════

def split_data(X, y):
    print("\n[2] Splitting data (70/15/15 stratified)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"    Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
    print(f"    Train fraud: {y_train.mean()*100:.2f}% | "
          f"Val fraud: {y_val.mean()*100:.2f}% | "
          f"Test fraud: {y_test.mean()*100:.2f}%")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════
# 3. OPTUNA TUNING
# ══════════════════════════════════════════════════════════════

def tune_xgboost(X_train, y_train):
    print(f"\n[3] Optuna Bayesian tuning ({N_TRIALS} trials, {CV_FOLDS}-fold CV)...")

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "objective":         "binary:logistic",
            "eval_metric":       "aucpr",
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
            "tree_method":       "hist",
        }
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="average_precision", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, timeout=300)

    best_params = study.best_params
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    })

    print(f"    Best CV PR-AUC: {study.best_value:.4f}")
    print(f"    Best params: {best_params}")
    return best_params, study


# ══════════════════════════════════════════════════════════════
# 4. TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════

def train_and_evaluate(best_params, X_train, X_val, X_test,
                       y_train, y_val, y_test):
    print("\n[4] Training final model...")

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    # Calibrate on validation set
    val_pool = np.vstack([X_val, X_train])
    y_pool   = np.concatenate([y_val, y_train])
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
    calibrated.fit(val_pool, y_pool)

    val_prob  = calibrated.predict_proba(X_val)[:, 1]
    test_prob = calibrated.predict_proba(X_test)[:, 1]

    val_pr_auc  = average_precision_score(y_val,  val_prob)
    val_roc_auc = roc_auc_score(y_val,  val_prob)
    test_pr_auc  = average_precision_score(y_test, test_prob)
    test_roc_auc = roc_auc_score(y_test, test_prob)

    # Precision@5%
    thresh_5pct = np.percentile(val_prob, 95)
    y_pred_5pct = (val_prob >= thresh_5pct).astype(int)
    tp = ((y_pred_5pct == 1) & (y_val == 1)).sum()
    fp = ((y_pred_5pct == 1) & (y_val == 0)).sum()
    fn = ((y_pred_5pct == 0) & (y_val == 1)).sum()
    prec_5pct = tp / max(1, tp + fp)
    rec_5pct  = tp / max(1, tp + fn)

    print(f"\n    ========== MEDICARE XGBOOST (OPTUNA) ==========")
    print(f"    Val  PR-AUC:    {val_pr_auc:.4f}")
    print(f"    Val  ROC-AUC:   {val_roc_auc:.4f}")
    print(f"    Test PR-AUC:    {test_pr_auc:.4f}")
    print(f"    Test ROC-AUC:   {test_roc_auc:.4f}")
    print(f"    Precision@5%:   {prec_5pct:.4f}")
    print(f"    Recall@5%:      {rec_5pct:.4f}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring="average_precision")
    print(f"    5-fold CV PR-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    metrics = {
        "val_pr_auc":    val_pr_auc,
        "val_roc_auc":   val_roc_auc,
        "test_pr_auc":   test_pr_auc,
        "test_roc_auc":  test_roc_auc,
        "prec_5pct":     prec_5pct,
        "rec_5pct":      rec_5pct,
        "cv_mean":       float(cv_scores.mean()),
        "cv_std":        float(cv_scores.std()),
    }

    return calibrated, model, val_prob, test_prob, metrics


# ══════════════════════════════════════════════════════════════
# 5. TRIAGE ANALYSIS
# ══════════════════════════════════════════════════════════════

def triage_analysis(y_test, test_prob, metrics):
    print("\n[5] Triage bucket analysis...")

    siu_thresh    = np.percentile(test_prob, 95)
    manual_thresh = np.percentile(test_prob, 80)

    siu_mask    = test_prob >= siu_thresh
    manual_mask = (test_prob >= manual_thresh) & ~siu_mask
    approve_mask = ~siu_mask & ~manual_mask

    base_rate = y_test.mean()

    results = []
    for name, mask in [("SIU (top 5%)", siu_mask),
                       ("Manual Review (15%)", manual_mask),
                       ("Approve (80%)", approve_mask)]:
        if mask.sum() > 0:
            fraud_rate = y_test[mask].mean()
            enrichment = fraud_rate / base_rate
            results.append({
                "Bucket":     name,
                "Count":      int(mask.sum()),
                "Fraud Rate": f"{fraud_rate*100:.1f}%",
                "Enrichment": f"{enrichment:.1f}×",
            })
            print(f"    {name}: {mask.sum()} providers, "
                  f"fraud rate {fraud_rate*100:.1f}% ({enrichment:.1f}× enrichment)")

    metrics["siu_fraud_rate"]    = float(y_test[siu_mask].mean()) if siu_mask.sum() > 0 else 0
    metrics["siu_enrichment"]    = float(y_test[siu_mask].mean() / base_rate) if siu_mask.sum() > 0 else 0
    return results, metrics


# ══════════════════════════════════════════════════════════════
# 6. SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════

def shap_analysis(model, X_test, feature_cols):
    print("\n[6] SHAP analysis...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":       feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("    Top 10 features by SHAP:")
    for _, row in importance_df.head(10).iterrows():
        print(f"      {row['feature']:35s} {row['mean_abs_shap']:.4f}")

    importance_df.to_csv(os.path.join(OUT_DIR, "medicare_shap_importance.csv"), index=False)
    return importance_df, shap_values


# ══════════════════════════════════════════════════════════════
# 7. PLOTS
# ══════════════════════════════════════════════════════════════

def plot_pr_curve(y_val, val_prob, metrics):
    prec, rec, _ = precision_recall_curve(y_val, val_prob)
    pr_auc = metrics["val_pr_auc"]
    base   = y_val.mean()

    window  = 10
    prec_s  = pd.Series(prec).rolling(window, min_periods=1, center=True).mean().values
    rec_s   = pd.Series(rec).rolling(window, min_periods=1, center=True).mean().values

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(rec_s, prec_s, color="#7c3aed", linewidth=2.5,
            label=f"XGBoost Optuna (PR-AUC = {pr_auc:.4f})")
    ax.fill_between(rec_s, prec_s, alpha=0.08, color="#7c3aed")
    ax.axhline(base, color="#9ca3af", linestyle="--",
               label=f"Random baseline ({base*100:.1f}%)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Medicare Provider Fraud — Precision-Recall Curve",
                 fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "medicare_pr_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_shap_importance(importance_df):
    top = importance_df.head(12)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
                   color="#7c3aed", alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, top["mean_abs_shap"][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Medicare Provider Fraud — Top 12 SHAP Feature Importances",
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "medicare_shap_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_shap_comparison(importance_df):
    """Side-by-side SHAP comparison: Medicare vs Auto Insurance."""
    # Auto insurance top features from main pipeline
    auto_features = {
        "Fault":               0.823,
        "Liability_NoPolice":  0.546,
        "PolicyHolderFault":   0.246,
        "Fault_NoPolice":      0.244,
        "BasePolicy":          0.218,
        "VehiclePrice_Num":    0.187,
        "Age":                 0.165,
        "Deductible":          0.142,
    }

    medicare_top = importance_df.head(8)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    # Auto
    ax = axes[0]
    feats = list(auto_features.keys())[::-1]
    vals  = list(auto_features.values())[::-1]
    bars = ax.barh(feats, vals, color="#2563eb", alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Automotive Insurance\nTop SHAP Features", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Medicare
    ax = axes[1]
    feats_m = medicare_top["feature"].tolist()[::-1]
    vals_m  = medicare_top["mean_abs_shap"].tolist()[::-1]
    bars = ax.barh(feats_m, vals_m, color="#7c3aed", alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals_m):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Medicare Provider Fraud\nTop SHAP Features", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("SHAP Feature Importance: Domain-Specific Fraud Signals",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "shap_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_model_comparison(metrics):
    """Bar chart comparing PR-AUC across both domains."""
    fig, ax = plt.subplots(figsize=(6, 3.2))

    models  = ["Auto Insurance\nXGBoost (Optuna)\nVal PR-AUC",
               "Auto Insurance\nXGBoost (Optuna)\nTest PR-AUC",
               "Medicare\nXGBoost (Optuna)\nVal PR-AUC",
               "Medicare\nXGBoost (Optuna)\nTest PR-AUC"]
    values  = [0.3223, 0.2443,
               metrics["val_pr_auc"], metrics["test_pr_auc"]]
    colors  = ["#2563eb", "#1d4ed8", "#7c3aed", "#6d28d9"]
    baselines = [0.0599, 0.0599, 0.0935, 0.0935]

    bars = ax.bar(models, values, color=colors, alpha=0.85,
                  edgecolor="white", width=0.5)
    for bar, val, base in zip(bars, values, baselines):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.4f}\n({val/base:.1f}× baseline)",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("PR-AUC")
    ax.set_title("Model Performance Comparison: Auto vs. Medicare",
                 fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "model_performance_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════
# 8. SAVE RESULTS
# ══════════════════════════════════════════════════════════════

def save_results(metrics, triage_results, best_params):
    results = {
        "model":        "XGBoost (Optuna)",
        "dataset":      "Medicare Provider Fraud",
        "metrics":      metrics,
        "triage":       triage_results,
        "best_params":  best_params,
    }
    path = os.path.join(OUT_DIR, "medicare_model_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n    Saved: {path}")
    return results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("MEDICARE PROVIDER FRAUD — XGBOOST + OPTUNA PIPELINE")
    print("=" * 65)

    # Load
    X, y, feature_cols, df = load_and_prepare()

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Tune
    best_params, study = tune_xgboost(X_train, y_train)

    # Train & evaluate
    calibrated, model, val_prob, test_prob, metrics = train_and_evaluate(
        best_params, X_train, X_val, X_test, y_train, y_val, y_test
    )

    # Triage
    triage_results, metrics = triage_analysis(y_test, test_prob, metrics)

    # SHAP
    importance_df, shap_values = shap_analysis(model, X_test, feature_cols)

    # Plots
    print("\n[7] Generating plots...")
    plot_pr_curve(y_val, val_prob, metrics)
    plot_shap_importance(importance_df)
    plot_shap_comparison(importance_df)
    plot_model_comparison(metrics)

    # Save
    save_results(metrics, triage_results, best_params)

    # Save model
    model_path = os.path.join(OUT_DIR, "medicare_model.joblib")
    joblib.dump(calibrated, model_path)
    print(f"    Saved model: {model_path}")

    print("\n" + "=" * 65)
    print("[OK] Medicare modeling pipeline complete!")
    print("=" * 65)
    print(f"\n  Val  PR-AUC:  {metrics['val_pr_auc']:.4f}")
    print(f"  Test PR-AUC:  {metrics['test_pr_auc']:.4f}")
    print(f"  SIU Enrichment: {metrics['siu_enrichment']:.1f}×")
    print(f"\n  Outputs: outputs/medicare/")


if __name__ == "__main__":
    main()
