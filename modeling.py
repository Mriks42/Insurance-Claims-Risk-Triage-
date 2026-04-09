"""
Modeling Module
===============
Handles: training Logistic Regression baseline, XGBoost, LightGBM,
evaluation (PR-AUC, Precision@K, Recall@K, confusion matrix),
model comparison, triage bucket analysis, and saving all results.

Reproduces everything from the notebook's Phase 2-3 (Weeks 3-6).

Usage:
    python modeling.py                # run standalone — trains, evaluates, saves
    from modeling import ...          # import into other modules
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

from config import (
    RANDOM_STATE, XGB_PARAMS,
    PCT_SIU, PCT_MANUAL,
    METRICS_DIR, PLOTS_DIR, MODELS_DIR,
)
from data_preprocessing import get_processed_data, build_preprocessor


# ============================================================
# Evaluation helpers
# ============================================================
def precision_at_k(y_true, y_prob, k):
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    return float(y_true[top_k_idx].mean())


def recall_at_k(y_true, y_prob, k):
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    return float(y_true[top_k_idx].sum() / max(1, y_true.sum()))


def evaluate_model(y_true, y_prob, model_name="Model", pct_review=0.05,
                   save_plot=True, plot_dir=PLOTS_DIR):
    """
    Full evaluation: PR-AUC, Precision@K, Recall@K, confusion matrix,
    classification report, PR curve plot. Returns a summary dict.
    """
    y_true, y_prob = np.array(y_true), np.array(y_prob)

    pr_auc = float(average_precision_score(y_true, y_prob))
    k = max(1, int(pct_review * len(y_true)))
    prec_k = precision_at_k(y_true, y_prob, k)
    rec_k = recall_at_k(y_true, y_prob, k)

    y_pred_05 = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_05)
    report = classification_report(y_true, y_pred_05, digits=4, zero_division=0)

    print(f"\n{'=' * 10} {model_name.upper()} {'=' * 10}")
    print(f"PR-AUC:        {pr_auc:.4f}")
    print(f"Precision@5%:  {prec_k:.4f}")
    print(f"Recall@5%:     {rec_k:.4f}")
    print(f"Confusion Matrix @ 0.5:\n{cm}")
    print(f"\nClassification Report @ 0.5:\n{report}")

    # PR curve
    if save_plot:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve ({model_name})")
        plt.grid(True)
        plt.tight_layout()
        fname = os.path.join(plot_dir, f"pr_curve_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved PR curve: {fname}")

    return {
        "Model": model_name,
        "PR_AUC": pr_auc,
        "Precision_at_5pct": prec_k,
        "Recall_at_5pct": rec_k,
    }


# ============================================================
# Train Logistic Regression baseline
# ============================================================
def train_logistic_regression(preprocessor, X_train, y_train, X_val, y_val):
    """Train LR baseline using the sklearn pipeline (preprocessor + model)."""
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=5000,
            solver="saga",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    clf.fit(X_train, y_train)
    y_val_prob = clf.predict_proba(X_val)[:, 1]
    summary = evaluate_model(y_val, y_val_prob, model_name="Logistic Regression (Baseline)")
    return clf, y_val_prob, summary


# ============================================================
# Train XGBoost
# ============================================================
def train_xgboost(X_train_t, y_train, X_val_t, y_val, scale_pos_weight):
    """Train XGBoost on preprocessed data."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        os.system(f"{sys.executable} -m pip install xgboost --quiet")
        from xgboost import XGBClassifier

    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    model = XGBClassifier(**params)
    model.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)

    y_val_prob = model.predict_proba(X_val_t)[:, 1]
    summary = evaluate_model(y_val, y_val_prob, model_name="XGBoost")
    return model, y_val_prob, summary


# ============================================================
# Train LightGBM
# ============================================================
def train_lightgbm(X_train_t, y_train, X_val_t, y_val, scale_pos_weight):
    """Train LightGBM on preprocessed data."""
    try:
        import lightgbm as lgb
        from lightgbm import LGBMClassifier
    except ImportError:
        os.system(f"{sys.executable} -m pip install lightgbm --quiet")
        import lightgbm as lgb
        from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        X_train_t, y_train,
        eval_set=[(X_val_t, y_val)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0),
        ],
    )

    y_val_prob = model.predict_proba(X_val_t)[:, 1]
    summary = evaluate_model(y_val, y_val_prob, model_name="LightGBM")
    return model, y_val_prob, summary


# ============================================================
# Model comparison
# ============================================================
def compare_models(summaries):
    """Create and save a model comparison table."""
    comparison_df = pd.DataFrame(summaries).sort_values("PR_AUC", ascending=False)
    comparison_df.to_csv(os.path.join(METRICS_DIR, "model_comparison.csv"), index=False)
    print(f"\n{'=' * 10} MODEL COMPARISON {'=' * 10}")
    print(comparison_df.to_string(index=False))
    return comparison_df


# ============================================================
# Triage bucket analysis
# ============================================================
def triage_analysis(y_true, y_prob, model_name="Best Model"):
    """Assign claims to SIU / Manual Review / Approve buckets and report fraud rates."""
    results = pd.DataFrame({
        "risk_score": y_prob,
        "y_true": np.array(y_true),
    }).sort_values("risk_score", ascending=False).reset_index(drop=True)

    n = len(results)
    siu_cut = int(PCT_SIU * n)
    manual_cut = int((PCT_SIU + PCT_MANUAL) * n)

    def assign_bucket(i):
        if i < siu_cut:
            return "SIU"
        elif i < manual_cut:
            return "Manual Review"
        else:
            return "Approve"

    results["triage_bucket"] = [assign_bucket(i) for i in range(n)]

    print(f"\n{'=' * 10} TRIAGE ANALYSIS ({model_name}) {'=' * 10}")
    print("\nBucket counts:")
    print(results["triage_bucket"].value_counts().to_string())
    print("\nFraud rate by bucket:")
    fraud_rates = results.groupby("triage_bucket")["y_true"].mean().sort_index()
    print(fraud_rates.to_string())

    results.to_csv(os.path.join(METRICS_DIR, "triage_results.csv"), index=False)
    fraud_rates.to_csv(os.path.join(METRICS_DIR, "triage_fraud_rates.csv"))

    return results


# ============================================================
# Risk score distribution plot
# ============================================================
def plot_risk_distribution(y_prob, model_name="Model"):
    """Plot and save risk score histogram."""
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob, bins=30, edgecolor="black", alpha=0.7)
    plt.title(f"Risk Score Distribution ({model_name})")
    plt.xlabel("Predicted fraud probability")
    plt.ylabel("Count")
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"risk_distribution_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved risk distribution: {fname}")


# ============================================================
# Precision@K / Recall@K sweep
# ============================================================
def precision_recall_at_k_sweep(y_true, y_prob, pcts=None):
    """Compute precision and recall at various top-% levels."""
    if pcts is None:
        pcts = [1, 2, 5, 10, 20]

    y_true, y_prob = np.array(y_true), np.array(y_prob)
    n = len(y_true)
    rows = []
    for pct in pcts:
        k = max(1, int(pct / 100.0 * n))
        p = precision_at_k(y_true, y_prob, k)
        r = recall_at_k(y_true, y_prob, k)
        rows.append({"pct": pct, "k": k, "precision": p, "recall": r})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS_DIR, "precision_recall_at_k.csv"), index=False)
    print(f"\nPrecision/Recall @ K:")
    print(df.to_string(index=False))
    return df


# ============================================================
# Save best model
# ============================================================
def save_best_model(model, model_name):
    """Save the best model to disk."""
    import pickle
    path = os.path.join(MODELS_DIR, f"best_model_{model_name.lower().replace(' ', '_')}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved best model: {path}")
    return path


# ============================================================
# Main: full modeling pipeline
# ============================================================
def run_full_pipeline():
    """Run the entire modeling pipeline end-to-end. Returns data + best model."""
    print("=" * 60)
    print("Modeling Pipeline")
    print("=" * 60)

    # 1) Get preprocessed data
    data = get_processed_data()

    # 2) Train Logistic Regression baseline
    # Note: LR uses the raw X_train/X_val with the preprocessor in a Pipeline
    print("\n" + "-" * 40)
    print("Training Logistic Regression baseline...")
    print("-" * 40)
    # Build a FRESH preprocessor for the LR pipeline (to avoid conflicts)
    lr_preprocessor, _, _ = build_preprocessor(data["X_train"])
    lr_model, lr_val_prob, lr_summary = train_logistic_regression(
        lr_preprocessor, data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    )

    # 3) Train XGBoost
    print("\n" + "-" * 40)
    print("Training XGBoost...")
    print("-" * 40)
    xgb_model, xgb_val_prob, xgb_summary = train_xgboost(
        data["X_train_t"], data["y_train"],
        data["X_val_t"], data["y_val"],
        data["scale_pos_weight"],
    )

    # 4) Train LightGBM
    print("\n" + "-" * 40)
    print("Training LightGBM...")
    print("-" * 40)
    lgbm_model, lgbm_val_prob, lgbm_summary = train_lightgbm(
        data["X_train_t"], data["y_train"],
        data["X_val_t"], data["y_val"],
        data["scale_pos_weight"],
    )

    # 5) Compare models
    comparison = compare_models([lr_summary, xgb_summary, lgbm_summary])

    # 6) Select best GBM model
    gbm_candidates = {
        "XGBoost": (xgb_model, xgb_val_prob, xgb_summary),
        "LightGBM": (lgbm_model, lgbm_val_prob, lgbm_summary),
    }
    best_name = max(gbm_candidates, key=lambda k: gbm_candidates[k][2]["PR_AUC"])
    best_model, best_val_prob, best_summary = gbm_candidates[best_name]
    print(f"\n[OK] Best GBM model: {best_name} (PR-AUC: {best_summary['PR_AUC']:.4f})")

    # 7) Evaluate best model on TEST set
    print("\n" + "-" * 40)
    print(f"Evaluating {best_name} on TEST set...")
    print("-" * 40)
    best_test_prob = best_model.predict_proba(data["X_test_t"])[:, 1]
    test_summary = evaluate_model(
        data["y_test"], best_test_prob,
        model_name=f"{best_name} (Test)",
    )

    # 8) Risk distribution plot
    plot_risk_distribution(best_val_prob, model_name=f"{best_name} (Validation)")

    # 9) Precision/Recall@K sweep
    precision_recall_at_k_sweep(data["y_val"], best_val_prob)

    # 10) Triage analysis
    triage_analysis(data["y_val"], best_val_prob, model_name=best_name)

    # 11) Save best model
    model_path = save_best_model(best_model, best_name)

    # Save test summary
    with open(os.path.join(METRICS_DIR, "test_summary.json"), "w") as f:
        json.dump(test_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("[OK] Modeling pipeline complete!")
    print("=" * 60)

    return {
        **data,
        "best_model": best_model,
        "best_model_name": best_name,
        "best_val_prob": best_val_prob,
        "best_test_prob": best_test_prob,
        "model_path": model_path,
    }


if __name__ == "__main__":
    run_full_pipeline()
