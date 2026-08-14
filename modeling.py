"""
Modeling Module
===============
Handles: training Logistic Regression baseline, XGBoost, LightGBM,
evaluation (PR-AUC, ROC-AUC, Precision@K, Recall@K, calibration,
confusion matrix at operational threshold), model comparison,
triage bucket analysis with cost-benefit ROI, and saving all results.

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
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

from config import (
    RANDOM_STATE, XGB_PARAMS,
    PCT_SIU, PCT_MANUAL,
    AVG_FRAUD_LOSS, SIU_COST, MANUAL_COST,
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
    Full evaluation:
      - PR-AUC (primary, imbalance-aware)
      - ROC-AUC (secondary, for literature comparison)
      - Precision@K / Recall@K at operational review rate
      - Confusion matrix at OPERATIONAL threshold (top pct_review%), not 0.5
      - Classification report at operational threshold
      - PR curve plot
    Returns a summary dict.
    """
    y_true, y_prob = np.array(y_true), np.array(y_prob)

    pr_auc  = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    k = max(1, int(pct_review * len(y_true)))
    prec_k = precision_at_k(y_true, y_prob, k)
    rec_k  = recall_at_k(y_true, y_prob, k)

    # Operational threshold = score at the top-pct_review percentile cutoff
    op_threshold = float(np.percentile(y_prob, 100 * (1 - pct_review)))
    y_pred_op = (y_prob >= op_threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred_op)
    report = classification_report(y_true, y_pred_op, digits=4, zero_division=0)

    print(f"\n{'=' * 10} {model_name.upper()} {'=' * 10}")
    print(f"PR-AUC:              {pr_auc:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"Precision@{int(pct_review*100)}%:       {prec_k:.4f}")
    print(f"Recall@{int(pct_review*100)}%:          {rec_k:.4f}")
    print(f"Operational threshold (top {int(pct_review*100)}%): {op_threshold:.4f}")
    print(f"Confusion Matrix @ operational threshold:\n{cm}")
    print(f"\nClassification Report @ operational threshold:\n{report}")

    # PR curve
    if save_plot:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, lw=2)
        plt.axhline(y=y_true.mean(), color="gray", linestyle="--",
                    label=f"Random baseline ({y_true.mean():.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve ({model_name})\nPR-AUC={pr_auc:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(
            plot_dir,
            f"pr_curve_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        )
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved PR curve: {fname}")

    return {
        "Model":              model_name,
        "PR_AUC":             pr_auc,
        "ROC_AUC":            roc_auc,
        "Precision_at_5pct":  prec_k,
        "Recall_at_5pct":     rec_k,
        "Op_Threshold":       op_threshold,
    }


# ============================================================
# Calibration
# ============================================================
def calibrate_model(model, X_val_t, y_val, method="isotonic"):
    """
    Fit an isotonic (or sigmoid) calibrator on top of an already-trained model.

    sklearn's CalibratedClassifierCV is deliberately not used here: cv='prefit'
    was removed in sklearn 1.2 and the FrozenEstimator replacement only exists
    in 1.6+, so wrapping the frozen estimator is not portable across the
    versions this project is expected to run on. Fitting a one-dimensional
    calibrator directly on the model's own predicted probabilities is
    equivalent and version-independent.

    NOTE: the calibrator is fit on the validation set, so evaluating it on that
    same set (as plot_calibration_curve does downstream) overstates how well
    calibrated it is. It is a diagnostic, not a held-out measurement — and the
    dashboard serves the raw model, not this wrapper.
    """
    y_prob_raw = model.predict_proba(X_val_t)[:, 1].reshape(-1, 1)

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(y_prob_raw.ravel(), y_val)
    else:
        from sklearn.linear_model import LogisticRegression as _LR
        calibrator = _LR(C=1.0)
        calibrator.fit(y_prob_raw, y_val)

    # Wrap into a simple callable class
    class _ManualCalibrated:
        def __init__(self, base, cal, method):
            self.base = base
            self.cal  = cal
            self.method = method

        def predict_proba(self, X):
            raw = self.base.predict_proba(X)[:, 1]
            if self.method == "isotonic":
                cal_prob = self.cal.predict(raw)
            else:
                cal_prob = self.cal.predict_proba(raw.reshape(-1, 1))[:, 1]
            cal_prob = np.clip(cal_prob, 0, 1)
            return np.column_stack([1 - cal_prob, cal_prob])

    wrapper = _ManualCalibrated(model, calibrator, method)
    print(f"  Calibration fitted ({method}) on {len(y_val)} validation samples.")
    return wrapper


def plot_calibration_curve(y_true, y_prob_raw, y_prob_cal, model_name, save_dir=PLOTS_DIR):
    """Plot reliability diagram: raw vs calibrated probabilities."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for probs, label, color in [
        (y_prob_raw, "Uncalibrated", "steelblue"),
        (y_prob_cal, "Calibrated (isotonic)", "darkorange"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", label=label, color=color)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability Diagram — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"calibration_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved calibration curve: {fname}")


# ============================================================
# Cross-validation score
# ============================================================
def cross_validate_model(model, X_train_t, y_train, cv_folds=5):
    """
    Run stratified k-fold CV and report mean ± std PR-AUC.
    Creates a CV-safe clone with early_stopping_rounds disabled so
    cross_val_score (which has no eval_set) doesn't error.
    Returns (mean, std).
    """
    from sklearn.base import clone

    # Build a CV-safe clone: copy all params but disable early stopping
    params = model.get_params()
    params["early_stopping_rounds"] = None
    # Fix n_estimators to the best iteration found during training
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        params["n_estimators"] = model.best_iteration + 1
    else:
        params["n_estimators"] = 300   # sensible fallback

    cv_model = model.__class__(**params)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        cv_model, X_train_t, y_train,
        cv=cv, scoring="average_precision", n_jobs=-1,
    )
    print(f"  {cv_folds}-fold CV PR-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return float(scores.mean()), float(scores.std())


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
        n_estimators=2000,
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
# Triage bucket analysis with cost-benefit ROI
# ============================================================
def triage_analysis(y_true, y_prob, model_name="Best Model",
                    avg_fraud_loss=AVG_FRAUD_LOSS,
                    siu_cost=SIU_COST, manual_cost=MANUAL_COST):
    """
    Assign claims to SIU / Manual Review / Approve buckets, report fraud
    rates per bucket, and compute estimated cost-benefit ROI.
    """
    results = pd.DataFrame({
        "risk_score": y_prob,
        "y_true":     np.array(y_true),
    }).sort_values("risk_score", ascending=False).reset_index(drop=True)

    n = len(results)
    siu_cut    = int(PCT_SIU * n)
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

    # ── Cost-benefit ROI ──────────────────────────────────────
    siu_df    = results[results["triage_bucket"] == "SIU"]
    manual_df = results[results["triage_bucket"] == "Manual Review"]

    fraud_caught = int(siu_df["y_true"].sum() + manual_df["y_true"].sum())
    savings      = fraud_caught * avg_fraud_loss
    costs        = len(siu_df) * siu_cost + len(manual_df) * manual_cost
    net_benefit  = savings - costs
    roi          = savings / max(costs, 1)

    print(f"\n── Cost-Benefit ROI ──")
    print(f"  Fraud claims caught (SIU + Manual): {fraud_caught}")
    print(f"  Estimated fraud losses prevented:   ${savings:>12,.0f}")
    print(f"  Investigation costs:                ${costs:>12,.0f}")
    print(f"  Net benefit:                        ${net_benefit:>12,.0f}")
    print(f"  ROI:                                {roi:.1f}x")

    roi_summary = {
        "fraud_caught": fraud_caught,
        "savings_usd":  savings,
        "costs_usd":    costs,
        "net_benefit":  net_benefit,
        "roi_x":        roi,
    }

    results.to_csv(os.path.join(METRICS_DIR, "triage_results.csv"), index=False)
    fraud_rates.to_csv(os.path.join(METRICS_DIR, "triage_fraud_rates.csv"))
    with open(os.path.join(METRICS_DIR, "triage_roi.json"), "w") as f:
        json.dump(roi_summary, f, indent=2)

    return results, roi_summary


# ============================================================
# Uncertainty on a metric
# ============================================================
def bootstrap_metric_ci(y_true, y_prob, metric=average_precision_score,
                        n_boot=2000, alpha=0.05, random_state=RANDOM_STATE):
    """
    Percentile bootstrap confidence interval for a ranking metric.

    Worth reporting because the test split holds only ~92 fraud cases, so the
    point estimate carries a wide interval (test PR-AUC 0.2443 sits in roughly
    [0.17, 0.33]). Quoting the interval alongside the estimate is the difference
    between "the tuned model scores higher" and "the tuned model scores higher
    than the noise floor".

    Returns (point, low, high).
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    point = float(metric(y_true, y_prob))

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0:      # undefined for an all-negative resample
            continue
        scores.append(float(metric(y_true[idx], y_prob[idx])))

    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# ============================================================
# Triage + ROI summary for an arbitrary split (no side effects)
# ============================================================
def compute_triage_summary(y_true, y_prob, split_name,
                           avg_fraud_loss=AVG_FRAUD_LOSS,
                           siu_cost=SIU_COST, manual_cost=MANUAL_COST):
    """
    Bucket a split's predictions into SIU / Manual Review / Approve and return
    per-bucket fraud rates, enrichment vs. that split's base rate, and the
    cost-benefit summary.

    Unlike triage_analysis() this writes nothing and prints nothing, so it can
    be called for the validation *and* the test split. The test-split output is
    the honest production estimate and is what the README quotes.
    """
    y_arr  = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_prob)
    base_rate = float(y_arr.mean())

    siu_cut    = int(PCT_SIU * n)
    manual_cut = int((PCT_SIU + PCT_MANUAL) * n)
    order      = np.argsort(y_prob)[::-1]
    idx = {
        "SIU":           order[:siu_cut],
        "Manual Review": order[siu_cut:manual_cut],
        "Approve":       order[manual_cut:],
    }

    buckets = {}
    for name, rows in idx.items():
        rate = float(y_arr[rows].mean()) if len(rows) else 0.0
        buckets[name] = {
            "count":      int(len(rows)),
            "fraud":      int(y_arr[rows].sum()),
            "fraud_rate": round(rate, 4),
            "enrichment": round(rate / base_rate, 2) if base_rate else 0.0,
        }

    # Score cut-points at the bucket boundaries. Persisting these is what lets a
    # single claim be bucketed at serving time — without them an API would have
    # to re-score a whole reference population to know where the top 5% starts.
    ordered = np.sort(y_prob)[::-1]
    thresholds = {
        "siu_threshold":    float(ordered[max(0, siu_cut - 1)]),
        "manual_threshold": float(ordered[max(0, manual_cut - 1)]),
    }

    fraud_caught = buckets["SIU"]["fraud"] + buckets["Manual Review"]["fraud"]
    savings = fraud_caught * avg_fraud_loss
    costs   = buckets["SIU"]["count"] * siu_cost + buckets["Manual Review"]["count"] * manual_cost

    return {
        "split":            split_name,
        "n":                n,
        "base_fraud_rate":  round(base_rate, 4),
        "thresholds":       thresholds,
        "buckets":          buckets,
        "roi": {
            "fraud_caught":  fraud_caught,
            "savings_usd":   savings,
            "costs_usd":     costs,
            "net_benefit":   savings - costs,
            # benefit-cost ratio: savings / costs (NOT net / costs, which is 1 lower)
            "roi_x":         round(savings / max(costs, 1), 2),
        },
        "cost_assumptions": {
            "avg_fraud_loss": avg_fraud_loss,
            "siu_cost":       siu_cost,
            "manual_cost":    manual_cost,
        },
    }


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
    """Save the best model to disk using joblib (faster for numpy-heavy objects)."""
    import joblib
    path = os.path.join(MODELS_DIR, f"best_model_{model_name.lower().replace(' ', '_')}.joblib")
    joblib.dump(model, path)
    print(f"\nSaved best model: {path}")
    return path


def load_best_model(model_name):
    """Load a saved model from disk."""
    import joblib
    path = os.path.join(MODELS_DIR, f"best_model_{model_name.lower().replace(' ', '_')}.joblib")
    return joblib.load(path)


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
    print("\n" + "-" * 40)
    print("Training Logistic Regression baseline...")
    print("-" * 40)
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

    # 7) Cross-validate best model
    print("\n" + "-" * 40)
    print(f"Cross-validating {best_name}...")
    print("-" * 40)
    cv_mean, cv_std = cross_validate_model(best_model, data["X_train_t"], data["y_train"])

    # 8) Calibrate best model
    print("\n" + "-" * 40)
    print(f"Calibrating {best_name}...")
    print("-" * 40)
    calibrated_model = calibrate_model(best_model, data["X_val_t"], data["y_val"])
    cal_val_prob = calibrated_model.predict_proba(data["X_val_t"])[:, 1]
    plot_calibration_curve(
        np.array(data["y_val"]), best_val_prob, cal_val_prob, best_name
    )

    # 9) Evaluate best model on TEST set
    print("\n" + "-" * 40)
    print(f"Evaluating {best_name} on TEST set...")
    print("-" * 40)
    best_test_prob = best_model.predict_proba(data["X_test_t"])[:, 1]
    test_summary = evaluate_model(
        data["y_test"], best_test_prob,
        model_name=f"{best_name} (Test)",
    )
    test_summary["CV_PR_AUC_mean"] = cv_mean
    test_summary["CV_PR_AUC_std"]  = cv_std

    # 10) Risk distribution plot
    plot_risk_distribution(best_val_prob, model_name=f"{best_name} (Validation)")

    # 11) Precision/Recall@K sweep
    precision_recall_at_k_sweep(data["y_val"], best_val_prob)

    # 12) Triage analysis with ROI
    triage_results, roi = triage_analysis(data["y_val"], best_val_prob, model_name=best_name)

    # 13) Save best model
    model_path = save_best_model(best_model, best_name)

    # Save test summary
    with open(os.path.join(METRICS_DIR, "test_summary.json"), "w") as f:
        json.dump(test_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("[OK] Modeling pipeline complete!")
    print("=" * 60)

    return {
        **data,
        "best_model":          best_model,
        "calibrated_model":    calibrated_model,
        "best_model_name":     best_name,
        "best_val_prob":       best_val_prob,
        "cal_val_prob":        cal_val_prob,
        "best_test_prob":      best_test_prob,
        "model_path":          model_path,
        "triage_results":      triage_results,
        "roi":                 roi,
    }


if __name__ == "__main__":
    run_full_pipeline()
