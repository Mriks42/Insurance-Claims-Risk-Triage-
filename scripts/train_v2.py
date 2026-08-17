"""
Train model v2 alongside v1
============================
Bundles the four modelling fixes this project's own analysis produced, as a
SECOND versioned model. v1 is never overwritten, so every published number
(test PR-AUC 0.2443, 5.20x enrichment, $763K) stays true, and v2 is promoted
only if it earns it.

  1. Impute Age == 0 from AgeOfPolicyHolder. 320 rows carry the sentinel and all
     of them are the "16 to 17" band, so the real age is recorded elsewhere.
     Uncorrected it zeroes Age_x_Deductible and Age_x_PastClaims (SHAP ranks 17
     and 13 of 90).
  2. Prune to the features v1 actually uses. 20 of 90 have mean |SHAP| of
     exactly zero — no tree ever splits on them — and removing them measured at
     -0.0008 PR-AUC, i.e. free.
  3. Drop the OOF stack and the CatBoost arm. Repeated-CV bake-off: no family
     came within 0.019 of XGBoost and every rank-average ensemble lost.
  4. Early-stop on an inner slice of TRAIN, not on the validation split. v1
     early-stops on val and then reports val, which is why 0.3223 reads higher
     than the honest 0.2877 out-of-fold.

Expect correctness, not a bigger number: the val metric should FALL, because
v1's was inflated by fix 4.

    python scripts/train_v2.py                 # full run
    python scripts/train_v2.py --trials 10     # quicker Optuna search

Writes outputs/improvement_v2/.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (COLS_TO_DROP, OPTUNA_CV_FOLDS, OPTUNA_TIMEOUT, RANDOM_STATE,
                    TARGET, TEST_SIZE, VAL_SIZE)
from data_pipeline import load_best_model
from feature_engineering import (REPLACED_CATEGORICALS, engineer_features,
                                 impute_missing_age)
from modeling import bootstrap_metric_ci, compute_triage_summary

V1_DIR = os.path.join("outputs", "improvement")
V2_DIR = os.path.join("outputs", "improvement_v2")


def selected_features_from_v1():
    """Feature names v1 actually splits on (mean |SHAP| > 0)."""
    import shap
    from data_pipeline import build_model_dataset

    v1 = build_model_dataset()
    model = load_best_model()
    sv = shap.TreeExplainer(model).shap_values(v1["X_train_t"])
    imp = np.abs(sv).mean(axis=0)
    names = np.array(v1["feature_names"])
    keep = names[imp > 0].tolist()
    print(f"  v1 uses {len(keep)} of {len(names)} features "
          f"({len(names)-len(keep)} never split on)")
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    args = ap.parse_args()
    t_start = time.perf_counter()

    print("=" * 72)
    print("TRAIN v2  (v1 is left untouched)")
    print("=" * 72)

    # ── 1. data with the Age fix ─────────────────────────────────────
    print("\n[1] Loading data with Age imputation")
    raw = pd.read_csv("fraud_oracle.csv")
    n_sentinel = int((raw["Age"] == 0).sum())
    raw_fixed = impute_missing_age(raw)
    print(f"  imputed Age for {n_sentinel} rows "
          f"(min age now {raw_fixed['Age'].min()}, was {raw['Age'].min()})")

    df = engineer_features(raw_fixed)
    X = df.drop(columns=[TARGET] + COLS_TO_DROP)
    X = X.drop(columns=[c for c in REPLACED_CATEGORICALS if c in X.columns])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp)

    cat = X_train.select_dtypes(include=["object"]).columns.tolist()
    num = X_train.select_dtypes(include=[np.number]).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    pre.fit(X_train)
    feature_names = list(num) + list(
        pre.named_transformers_["cat"].named_steps["o"].get_feature_names_out(cat))

    def tf(frame):
        m = pre.transform(frame)
        return m.toarray() if hasattr(m, "toarray") else m

    Xtr_all, Xval_all, Xte_all = tf(X_train), tf(X_val), tf(X_test)

    # ── 2. prune to the features v1 uses ─────────────────────────────
    print("\n[2] Feature pruning")
    keep_names = selected_features_from_v1()
    keep_idx = [feature_names.index(n) for n in keep_names if n in feature_names]
    kept = [feature_names[i] for i in keep_idx]
    print(f"  v2 feature space: {len(kept)} columns (from {len(feature_names)})")

    Xtr, Xval, Xte = Xtr_all[:, keep_idx], Xval_all[:, keep_idx], Xte_all[:, keep_idx]

    # ── 3. tree budget ───────────────────────────────────────────────
    # v1 early-stops on the validation split, which is also the split it reports
    # and selects on. v2 removes that by not using val during fitting at all:
    # Optuna scores candidates by CV on train at a FIXED tree budget, and the
    # final model is fit at that same budget.
    #
    # An earlier version of this script early-stopped on an inner train holdout
    # instead. That reintroduced the mismatch it was meant to remove: Optuna
    # evaluated candidates at 400 trees with no early stopping, then the final
    # fit applied patience-50 early stopping, and at the learning rate Optuna
    # selected (0.012) fifty rounds of near-flat progress read as "no
    # improvement" — the model stopped at 2 trees and scored accordingly. The
    # selection procedure and the final fit must agree.
    N_TREES = 400
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"\n[3] Fixed budget: {N_TREES} trees, no early stopping "
          f"(val untouched) | scale_pos_weight {spw:.2f}")

    # ── 4. Optuna on the new feature space ───────────────────────────
    print(f"\n[4] Optuna re-tuning ({args.trials} trials, {OPTUNA_CV_FOLDS}-fold CV on train)")
    import optuna
    from xgboost import XGBClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
        }
        m = XGBClassifier(n_estimators=N_TREES, scale_pos_weight=spw,
                          objective="binary:logistic", eval_metric="aucpr",
                          tree_method="hist", n_jobs=-1,
                          random_state=RANDOM_STATE, **params)
        cv = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
        return cross_val_score(m, Xtr, y_train, cv=cv,
                               scoring="average_precision", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=args.trials, timeout=OPTUNA_TIMEOUT,
                   show_progress_bar=False)
    print(f"  best CV PR-AUC {study.best_value:.4f}")
    print(f"  best params    {study.best_params}")

    # ── 5. final fit, early stopping on the inner holdout ────────────
    print("\n[5] Final fit")
    model = XGBClassifier(n_estimators=N_TREES, scale_pos_weight=spw,
                          objective="binary:logistic", eval_metric="aucpr",
                          tree_method="hist", n_jobs=-1,
                          random_state=RANDOM_STATE, **study.best_params)
    model.fit(Xtr, y_train, verbose=False)
    print(f"  fit {N_TREES} trees on all {Xtr.shape[0]:,} training rows "
          f"(validation split never seen)")

    val_prob  = model.predict_proba(Xval)[:, 1]
    test_prob = model.predict_proba(Xte)[:, 1]

    v2_val  = average_precision_score(y_val, val_prob)
    point, lo, hi = bootstrap_metric_ci(y_test, test_prob)
    triage = {"validation": compute_triage_summary(y_val, val_prob, "validation"),
              "test":       compute_triage_summary(y_test, test_prob, "test")}

    # ── 6. artifacts ─────────────────────────────────────────────────
    os.makedirs(V2_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(V2_DIR, "best_model_improved.joblib"))
    joblib.dump(pre, os.path.join(V2_DIR, "preprocessor.joblib"))

    bundle = {
        "model_version":  "v2",
        "model_name":     "XGBoost (Optuna, pruned features, Age imputed)",
        "trained_at":     pd.Timestamp.now().isoformat(),
        "n_features":     len(kept),
        "feature_names":  feature_names,            # full preprocessor output
        "selected_feature_names": kept,             # what the model consumes
        "raw_input_columns": sorted(set(X_train.columns)),
        "cat_cols": cat, "num_cols": num,
        "age_imputation": True,
        "thresholds": triage["test"]["thresholds"],
        "metrics": {
            "test_pr_auc": round(point, 4),
            "test_pr_auc_ci_95": [round(lo, 4), round(hi, 4)],
            "test_roc_auc": round(float(roc_auc_score(y_test, test_prob)), 4),
            "val_pr_auc": round(float(v2_val), 4),
            "cv_pr_auc": round(float(study.best_value), 4),
            "base_fraud_rate": triage["test"]["base_fraud_rate"],
        },
        "cost_assumptions": triage["test"]["cost_assumptions"],
        "changes_from_v1": [
            f"Age == 0 imputed from AgeOfPolicyHolder ({n_sentinel} rows)",
            f"Feature space pruned {len(feature_names)} -> {len(kept)} (zero-SHAP removed)",
            "OOF stack and CatBoost arm removed (bake-off: ensembles lost)",
            "Early stopping moved to an inner train holdout, so val is unused during fitting",
        ],
    }
    with open(os.path.join(V2_DIR, "serving_bundle.json"), "w") as f:
        json.dump(bundle, f, indent=2)
    with open(os.path.join(V2_DIR, "triage_summary.json"), "w") as f:
        json.dump({"model": "v2", **triage}, f, indent=2)
    with open(os.path.join(V2_DIR, "optuna_best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    # ── 7. v1 vs v2 ──────────────────────────────────────────────────
    with open(os.path.join(V1_DIR, "serving_bundle.json")) as f:
        v1b = json.load(f)
    with open(os.path.join(V1_DIR, "triage_summary.json")) as f:
        v1t = json.load(f)

    print("\n" + "=" * 72)
    print("v1 vs v2  (test split, identical rows)")
    print("=" * 72)
    rows = [
        ("features",          v1b["n_features"],            len(kept)),
        ("val PR-AUC",        v1b["metrics"]["val_pr_auc"], round(float(v2_val), 4)),
        ("test PR-AUC",       v1b["metrics"]["test_pr_auc"], round(point, 4)),
        ("test ROC-AUC",      v1b["metrics"]["test_roc_auc"],
                              round(float(roc_auc_score(y_test, test_prob)), 4)),
        ("SIU fraud rate",    v1t["test"]["buckets"]["SIU"]["fraud_rate"],
                              triage["test"]["buckets"]["SIU"]["fraud_rate"]),
        ("SIU enrichment",    v1t["test"]["buckets"]["SIU"]["enrichment"],
                              triage["test"]["buckets"]["SIU"]["enrichment"]),
        ("net benefit $",     v1t["test"]["roi"]["net_benefit"],
                              triage["test"]["roi"]["net_benefit"]),
    ]
    print(f"  {'metric':18} {'v1':>14} {'v2':>14}   delta")
    for name, a, b in rows:
        try:
            delta = f"{b - a:+.4f}" if isinstance(a, float) else f"{b - a:+,}"
        except TypeError:
            delta = ""
        print(f"  {name:18} {a:>14} {b:>14}   {delta}")

    v1_ci = v1b["metrics"]["test_pr_auc_ci_95"]
    print(f"\n  v1 test CI [{v1_ci[0]}, {v1_ci[1]}]")
    print(f"  v2 test CI [{lo:.4f}, {hi:.4f}]")
    overlap = not (hi < v1_ci[0] or lo > v1_ci[1])
    print(f"  intervals {'overlap — difference is not distinguishable' if overlap else 'are disjoint'}")
    print(f"\n  runtime {time.perf_counter() - t_start:.0f}s   artifacts -> {V2_DIR}")


if __name__ == "__main__":
    main()
