"""
Model bake-off and feature ablation under repeated cross-validation
====================================================================
Answers two questions with evidence rather than assertion:

  1. Is there a better model family than the tuned XGBoost?
  2. Do the 41 engineered features earn their place?

Why repeated CV and not the project's validation split: that split holds 92
fraud cases and its 95% CI on PR-AUC is roughly +/-0.08, so the differences
between XGBoost, the stack and CatBoost there are inside the noise. Repeated CV
reports a mean, a spread, and a paired per-fold comparison.

The TEST split is never touched — model selection happens on train only.

    python scripts/model_bakeoff.py            # full run (~10 min)
    python scripts/model_bakeoff.py --quick    # 5-fold x 1, fewer models

Results are written to outputs/experiments/model_bakeoff.json.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import COLS_TO_DROP, RANDOM_STATE, TARGET
from data_pipeline import build_model_dataset
from feature_engineering import (REPLACED_CATEGORICALS, engineer_features,
                                 get_engineered_numeric_cols)

OUT_DIR = os.path.join("outputs", "experiments")
IMPROVEMENT = os.path.join("outputs", "improvement")


def make_matrix(frame, fit_index):
    """One-hot + scale a frame, fitting on the training rows only."""
    cat = frame.select_dtypes(include=["object"]).columns.tolist()
    num = frame.select_dtypes(include=[np.number]).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    pre.fit(frame.loc[fit_index])
    return pre.transform(frame.loc[fit_index])


def cv_scores(fit_predict, X, y, splits):
    """Run a fit/predict callable over pre-computed folds."""
    return np.array([average_precision_score(y[te], fit_predict(X, y, tr, te))
                     for tr, te in splits])


def summarise(scores):
    return {"mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
            "folds": [round(float(s), 4) for s in scores]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5-fold x 1, skip slow models")
    args = ap.parse_args()
    n_repeats = 1 if args.quick else 2

    data = build_model_dataset(with_catboost_frames=True)
    y = data["y_train"].values
    spw = float(data["scale_pos_weight"])
    idx = data["X_train"].index

    X_cur = data["X_train_t"]
    X_cur = X_cur.toarray() if hasattr(X_cur, "toarray") else X_cur

    with open(os.path.join(IMPROVEMENT, "optuna_xgb_best_params.json")) as f:
        XGB_BEST = json.load(f)
    with open(os.path.join(IMPROVEMENT, "optuna_catboost_best_params.json")) as f:
        CB_BEST = json.load(f)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)
    splits = list(cv.split(X_cur, y))
    print(f"train {len(y):,} rows, {int(y.sum())} fraud | {len(splits)} folds "
          f"(5 x {n_repeats}) | scale_pos_weight {spw:.2f}\n")

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    def xgb():
        return XGBClassifier(n_estimators=400, scale_pos_weight=spw,
                             objective="binary:logistic", eval_metric="aucpr",
                             tree_method="hist", n_jobs=-1,
                             random_state=RANDOM_STATE, **XGB_BEST)

    builders = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=2000,
                                                         class_weight="balanced",
                                                         random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                       class_weight="balanced_subsample",
                                                       n_jobs=-1, random_state=RANDOM_STATE),
        "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3,
                                                   class_weight="balanced_subsample",
                                                   n_jobs=-1, random_state=RANDOM_STATE),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced",
            random_state=RANDOM_STATE),
        "LightGBM": lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                           scale_pos_weight=spw, n_jobs=-1,
                                           random_state=RANDOM_STATE, verbose=-1),
        "CatBoost (one-hot input)": lambda: CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=6, scale_pos_weight=spw,
            verbose=0, random_seed=RANDOM_STATE),
        "XGBoost (tuned)": xgb,
    }
    if args.quick:
        for slow in ("RandomForest", "ExtraTrees", "CatBoost (one-hot input)"):
            builders.pop(slow, None)

    def sk_fit(build):
        def run(X, y_, tr, te):
            m = build()
            m.fit(X[tr], y_[tr])
            return m.predict_proba(X[te])[:, 1]
        return run

    print("=" * 76)
    print("1. MODEL BAKE-OFF (current feature set)")
    print("=" * 76)
    results = {}
    for name, build in builders.items():
        t0 = time.perf_counter()
        s = cv_scores(sk_fit(build), X_cur, y, splits)
        results[name] = summarise(s)
        print(f"  {name:28} {s.mean():.4f} +/- {s.std():.4f}   {time.perf_counter()-t0:5.1f}s")

    # ── CatBoost the way the project actually uses it ────────────────────
    # The bake-off above feeds CatBoost the one-hot matrix, which removes the
    # native categorical handling that is its main advantage. This arm gives it
    # the raw frame, its own Optuna parameters, and the categorical indices.
    cb_native = None
    if not args.quick:
        Xc = data["X_train_full"].copy()
        for c in Xc.select_dtypes(include=["object"]).columns:
            Xc[c] = Xc[c].fillna("Unknown")
        for c in Xc.select_dtypes(include=[np.number]).columns:
            Xc[c] = Xc[c].fillna(Xc[c].median())
        cat_idx = [Xc.columns.get_loc(c)
                   for c in Xc.select_dtypes(include=["object"]).columns]
        Xc_arr = Xc.values

        def cb_run(X, y_, tr, te):
            m = CatBoostClassifier(iterations=600, scale_pos_weight=spw,
                                   cat_features=cat_idx, random_seed=RANDOM_STATE,
                                   verbose=0, eval_metric="PRAUC", **CB_BEST)
            m.fit(X[tr], y_[tr])
            return m.predict_proba(X[te])[:, 1]

        t0 = time.perf_counter()
        s = cv_scores(cb_run, Xc_arr, y, splits)
        cb_native = summarise(s)
        results["CatBoost (native cats, tuned)"] = cb_native
        print(f"  {'CatBoost (native cats, tuned)':28} {s.mean():.4f} +/- {s.std():.4f}   "
              f"{time.perf_counter()-t0:5.1f}s   <- as the project uses it")

    base = np.array(results["XGBoost (tuned)"]["folds"])
    print("\n  paired difference vs XGBoost (same folds):")
    for name, r in results.items():
        if name == "XGBoost (tuned)":
            continue
        d = np.array(r["folds"]) - base
        r["vs_xgboost"] = round(float(d.mean()), 4)
        r["folds_won_vs_xgboost"] = int((d > 0).sum())
        print(f"    {name:28} {d.mean():+.4f}   wins {int((d>0).sum())}/{len(d)}")

    # ── 2. ensembles ─────────────────────────────────────────────────────
    ensembles = {}
    if not args.quick:
        print()
        print("=" * 76)
        print("2. RANK-AVERAGE ENSEMBLES")
        print("=" * 76)
        members = {"xgb": xgb,
                   "hgb": builders["HistGradientBoosting"],
                   "lgb": builders["LightGBM"]}
        combos = {"XGB + HistGB": ("xgb", "hgb"),
                  "XGB + LightGBM": ("xgb", "lgb"),
                  "XGB + HistGB + LightGBM": ("xgb", "hgb", "lgb")}
        per_fold = {k: [] for k in combos}
        for tr, te in splits:
            preds = {}
            for key, build in members.items():
                m = build()
                m.fit(X_cur[tr], y[tr])
                preds[key] = rankdata(m.predict_proba(X_cur[te])[:, 1]) / len(te)
            for label, keys in combos.items():
                blended = np.mean([preds[k] for k in keys], axis=0)
                per_fold[label].append(average_precision_score(y[te], blended))
        for label, scores in per_fold.items():
            s = np.array(scores)
            d = s - base
            ensembles[label] = {**summarise(s),
                                "vs_xgboost": round(float(d.mean()), 4),
                                "folds_won_vs_xgboost": int((d > 0).sum())}
            print(f"  {label:28} {s.mean():.4f} +/- {s.std():.4f}   "
                  f"{d.mean():+.4f} vs XGB, wins {int((d>0).sum())}/{len(d)}")

    # ── 3. feature ablation ──────────────────────────────────────────────
    print()
    print("=" * 76)
    print("3. FEATURE ABLATION")
    print("=" * 76)
    raw_df = pd.read_csv("fraud_oracle.csv")
    eng_df = engineer_features(raw_df)
    arms = {
        # NOTE: the raw arm keeps every original predictor. Dropping
        # REPLACED_CATEGORICALS here as well would remove 17 informative columns
        # and make feature engineering look far better than it is.
        "raw originals only": raw_df.drop(columns=[TARGET] + COLS_TO_DROP),
        "current (used by model)": eng_df.drop(columns=[TARGET] + COLS_TO_DROP).drop(
            columns=[c for c in REPLACED_CATEGORICALS if c in eng_df.columns]),
        "originals + engineered": eng_df.drop(columns=[TARGET] + COLS_TO_DROP),
    }
    ablation = {}
    for label, frame in arms.items():
        X = make_matrix(frame, idx)
        X = X.toarray() if hasattr(X, "toarray") else X
        s = cv_scores(sk_fit(xgb), X, y, splits)
        ablation[label] = {**summarise(s), "n_columns": int(X.shape[1])}
        print(f"  {label:28} {X.shape[1]:>4} cols   {s.mean():.4f} +/- {s.std():.4f}")

    # pruned: only features the trained model actually uses
    try:
        import shap
        from data_pipeline import load_best_model
        sv = shap.TreeExplainer(load_best_model()).shap_values(X_cur)
        keep = np.abs(sv).mean(axis=0) > 0
        s = cv_scores(sk_fit(xgb), X_cur[:, keep], y, splits)
        ablation["pruned (non-zero SHAP)"] = {**summarise(s), "n_columns": int(keep.sum())}
        print(f"  {'pruned (non-zero SHAP)':28} {int(keep.sum()):>4} cols   "
              f"{s.mean():.4f} +/- {s.std():.4f}")
    except ImportError:
        print("  (shap not installed — skipping pruned arm)")

    cur = np.array(ablation["current (used by model)"]["folds"])
    gain = cur - np.array(ablation["raw originals only"]["folds"])
    print(f"\n  feature engineering is worth {gain.mean():+.4f} +/- {gain.std():.4f} "
          f"PR-AUC, winning {int((gain>0).sum())}/{len(gain)} folds")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "model_bakeoff.json")
    with open(path, "w") as f:
        json.dump({
            "cv": {"n_splits": 5, "n_repeats": n_repeats, "metric": "average_precision"},
            "train_rows": int(len(y)), "train_fraud": int(y.sum()),
            "models": results,
            "ensembles": ensembles,
            "ablation": ablation,
            "feature_engineering_gain": {
                "mean": round(float(gain.mean()), 4),
                "std": round(float(gain.std()), 4),
                "folds_won": int((gain > 0).sum()), "folds": len(gain),
            },
            "note": ("Only XGBoost and CatBoost use tuned hyperparameters; other "
                     "families run near-default. This answers 'is there an easy win "
                     "from another family', not 'is XGBoost intrinsically best'."),
        }, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
