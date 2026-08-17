"""
Out-of-fold evaluation over the full dataset
=============================================
Scores all 15,420 claims, each by a model that never saw it.

Why this exists alongside the held-out test metric: the test split contains
1,542 claims and 92 fraud cases, which gives its PR-AUC a 95% confidence
interval about 0.16 wide. That is wide enough that the tuned model is not
statistically distinguishable from the uncorrected baseline. Out-of-fold
prediction reuses every claim exactly once as held-out data, so the estimate
rests on all 923 fraud cases and the interval narrows by roughly 3x.

What each number is for:
  * test-split PR-AUC  - the honest measurement of the DEPLOYED artifact
  * out-of-fold PR-AUC - the better-powered estimate of the PIPELINE

They answer different questions and both belong in the README. The dashboard
keeps showing the test split, because that is the split the deployed model
never saw.

Two caveats, stated rather than buried:
  1. OOF predictions come from k different models, so no single artifact
     corresponds to this score.
  2. The Optuna hyperparameters were tuned on ~80% of these same rows, so the
     estimate carries mild optimism. Nested CV would remove it.

    python scripts/evaluate_oof.py
    python scripts/evaluate_oof.py --folds 10

Writes outputs/experiments/oof_evaluation.json.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (AVG_FRAUD_LOSS, COLS_TO_DROP, MANUAL_COST, PCT_MANUAL,
                    PCT_SIU, RANDOM_STATE, SIU_COST, TARGET)
from feature_engineering import REPLACED_CATEGORICALS, engineer_features
from modeling import bootstrap_metric_ci

OUT_DIR = os.path.join("outputs", "experiments")
IMPROVEMENT = os.path.join("outputs", "improvement")


def build_frame():
    df = engineer_features(pd.read_csv("fraud_oracle.csv"))
    X = df.drop(columns=[TARGET] + COLS_TO_DROP)
    X = X.drop(columns=[c for c in REPLACED_CATEGORICALS if c in X.columns])
    return X, df[TARGET].values


def fold_preprocessor(cat, num):
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    from xgboost import XGBClassifier
    with open(os.path.join(IMPROVEMENT, "optuna_xgb_best_params.json")) as f:
        best_params = json.load(f)

    X, y = build_frame()
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Out-of-fold evaluation over all {len(y):,} claims "
          f"({int(y.sum())} fraud, {y.mean()*100:.2f}%)")
    print(f"{args.folds}-fold; the preprocessor is refitted inside each fold, because "
          f"fitting it\nonce on everything would leak held-out rows into their own "
          f"transform.\n")

    oof = np.zeros(len(y))
    per_fold = []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        pre = fold_preprocessor(cat, num)
        X_tr = pre.fit_transform(X.iloc[tr])
        X_te = pre.transform(X.iloc[te])
        spw = (y[tr] == 0).sum() / (y[tr] == 1).sum()
        model = XGBClassifier(n_estimators=400, scale_pos_weight=spw,
                              objective="binary:logistic", eval_metric="aucpr",
                              tree_method="hist", n_jobs=-1,
                              random_state=RANDOM_STATE, **best_params)
        model.fit(X_tr, y[tr])
        oof[te] = model.predict_proba(X_te)[:, 1]
        score = average_precision_score(y[te], oof[te])
        per_fold.append(round(float(score), 4))
        print(f"  fold {k}: PR-AUC {score:.4f}  ({len(te):,} claims, "
              f"{int(y[te].sum())} fraud)")

    point, lo, hi = bootstrap_metric_ci(y, oof)
    roc = roc_auc_score(y, oof)

    n = len(y)
    siu_cut, manual_cut = int(PCT_SIU * n), int((PCT_SIU + PCT_MANUAL) * n)
    order = np.argsort(oof)[::-1]
    buckets, base = {}, float(y.mean())
    for name, rows in (("SIU", order[:siu_cut]),
                       ("Manual Review", order[siu_cut:manual_cut]),
                       ("Approve", order[manual_cut:])):
        rate = float(y[rows].mean())
        buckets[name] = {"count": int(len(rows)), "fraud": int(y[rows].sum()),
                         "fraud_rate": round(rate, 4),
                         "enrichment": round(rate / base, 2)}

    caught = buckets["SIU"]["fraud"] + buckets["Manual Review"]["fraud"]
    savings = caught * AVG_FRAUD_LOSS
    costs = buckets["SIU"]["count"] * SIU_COST + buckets["Manual Review"]["count"] * MANUAL_COST

    print(f"\nALL {n:,} CLAIMS, out-of-fold")
    print(f"  PR-AUC    {point:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   width {hi-lo:.4f}")
    print(f"  ROC-AUC   {roc:.4f}")
    for name, b in buckets.items():
        print(f"  {name:14} {b['count']:>6,} claims  {b['fraud_rate']*100:5.1f}% fraud  "
              f"{b['enrichment']:.2f}x")
    print(f"  flagged top 20% catches {caught}/{int(y.sum())} fraud = "
          f"{caught/y.sum()*100:.1f}% recall")

    payload = {
        "n_claims": int(n), "n_fraud": int(y.sum()), "base_fraud_rate": round(base, 4),
        "folds": args.folds, "per_fold_pr_auc": per_fold,
        "pr_auc": round(point, 4),
        "pr_auc_ci_95": [round(lo, 4), round(hi, 4)],
        "pr_auc_ci_width": round(hi - lo, 4),
        "roc_auc": round(float(roc), 4),
        "buckets": buckets,
        "recall_at_20pct": round(float(caught / y.sum()), 4),
        "roi": {"fraud_caught": int(caught), "savings_usd": int(savings),
                "costs_usd": int(costs), "net_benefit": int(savings - costs),
                "roi_x": round(savings / costs, 2)},
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "caveats": [
            "OOF predictions come from k different models; no single deployed "
            "artifact corresponds to this score.",
            "Hyperparameters were tuned on ~80% of these rows, so the estimate "
            "carries mild optimism. Nested CV would remove it.",
            "The deployed model's honest number remains the held-out test PR-AUC.",
        ],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "oof_evaluation.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    try:
        with open(os.path.join(IMPROVEMENT, "serving_bundle.json")) as f:
            test_ci = json.load(f)["metrics"]["test_pr_auc_ci_95"]
        width = test_ci[1] - test_ci[0]
        print(f"\n  test split (1,542 claims, 92 fraud): CI width {width:.4f}")
        print(f"  full dataset ({n:,} claims, {int(y.sum())} fraud): CI width {hi-lo:.4f}")
        print(f"  narrowed {width/(hi-lo):.1f}x")
    except (FileNotFoundError, KeyError):
        pass
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
