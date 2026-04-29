"""
Model Improvement Script
=========================
Improvements over the base modeling pipeline:
1.  Feature engineering (41 new features, including 7 guideline-grounded flags)
2.  Bayesian hyperparameter tuning with Optuna (XGBoost + CatBoost)
3.  CatBoost with native categorical handling + its own SHAP analysis
4.  Permutation importance-based feature selection
5.  OOF (out-of-fold) stacking ensemble: XGBoost + CatBoost → LR meta-learner
6.  Calibration + reliability diagram for the stacking ensemble
7.  Full model comparison with CV mean ± std
8.  Cost-benefit ROI triage analysis

Usage:
    python model_improvement.py
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

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from config import (
    DATA_PATH, TARGET, RANDOM_STATE, COLS_TO_DROP,
    METRICS_DIR, PLOTS_DIR, MODELS_DIR, OUTPUTS_DIR,
    OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS, OPTUNA_TIMEOUT,
    AVG_FRAUD_LOSS, SIU_COST, MANUAL_COST,
    PCT_SIU, PCT_MANUAL,
)
from feature_engineering import (
    engineer_features,
    REPLACED_CATEGORICALS,
    get_engineered_numeric_cols,
)
from modeling import (
    precision_at_k,
    recall_at_k,
    evaluate_model,
    triage_analysis,
    save_best_model,
    calibrate_model,
    plot_calibration_curve,
    cross_validate_model,
)

IMPROVEMENT_DIR = os.path.join(OUTPUTS_DIR, "improvement")
os.makedirs(IMPROVEMENT_DIR, exist_ok=True)


# ============================================================
# 1) Load and engineer features
# ============================================================
def load_engineered_data():
    """Load data, engineer features, split, and build preprocessor."""
    print("\n[1] Loading and engineering features...")

    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    X = df.drop(columns=[TARGET] + COLS_TO_DROP)
    y = df[TARGET]

    # XGBoost path: drop original categoricals replaced by numeric versions
    cols_to_remove = [c for c in REPLACED_CATEGORICALS if c in X.columns]
    X_xgb = X.drop(columns=cols_to_remove)

    # CatBoost path: keep full feature set (handles cats natively)
    X_full = X.copy()

    print(f"  Features for XGBoost:  {X_xgb.shape[1]}")
    print(f"  Features for CatBoost: {X_full.shape[1]}")

    # Stratified 80/10/10 split
    X_train_xgb, X_temp_xgb, y_train, y_temp = train_test_split(
        X_xgb, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_val_xgb, X_test_xgb, y_val, y_test = train_test_split(
        X_temp_xgb, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    X_train_full, X_temp_full, _, _ = train_test_split(
        X_full, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_val_full, X_test_full, _, _ = train_test_split(
        X_temp_full, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"  Train: {X_train_xgb.shape}, Val: {X_val_xgb.shape}, Test: {X_test_xgb.shape}")

    # Build sklearn preprocessor for XGBoost
    cat_cols = X_train_xgb.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train_xgb.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )
    preprocessor.fit(X_train_xgb)

    X_train_t = preprocessor.transform(X_train_xgb)
    X_val_t   = preprocessor.transform(X_val_xgb)
    X_test_t  = preprocessor.transform(X_test_xgb)

    feature_names = list(num_cols)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    feature_names.extend(ohe.get_feature_names_out(cat_cols))

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    print(f"  Transformed features: {X_train_t.shape[1]}")
    print(f"  scale_pos_weight: {spw:.4f}")

    return {
        "X_train_xgb": X_train_xgb, "X_val_xgb": X_val_xgb, "X_test_xgb": X_test_xgb,
        "X_train_full": X_train_full, "X_val_full": X_val_full, "X_test_full": X_test_full,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_t": X_train_t, "X_val_t": X_val_t, "X_test_t": X_test_t,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "cat_cols": cat_cols, "num_cols": num_cols,
        "scale_pos_weight": spw,
    }


# ============================================================
# 2) Optuna tuning — XGBoost
# ============================================================
def tune_xgboost_optuna(data, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT):
    """
    Bayesian hyperparameter search for XGBoost using Optuna TPE.
    n_estimators is NOT in the search space — early stopping handles it.
    """
    print(f"\n[2] Optuna XGBoost tuning ({n_trials} trials, {OPTUNA_CV_FOLDS}-fold CV)...")
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        os.system(f"{sys.executable} -m pip install optuna --quiet")
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    from xgboost import XGBClassifier

    X_tr, y_tr = data["X_train_t"], data["y_train"]
    spw = data["scale_pos_weight"]

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
            "scale_pos_weight": spw,
        }
        model = XGBClassifier(
            n_estimators=500,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
        cv = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_tr, y_tr, cv=cv,
                                 scoring="average_precision", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    best_cv     = study.best_value
    print(f"  Best CV PR-AUC: {best_cv:.4f}")
    print(f"  Best params:    {best_params}")

    # Save Optuna results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(IMPROVEMENT_DIR, "optuna_xgb_trials.csv"), index=False)
    with open(os.path.join(IMPROVEMENT_DIR, "optuna_xgb_best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # Retrain on full training set with best params + early stopping
    best_model = XGBClassifier(
        n_estimators=2000,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
        scale_pos_weight=spw,
        **best_params,
    )
    best_model.fit(
        data["X_train_t"], data["y_train"],
        eval_set=[(data["X_val_t"], data["y_val"])],
        verbose=False,
    )

    val_prob = best_model.predict_proba(data["X_val_t"])[:, 1]
    summary  = evaluate_model(data["y_val"], val_prob,
                               model_name="XGBoost (Optuna)",
                               plot_dir=IMPROVEMENT_DIR)
    summary["CV_PR_AUC_mean"] = best_cv

    return best_model, val_prob, summary


# ============================================================
# 3) Optuna tuning — CatBoost
# ============================================================
def tune_catboost_optuna(data, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT):
    """
    Bayesian hyperparameter search for CatBoost using Optuna TPE.
    CatBoost-specific params: depth, l2_leaf_reg, border_count,
    bagging_temperature, random_strength.
    """
    print(f"\n[3] Optuna CatBoost tuning ({n_trials} trials, {OPTUNA_CV_FOLDS}-fold CV)...")
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        os.system(f"{sys.executable} -m pip install optuna --quiet")
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        os.system(f"{sys.executable} -m pip install catboost --quiet")
        from catboost import CatBoostClassifier, Pool

    X_train = data["X_train_full"].copy()
    X_val   = data["X_val_full"].copy()
    y_train = data["y_train"]
    y_val   = data["y_val"]
    spw     = data["scale_pos_weight"]

    cat_cols    = X_train.select_dtypes(include=["object"]).columns.tolist()
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    # Fill NaN
    for c in cat_cols:
        X_train[c] = X_train[c].fillna("Unknown")
        X_val[c]   = X_val[c].fillna("Unknown")
    for c in X_train.select_dtypes(include=[np.number]).columns:
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med)
        X_val[c]   = X_val[c].fillna(med)

    def objective(trial):
        params = {
            "depth":               trial.suggest_int("depth", 4, 10),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
            "border_count":        trial.suggest_categorical("border_count", [32, 64, 128, 254]),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength":     trial.suggest_float("random_strength", 0.5, 2.0),
        }
        # Manual CV — CatBoost can't be cloned by sklearn due to cat_features
        cv = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        X_arr = X_train.values
        y_arr = y_train.values
        for tr_idx, val_idx in cv.split(X_arr, y_arr):
            m = CatBoostClassifier(
                iterations=100,
                scale_pos_weight=float(spw),
                cat_features=cat_indices,
                random_seed=RANDOM_STATE,
                verbose=0,
                eval_metric="PRAUC",
                **params,
            )
            m.fit(X_arr[tr_idx], y_arr[tr_idx])
            prob = m.predict_proba(X_arr[val_idx])[:, 1]
            fold_scores.append(average_precision_score(y_arr[val_idx], prob))
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    best_cv     = study.best_value
    print(f"  Best CV PR-AUC: {best_cv:.4f}")
    print(f"  Best params:    {best_params}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(IMPROVEMENT_DIR, "optuna_catboost_trials.csv"), index=False)
    with open(os.path.join(IMPROVEMENT_DIR, "optuna_catboost_best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # Retrain with best params + early stopping
    best_model = CatBoostClassifier(
        iterations=2000,
        scale_pos_weight=float(spw),
        cat_features=cat_indices,
        random_seed=RANDOM_STATE,
        verbose=0,
        early_stopping_rounds=50,
        eval_metric="PRAUC",
        **best_params,
    )
    best_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    val_prob = best_model.predict_proba(X_val)[:, 1]
    summary  = evaluate_model(data["y_val"], val_prob,
                               model_name="CatBoost (Optuna)",
                               plot_dir=IMPROVEMENT_DIR)
    summary["CV_PR_AUC_mean"] = best_cv

    # CatBoost native feature importance
    importances = best_model.get_feature_importance()
    feat_imp = pd.DataFrame({
        "feature":    X_train.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(os.path.join(IMPROVEMENT_DIR, "catboost_feature_importance.csv"), index=False)

    return best_model, val_prob, summary, (X_train, X_val, data["X_test_full"].copy())


# ============================================================
# 4) CatBoost SHAP (native, faster than shap.TreeExplainer)
# ============================================================
def catboost_shap_analysis(catboost_model, X_val, y_val, feature_names, save_dir=IMPROVEMENT_DIR):
    """
    Compute SHAP values using CatBoost's native get_feature_importance(type='ShapValues').
    Returns shap_matrix (n_samples, n_features).
    """
    print("\n[4] CatBoost native SHAP analysis...")
    from catboost import Pool

    pool = Pool(X_val, label=y_val,
                cat_features=[X_val.columns.get_loc(c)
                               for c in X_val.select_dtypes(include=["object"]).columns])
    shap_raw = catboost_model.get_feature_importance(pool, type="ShapValues")
    shap_matrix = shap_raw[:, :-1]   # last column is expected value

    # Global importance
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(save_dir, "catboost_shap_importance.csv"), index=False)

    top = imp_df.head(20)
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), top["mean_abs_shap"].values[::-1])
    plt.yticks(range(20), top["feature"].values[::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 20 CatBoost SHAP Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "catboost_shap_importance.png"), dpi=150)
    plt.close()

    print(f"  CatBoost SHAP saved to: {save_dir}")
    print(f"\n  Top 10 features:")
    print(imp_df.head(10).to_string(index=False))

    return shap_matrix, imp_df


# ============================================================
# 5) Permutation importance — drop features hurting the model
# ============================================================
def permutation_feature_selection(model, X_val_t, y_val, feature_names,
                                   save_dir=IMPROVEMENT_DIR):
    """
    Compute permutation importance on the validation set.
    Features with mean importance < 0 are actively hurting the model.
    Returns list of feature names to keep.
    """
    print("\n[5] Permutation importance feature selection...")
    result = permutation_importance(
        model, X_val_t, y_val,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="average_precision",
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "feature":      feature_names,
        "importance":   result.importances_mean,
        "std":          result.importances_std,
    }).sort_values("importance", ascending=False)
    perm_df.to_csv(os.path.join(save_dir, "permutation_importance.csv"), index=False)

    negative_mask = perm_df["importance"] < 0
    drop_features = perm_df.loc[negative_mask, "feature"].tolist()
    keep_features = perm_df.loc[~negative_mask, "feature"].tolist()

    print(f"  Features with negative permutation importance (drop): {len(drop_features)}")
    if drop_features:
        print(f"  {drop_features[:10]}{'...' if len(drop_features) > 10 else ''}")
    print(f"  Features to keep: {len(keep_features)}")

    return keep_features, drop_features, perm_df


# ============================================================
# 6) OOF stacking ensemble
# ============================================================
def build_oof_stack(xgb_model, catboost_model, catboost_data,
                    data, cv_folds=3):
    """
    Build an OOF stacking ensemble (3-fold for speed).
    - XGBoost: full OOF on training set
    - CatBoost: uses already-trained model predictions (avoids slow retraining)
    Meta-learner: Logistic Regression on [xgb_oof, catboost_prob]
    """
    print(f"\n[6] Building OOF stacking ensemble ({cv_folds}-fold)...")

    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    X_train_t    = data["X_train_t"]
    X_val_t      = data["X_val_t"]
    X_test_t     = data["X_test_t"]
    y_train      = data["y_train"].values
    y_val        = data["y_val"].values
    spw          = data["scale_pos_weight"]

    X_train_cb, X_val_cb, X_test_cb = catboost_data

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    # ── XGBoost OOF ──────────────────────────────────────────
    print("  Computing XGBoost OOF predictions...")
    xgb_params = {k: v for k, v in xgb_model.get_params().items()
                  if k not in ("early_stopping_rounds",)}
    xgb_params["n_estimators"] = xgb_model.best_iteration + 1 if hasattr(xgb_model, "best_iteration") else 500
    xgb_params["early_stopping_rounds"] = None

    xgb_oof_train = np.zeros(len(y_train))
    for fold, (tr_idx, oof_idx) in enumerate(skf.split(X_train_t, y_train)):
        m = XGBClassifier(**xgb_params)
        m.fit(X_train_t[tr_idx], y_train[tr_idx], verbose=False)
        xgb_oof_train[oof_idx] = m.predict_proba(X_train_t[oof_idx])[:, 1]
        print(f"    XGB fold {fold+1}/{cv_folds} done")

    xgb_oof_val  = xgb_model.predict_proba(X_val_t)[:, 1]
    xgb_oof_test = xgb_model.predict_proba(X_test_t)[:, 1]

    # ── CatBoost: use trained model directly (no OOF retraining) ─
    print("  Using trained CatBoost predictions (no OOF retraining for speed)...")
    # For training meta-features, use cross-val predictions on train set
    # by splitting train into 3 folds and predicting each fold
    cb_oof_train = np.zeros(len(y_train))
    X_train_cb_arr = X_train_cb.values
    cat_idx_cb = [X_train_cb.columns.get_loc(c)
                  for c in X_train_cb.select_dtypes(include=["object"]).columns]

    cb_params = catboost_model.get_params()
    for key in ("early_stopping_rounds", "verbose", "iterations", "cat_features"):
        cb_params.pop(key, None)

    for fold, (tr_idx, oof_idx) in enumerate(skf.split(X_train_cb_arr, y_train)):
        m = CatBoostClassifier(
            iterations=100,   # fast proxy for OOF
            cat_features=cat_idx_cb,
            verbose=0,
            **cb_params,
        )
        m.fit(X_train_cb_arr[tr_idx], y_train[tr_idx])
        cb_oof_train[oof_idx] = m.predict_proba(X_train_cb_arr[oof_idx])[:, 1]
        print(f"    CB fold {fold+1}/{cv_folds} done")

    cb_oof_val  = catboost_model.predict_proba(X_val_cb)[:, 1]
    cb_oof_test = catboost_model.predict_proba(X_test_cb)[:, 1]

    # ── Meta-learner ─────────────────────────────────────────
    meta_X_train = np.column_stack([xgb_oof_train, cb_oof_train])
    meta_X_val   = np.column_stack([xgb_oof_val,   cb_oof_val])
    meta_X_test  = np.column_stack([xgb_oof_test,  cb_oof_test])

    meta_model = LogisticRegression(C=0.1, class_weight="balanced",
                                    random_state=RANDOM_STATE, max_iter=1000)
    meta_model.fit(meta_X_train, y_train)

    val_stack_prob  = meta_model.predict_proba(meta_X_val)[:, 1]
    test_stack_prob = meta_model.predict_proba(meta_X_test)[:, 1]

    summary = evaluate_model(data["y_val"], val_stack_prob,
                              model_name="OOF Stack (XGB + CatBoost)",
                              plot_dir=IMPROVEMENT_DIR)

    print(f"  Meta-learner coefs: XGB={meta_model.coef_[0][0]:.4f}, "
          f"CatBoost={meta_model.coef_[0][1]:.4f}")

    return meta_model, val_stack_prob, test_stack_prob, summary


# ============================================================
# 7) Compare all models
# ============================================================
def compare_all(summaries):
    """Compare all model variants and save chart."""
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame(summaries).sort_values("PR_AUC", ascending=False)
    comparison.to_csv(os.path.join(IMPROVEMENT_DIR, "model_comparison_improved.csv"), index=False)
    print(comparison[["Model", "PR_AUC", "ROC_AUC",
                       "Precision_at_5pct", "Recall_at_5pct"]].to_string(index=False))

    metrics = ["PR_AUC", "ROC_AUC", "Precision_at_5pct", "Recall_at_5pct"]
    titles  = ["PR-AUC", "ROC-AUC", "Precision@5%", "Recall@5%"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        if metric not in comparison.columns:
            continue
        bars = ax.barh(comparison["Model"], comparison[metric])
        ax.set_xlabel(title)
        ax.set_title(title)
        for bar, val in zip(bars, comparison[metric]):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(IMPROVEMENT_DIR, "model_comparison_chart.png"), dpi=150)
    plt.close()
    print(f"\nSaved comparison chart.")
    return comparison


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Model Improvement Pipeline (v2 — Optuna + Stacking)")
    print("=" * 60)

    # 1) Load + engineer
    data = load_engineered_data()
    all_summaries = []

    # 2) Optuna XGBoost
    xgb_model, xgb_val_prob, xgb_summary = tune_xgboost_optuna(data)
    all_summaries.append(xgb_summary)

    # 3) Optuna CatBoost
    catboost_model, catboost_val_prob, catboost_summary, catboost_data = tune_catboost_optuna(data)
    all_summaries.append(catboost_summary)

    # 4) CatBoost native SHAP
    X_train_cb, X_val_cb, X_test_cb = catboost_data
    cb_shap_matrix, cb_shap_imp = catboost_shap_analysis(
        catboost_model, X_val_cb, data["y_val"],
        feature_names=list(X_val_cb.columns),
    )

    # 5) Permutation importance on best single model (XGBoost)
    keep_feats, drop_feats, perm_df = permutation_feature_selection(
        xgb_model, data["X_val_t"], data["y_val"], data["feature_names"]
    )

    # 6) OOF stacking ensemble
    meta_model, stack_val_prob, stack_test_prob, stack_summary = build_oof_stack(
        xgb_model, catboost_model, catboost_data, data
    )
    all_summaries.append(stack_summary)

    # 7) Calibrate the stacking ensemble
    print("\n[7] Calibrating stacking ensemble...")
    cal_stack = calibrate_model(meta_model,
                                np.column_stack([xgb_val_prob, catboost_val_prob]),
                                data["y_val"])
    cal_stack_prob = cal_stack.predict_proba(
        np.column_stack([xgb_val_prob, catboost_val_prob])
    )[:, 1]
    plot_calibration_curve(
        np.array(data["y_val"]), stack_val_prob, cal_stack_prob,
        "OOF Stack", save_dir=IMPROVEMENT_DIR
    )

    # 8) Compare all
    comparison = compare_all(all_summaries)

    # 9) Evaluate best on test set
    best_name = comparison.iloc[0]["Model"]
    print(f"\n[OK] Best model: {best_name}")

    if "Stack" in best_name:
        test_prob = stack_test_prob
        best_model_obj = meta_model
    elif "CatBoost" in best_name:
        test_prob = catboost_model.predict_proba(X_test_cb)[:, 1]
        best_model_obj = catboost_model
    else:
        test_prob = xgb_model.predict_proba(data["X_test_t"])[:, 1]
        best_model_obj = xgb_model

    test_summary = evaluate_model(
        data["y_test"], test_prob,
        model_name=f"{best_name} (Test)",
        plot_dir=IMPROVEMENT_DIR,
    )
    with open(os.path.join(IMPROVEMENT_DIR, "test_summary_improved.json"), "w") as f:
        json.dump(test_summary, f, indent=2)

    # 10) Triage + ROI with best model
    best_val_prob = (stack_val_prob if "Stack" in best_name
                     else catboost_val_prob if "CatBoost" in best_name
                     else xgb_val_prob)
    triage_results, roi = triage_analysis(
        data["y_val"], best_val_prob, model_name=best_name
    )

    # 11) Save best model
    import joblib
    joblib.dump(best_model_obj,
                os.path.join(IMPROVEMENT_DIR, "best_model_improved.joblib"))
    print(f"\nSaved best model to: {IMPROVEMENT_DIR}/best_model_improved.joblib")

    # 11b) Register model in MLflow Model Registry
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("fraud-triage")

        with mlflow.start_run(run_name=f"register_{best_name.replace(' ', '_')}"):
            # Log metrics
            mlflow.log_metrics({
                "val_pr_auc":         float(comparison.iloc[0]["PR_AUC"]),
                "val_roc_auc":        float(comparison.iloc[0]["ROC_AUC"]),
                "test_pr_auc":        float(test_summary["PR_AUC"]),
                "test_roc_auc":       float(test_summary["ROC_AUC"]),
                "val_precision_5pct": float(comparison.iloc[0]["Precision_at_5pct"]),
                "val_recall_5pct":    float(comparison.iloc[0]["Recall_at_5pct"]),
                "roi_x":              float(roi["roi_x"]),
                "net_benefit_usd":    float(roi["net_benefit"]),
                "fraud_caught":       float(roi["fraud_caught"]),
            })
            mlflow.log_params({
                "best_model_name": best_name,
                "feature_count":   int(data["X_train_t"].shape[1]),
                "train_size":      int(len(data["y_train"])),
            })

            # Log and register the model
            model_path = os.path.join(IMPROVEMENT_DIR, "best_model_improved.joblib")
            mlflow.log_artifact(model_path, artifact_path="model")

            # Register in Model Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            registered = mlflow.register_model(
                model_uri=model_uri,
                name="fraud-triage-xgboost",
            )

            # Add description and tags
            client = MlflowClient()
            client.update_registered_model(
                name="fraud-triage-xgboost",
                description="XGBoost Optuna fraud triage model — FSE 570 Capstone",
            )
            client.set_model_version_tag(
                name="fraud-triage-xgboost",
                version=registered.version,
                key="val_pr_auc",
                value=str(round(float(comparison.iloc[0]["PR_AUC"]), 4)),
            )
            client.set_model_version_tag(
                name="fraud-triage-xgboost",
                version=registered.version,
                key="roi_x",
                value=str(round(float(roi["roi_x"]), 1)),
            )

            print(f"\n[MLflow] Model registered: fraud-triage-xgboost v{registered.version}")
            print(f"  Val PR-AUC: {comparison.iloc[0]['PR_AUC']:.4f}")
            print(f"  ROI: {roi['roi_x']:.1f}x")

    except Exception as e:
        print(f"\n[MLflow] Model registration skipped: {e}")

    # 12) Save model metadata — single source of truth for the Streamlit app
    metadata = {
        "best_model_name":  best_name,
        "best_model_path":  os.path.join(IMPROVEMENT_DIR, "best_model_improved.joblib"),
        "val_pr_auc":       float(comparison.iloc[0]["PR_AUC"]),
        "val_roc_auc":      float(comparison.iloc[0]["ROC_AUC"]),
        "val_precision_5pct": float(comparison.iloc[0]["Precision_at_5pct"]),
        "val_recall_5pct":  float(comparison.iloc[0]["Recall_at_5pct"]),
        "test_pr_auc":      float(test_summary["PR_AUC"]),
        "test_roc_auc":     float(test_summary["ROC_AUC"]),
        "cv_pr_auc_mean":   float(xgb_summary.get("CV_PR_AUC_mean", 0)),
        "feature_count":    int(data["X_train_t"].shape[1]),
        "train_size":       int(len(data["y_train"])),
        "val_size":         int(len(data["y_val"])),
        "test_size":        int(len(data["y_test"])),
        "fraud_rate":       float(data["y_train"].mean()),
        "triage_roi":       roi,
        "trained_at":       pd.Timestamp.now().isoformat(),
        "models_compared":  comparison["Model"].tolist(),
    }
    metadata_path = os.path.join(IMPROVEMENT_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {metadata_path}")

    print("\n" + "=" * 60)
    print("[OK] Model improvement pipeline complete!")
    print("=" * 60)
    print(f"\nAll outputs saved to: {IMPROVEMENT_DIR}")


if __name__ == "__main__":
    main()
