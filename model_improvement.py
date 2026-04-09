"""
Model Improvement Script
=========================
Improves model performance through:
1. Feature engineering (34 new features)
2. Hyperparameter tuning (RandomizedSearchCV on XGBoost)
3. Alternative models (CatBoost)
4. Comparison of all approaches

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
    RandomizedSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
)

from config import (
    DATA_PATH, TARGET, RANDOM_STATE, COLS_TO_DROP,
    PCT_SIU, PCT_MANUAL,
    METRICS_DIR, PLOTS_DIR, MODELS_DIR, OUTPUTS_DIR,
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

    # Prepare features — drop target + identifiers
    X = df.drop(columns=[TARGET] + COLS_TO_DROP)
    y = df[TARGET]

    # For the XGBoost approach: also drop the original categoricals
    # that we replaced with numeric versions (to avoid redundancy)
    cols_to_remove = [c for c in REPLACED_CATEGORICALS if c in X.columns]
    X_xgb = X.drop(columns=cols_to_remove)

    print(f"  Total features (all):     {X.shape[1]}")
    print(f"  Features for XGBoost:     {X_xgb.shape[1]}")

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_xgb, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Also split the full-feature X for CatBoost
    X_train_full, X_temp_full, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_val_full, X_test_full, _, _ = train_test_split(
        X_temp_full, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Fraud rates: train={y_train.mean():.4f}, val={y_val.mean():.4f}, test={y_test.mean():.4f}")

    # Build preprocessor for XGBoost
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    # Feature names
    feature_names = list(num_cols)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    feature_names.extend(ohe.get_feature_names_out(cat_cols))

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    print(f"  Transformed features: {X_train_t.shape[1]}")
    print(f"  scale_pos_weight: {spw:.4f}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_full": X_train_full, "X_val_full": X_val_full, "X_test_full": X_test_full,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_t": X_train_t, "X_val_t": X_val_t, "X_test_t": X_test_t,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "cat_cols": cat_cols, "num_cols": num_cols,
        "scale_pos_weight": spw,
    }


# ============================================================
# 2) XGBoost with engineered features (default hyperparams)
# ============================================================
def train_xgb_engineered(data):
    """Train XGBoost with engineered features using default hyperparameters."""
    print("\n[2] Training XGBoost with engineered features (default hyperparams)...")

    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=data["scale_pos_weight"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    model.fit(
        data["X_train_t"], data["y_train"],
        eval_set=[(data["X_val_t"], data["y_val"])],
        verbose=False,
    )

    val_prob = model.predict_proba(data["X_val_t"])[:, 1]
    summary = evaluate_model(
        data["y_val"], val_prob,
        model_name="XGBoost + Feature Eng",
        plot_dir=IMPROVEMENT_DIR,
    )

    return model, val_prob, summary


# ============================================================
# 3) Hyperparameter tuning with RandomizedSearchCV
# ============================================================
def tune_xgboost(data):
    """Tune XGBoost hyperparameters using RandomizedSearchCV."""
    print("\n[3] Tuning XGBoost hyperparameters (RandomizedSearchCV)...")
    print("    This may take a few minutes...")

    from xgboost import XGBClassifier

    param_distributions = {
        "n_estimators": [100, 200, 300, 500, 700],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_child_weight": [1, 3, 5, 7, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5, 1.0],
        "reg_alpha": [0, 0.001, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=data["scale_pos_weight"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=40,
        scoring=pr_auc_scorer,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(data["X_train_t"], data["y_train"])

    print(f"    Best CV PR-AUC: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")

    # Save search results
    results_df = pd.DataFrame(search.cv_results_)[
        ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    ].sort_values("rank_test_score")
    results_df.to_csv(os.path.join(IMPROVEMENT_DIR, "xgb_tuning_results.csv"), index=False)

    # Evaluate best model on validation set
    best_model = search.best_estimator_
    val_prob = best_model.predict_proba(data["X_val_t"])[:, 1]
    summary = evaluate_model(
        data["y_val"], val_prob,
        model_name="XGBoost Tuned",
        plot_dir=IMPROVEMENT_DIR,
    )

    # Save best params
    with open(os.path.join(IMPROVEMENT_DIR, "xgb_best_params.json"), "w") as f:
        json.dump(search.best_params_, f, indent=2)

    return best_model, val_prob, summary


# ============================================================
# 4) CatBoost (handles categoricals natively)
# ============================================================
def train_catboost(data):
    """Train CatBoost which handles categorical features natively."""
    print("\n[4] Training CatBoost...")

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("    Installing catboost...")
        os.system(f"{sys.executable} -m pip install catboost --quiet")
        from catboost import CatBoostClassifier

    # CatBoost works with the FULL feature set (with original categoricals)
    X_train = data["X_train_full"].copy()
    X_val = data["X_val_full"].copy()
    X_test = data["X_test_full"].copy()

    # Identify categorical columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    # Fill any NaN in categoricals
    for c in cat_cols:
        X_train[c] = X_train[c].fillna("Unknown")
        X_val[c] = X_val[c].fillna("Unknown")
        X_test[c] = X_test[c].fillna("Unknown")

    # Fill any NaN in numerics
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        median_val = X_train[c].median()
        X_train[c] = X_train[c].fillna(median_val)
        X_val[c] = X_val[c].fillna(median_val)
        X_test[c] = X_test[c].fillna(median_val)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        scale_pos_weight=data["scale_pos_weight"],
        cat_features=cat_indices,
        random_seed=RANDOM_STATE,
        verbose=0,
        early_stopping_rounds=50,
        eval_metric="PRAUC",
    )
    model.fit(
        X_train, data["y_train"],
        eval_set=(X_val, data["y_val"]),
    )

    val_prob = model.predict_proba(X_val)[:, 1]
    summary = evaluate_model(
        data["y_val"], val_prob,
        model_name="CatBoost",
        plot_dir=IMPROVEMENT_DIR,
    )

    # Save CatBoost feature importances
    importances = model.get_feature_importance()
    feat_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(os.path.join(IMPROVEMENT_DIR, "catboost_feature_importance.csv"), index=False)

    return model, val_prob, summary, (X_train, X_val, X_test)


# ============================================================
# 5) Compare all models
# ============================================================
def compare_all(summaries):
    """Compare all model variants."""
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame(summaries).sort_values("PR_AUC", ascending=False)
    comparison.to_csv(os.path.join(IMPROVEMENT_DIR, "model_comparison_improved.csv"), index=False)
    print(comparison.to_string(index=False))

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["PR_AUC", "Precision_at_5pct", "Recall_at_5pct"]
    titles = ["PR-AUC", "Precision@5%", "Recall@5%"]

    for ax, metric, title in zip(axes, metrics, titles):
        bars = ax.barh(comparison["Model"], comparison[metric])
        ax.set_xlabel(title)
        ax.set_title(title)
        for bar, val in zip(bars, comparison[metric]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(IMPROVEMENT_DIR, "model_comparison_chart.png"), dpi=150)
    plt.close()
    print(f"\nSaved comparison chart: {os.path.join(IMPROVEMENT_DIR, 'model_comparison_chart.png')}")

    return comparison


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Model Improvement Pipeline")
    print("=" * 60)

    # Load and engineer
    data = load_engineered_data()

    # Train models
    all_summaries = []

    # 2a) XGBoost with feature engineering (default params)
    xgb_eng_model, xgb_eng_prob, xgb_eng_summary = train_xgb_engineered(data)
    all_summaries.append(xgb_eng_summary)

    # 3) Tuned XGBoost
    xgb_tuned_model, xgb_tuned_prob, xgb_tuned_summary = tune_xgboost(data)
    all_summaries.append(xgb_tuned_summary)

    # 4) CatBoost
    catboost_model, catboost_prob, catboost_summary, catboost_data = train_catboost(data)
    all_summaries.append(catboost_summary)

    # 5) Compare all
    comparison = compare_all(all_summaries)

    # 6) Select best and evaluate on test set
    best_idx = comparison["PR_AUC"].idxmax()
    best_name = comparison.loc[best_idx, "Model"]
    print(f"\n[OK] Best model: {best_name}")

    # Evaluate best on test
    print("\n" + "-" * 40)
    print(f"Evaluating {best_name} on TEST set...")
    print("-" * 40)

    if "CatBoost" in best_name:
        _, X_val_cb, X_test_cb = catboost_data
        test_prob = catboost_model.predict_proba(X_test_cb)[:, 1]
        test_summary = evaluate_model(
            data["y_test"], test_prob,
            model_name=f"{best_name} (Test)",
            plot_dir=IMPROVEMENT_DIR,
        )
        save_best_model(catboost_model, "catboost_improved")
    elif "Tuned" in best_name:
        test_prob = xgb_tuned_model.predict_proba(data["X_test_t"])[:, 1]
        test_summary = evaluate_model(
            data["y_test"], test_prob,
            model_name=f"{best_name} (Test)",
            plot_dir=IMPROVEMENT_DIR,
        )
        save_best_model(xgb_tuned_model, "xgboost_tuned")
    else:
        test_prob = xgb_eng_model.predict_proba(data["X_test_t"])[:, 1]
        test_summary = evaluate_model(
            data["y_test"], test_prob,
            model_name=f"{best_name} (Test)",
            plot_dir=IMPROVEMENT_DIR,
        )
        save_best_model(xgb_eng_model, "xgboost_engineered")

    # Save test summary
    with open(os.path.join(IMPROVEMENT_DIR, "test_summary_improved.json"), "w") as f:
        json.dump(test_summary, f, indent=2)

    # Triage analysis with best model
    triage_analysis(data["y_val"],
                    catboost_prob if "CatBoost" in best_name
                    else (xgb_tuned_prob if "Tuned" in best_name else xgb_eng_prob),
                    model_name=best_name)

    print("\n" + "=" * 60)
    print("[OK] Model improvement complete!")
    print("=" * 60)
    print(f"\nAll improvement outputs saved to: {IMPROVEMENT_DIR}")


if __name__ == "__main__":
    main()
