"""
Shared Dataset Builder
======================
Single source of truth for the "raw CSV -> engineered features -> 80/10/10 split
-> fitted preprocessor" path.

Before this module existed, that block was copy-pasted into five places
(model_improvement, app, and the standalone entry points of fairness_analysis,
monitoring and temporal_analysis). Any edit to the split ratio, the imputer
strategy or REPLACED_CATEGORICALS had to be made five times, and a missed one
would silently feed the trained model a different feature space than it saw
during training.

Usage:
    from data_pipeline import build_model_dataset, load_best_model, raw_rows_for

    data  = build_model_dataset()
    model = load_best_model()
    test_prob = model.predict_proba(data["X_test_t"])[:, 1]
    raw_test  = raw_rows_for(data, "test")
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    DATA_PATH, TARGET, COLS_TO_DROP, RANDOM_STATE, OUTPUTS_DIR,
    TEST_SIZE, VAL_SIZE,
)
from feature_engineering import engineer_features, REPLACED_CATEGORICALS

BEST_MODEL_PATH = os.path.join(OUTPUTS_DIR, "improvement", "best_model_improved.joblib")


def build_preprocessing_pipeline(cat_cols, num_cols):
    """The ColumnTransformer used everywhere: median-impute + scale numerics,
    constant-impute + one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )


def build_model_dataset(data_path=DATA_PATH, random_state=RANDOM_STATE,
                        with_catboost_frames=False, verbose=False):
    """
    Load the raw CSV, engineer features, split 80/10/10 stratified, and fit the
    preprocessor on the training split only.

    Two feature views are produced from the same split:
      * the XGBoost view  - original categoricals that were replaced by numeric
                            encodings are dropped, then one-hot encoded
      * the CatBoost view - full frame kept, categoricals handled natively
                            (only built when with_catboost_frames=True)

    Returns a dict consumed by the training pipeline, the dashboard and every
    standalone analysis script.
    """
    df_raw = pd.read_csv(data_path)
    df_eng = engineer_features(df_raw)

    X = df_eng.drop(columns=[TARGET] + COLS_TO_DROP)
    y = df_eng[TARGET]

    # XGBoost view: drop originals that engineer_features() replaced with numerics
    cols_to_remove = [c for c in REPLACED_CATEGORICALS if c in X.columns]
    X_xgb = X.drop(columns=cols_to_remove)

    # 80/10/10 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_xgb, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=random_state, stratify=y_temp
    )

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)
    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_val_t   = preprocessor.transform(X_val)
    X_test_t  = preprocessor.transform(X_test)

    feature_names = list(num_cols)
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    feature_names.extend(ohe.get_feature_names_out(cat_cols))

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    data = {
        "df_raw": df_raw, "df_eng": df_eng,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_t": X_train_t, "X_val_t": X_val_t, "X_test_t": X_test_t,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "cat_cols": cat_cols, "num_cols": num_cols,
        "scale_pos_weight": scale_pos_weight,
    }

    if with_catboost_frames:
        # Same split indices, full feature frame (CatBoost handles categoricals natively)
        X_full = X
        data["X_train_full"] = X_full.loc[X_train.index]
        data["X_val_full"]   = X_full.loc[X_val.index]
        data["X_test_full"]  = X_full.loc[X_test.index]

    if verbose:
        print(f"  Rows: {len(df_raw):,} | XGB features: {X_xgb.shape[1]} "
              f"| transformed: {X_train_t.shape[1]}")
        print(f"  Train: {X_train.shape}  fraud_rate={y_train.mean():.4f}")
        print(f"  Val:   {X_val.shape}  fraud_rate={y_val.mean():.4f}")
        print(f"  Test:  {X_test.shape}  fraud_rate={y_test.mean():.4f}")
        print(f"  scale_pos_weight: {scale_pos_weight:.4f}")

    return data


IMPROVEMENT_DIR   = os.path.join(OUTPUTS_DIR, "improvement")
PREPROCESSOR_PATH = os.path.join(IMPROVEMENT_DIR, "preprocessor.joblib")
BUNDLE_PATH       = os.path.join(IMPROVEMENT_DIR, "serving_bundle.json")


def load_best_model(path=BEST_MODEL_PATH):
    """Load the serialized best model produced by model_improvement.py."""
    import joblib
    return joblib.load(path)


class ArtifactsMissing(RuntimeError):
    """Raised when the serving artifacts are absent — never fall back silently."""


def load_serving_artifacts(improvement_dir=IMPROVEMENT_DIR):
    """
    Load everything needed to score a claim WITHOUT the training dataset:
    the model, the fitted preprocessor, and the serving bundle (feature order,
    bucket thresholds, metrics).

    This is the boundary between training and serving. Anything that scores a
    claim should come through here rather than calling build_model_dataset(),
    which re-reads fraud_oracle.csv and re-fits the ColumnTransformer — the
    train/serve skew risk this exists to remove.

    Raises ArtifactsMissing with the exact path when something is absent, so a
    misconfigured deployment fails loudly at boot instead of scoring claims with
    a transform that was never fitted on the training data.
    """
    import json
    import joblib

    model_path = os.path.join(improvement_dir, "best_model_improved.joblib")
    pre_path   = os.path.join(improvement_dir, "preprocessor.joblib")
    bundle_p   = os.path.join(improvement_dir, "serving_bundle.json")

    for path, what in ((model_path, "trained model"),
                       (pre_path,   "fitted preprocessor"),
                       (bundle_p,   "serving bundle")):
        if not os.path.exists(path):
            raise ArtifactsMissing(
                f"Missing {what}: {path}. Run `python train.py` to generate the "
                f"serving artifacts."
            )

    with open(bundle_p) as f:
        bundle = json.load(f)

    return {
        "model":        joblib.load(model_path),
        "preprocessor": joblib.load(pre_path),
        "bundle":       bundle,
    }


def score_claims_frame(raw_df, artifacts):
    """
    Score a frame of RAW claims (the dataset's own column layout) end to end.

    Applies the same engineer -> drop replaced categoricals -> transform path the
    model was trained under, using the persisted preprocessor. Shared by the API
    and by any batch scoring job so the two cannot drift apart.

    Returns (probabilities, transformed_matrix).
    """
    import numpy as np

    eng = engineer_features(raw_df.copy())
    X = eng.drop(columns=[c for c in [TARGET] + COLS_TO_DROP if c in eng.columns],
                 errors="ignore")
    X = X.drop(columns=[c for c in REPLACED_CATEGORICALS if c in X.columns])

    pre      = artifacts["preprocessor"]
    cat_cols = artifacts["bundle"]["cat_cols"]
    num_cols = artifacts["bundle"]["num_cols"]

    # Align to the training column space. Missing categoricals become "Unknown"
    # (the value the fitted imputer expects); missing numerics become 0.
    for col in cat_cols:
        if col not in X.columns:
            X[col] = "Unknown"
    for col in num_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[cat_cols + num_cols]

    X_t  = pre.transform(X)
    prob = artifacts["model"].predict_proba(X_t)[:, 1]
    return np.asarray(prob), X_t


def assign_bucket_from_thresholds(score, thresholds):
    """Bucket a single score against the persisted cut-points."""
    if score >= thresholds["siu_threshold"]:
        return "SIU"
    if score >= thresholds["manual_threshold"]:
        return "Manual Review"
    return "Approve"


def raw_rows_for(data, split="test"):
    """
    Return the raw (un-engineered) rows corresponding to a split, in the same
    row order as that split's feature matrix and labels.
    """
    y = data[f"y_{split}"]
    return data["df_raw"].iloc[y.index].reset_index(drop=True)
