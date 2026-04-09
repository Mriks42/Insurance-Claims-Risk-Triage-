"""
Data Preprocessing Module
=========================
Handles: loading, cleaning, splitting, and building the sklearn preprocessing
pipeline. Reproduces everything from the notebook's Phase 1-2 (Weeks 1-4).

Usage:
    python data_preprocessing.py          # run standalone with EDA plots
    from data_preprocessing import ...    # import into other modules
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

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    DATA_PATH, TARGET, RANDOM_STATE, COLS_TO_DROP,
    EDA_DIR, OUTPUTS_DIR,
)


# ============================================================
# 1) Load dataset
# ============================================================
def load_data(path=DATA_PATH):
    """Load the fraud_oracle.csv dataset."""
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ============================================================
# 2) Basic EDA + integrity checks
# ============================================================
def run_eda(df, save_dir=EDA_DIR):
    """Run EDA checks and save summary plots. Returns a dict of findings."""
    findings = {}

    # --- Target distribution ---
    counts = df[TARGET].value_counts(dropna=False)
    props = df[TARGET].value_counts(normalize=True, dropna=False)
    findings["target_counts"] = counts.to_dict()
    findings["target_proportions"] = props.to_dict()
    findings["fraud_rate"] = float(props.get(1, 0))

    print(f"\nTarget distribution:")
    print(f"  Non-fraud (0): {counts.get(0, 0)} ({props.get(0, 0):.4f})")
    print(f"  Fraud (1):     {counts.get(1, 0)} ({props.get(1, 0):.4f})")

    # Target bar chart
    plt.figure(figsize=(6, 4))
    props.sort_index().plot(kind="bar")
    plt.title("Target Class Proportions")
    plt.xlabel(TARGET)
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "target_distribution.png"), dpi=150)
    plt.close()

    # --- Missing values ---
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct,
    })
    missing_summary.to_csv(os.path.join(save_dir, "missing_values.csv"))
    total_missing = int(missing.sum())
    findings["total_missing_values"] = total_missing
    print(f"\nMissing values: {total_missing} total")

    # --- Duplicates ---
    dup_count = int(df.duplicated().sum())
    findings["duplicate_rows"] = dup_count
    print(f"Duplicate rows: {dup_count}")

    # --- High-cardinality (ID-like) columns ---
    high_card = [c for c in df.columns if df[c].nunique(dropna=False) > 0.9 * len(df)]
    findings["high_cardinality_cols"] = high_card
    print(f"High-cardinality columns (>90% unique): {high_card}")

    # --- Data types ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    findings["numeric_cols"] = num_cols
    findings["categorical_cols"] = cat_cols
    print(f"Numeric columns ({len(num_cols)}): {num_cols}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    print(f"\nEDA outputs saved to: {save_dir}")
    return findings


# ============================================================
# 3) Prepare features and target
# ============================================================
def prepare_features(df, cols_to_drop=COLS_TO_DROP):
    """
    Separate features (X) and target (y), dropping identifier columns.
    Also returns policy_numbers as a separate Series for downstream lookup.
    """
    X = df.drop(columns=[TARGET] + cols_to_drop)
    y = df[TARGET]
    policy_numbers = df["PolicyNumber"].copy() if "PolicyNumber" in df.columns else None

    print(f"\nFeature preparation:")
    print(f"  Dropped from features: {cols_to_drop}")
    print(f"  Feature matrix: {X.shape}")
    return X, y, policy_numbers


# ============================================================
# 4) Stratified train/val/test split (80/10/10)
# ============================================================
def split_data(X, y, random_state=RANDOM_STATE):
    """Stratified 80/10/10 split. Returns (X_train, X_val, X_test, y_train, y_val, y_test)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    print(f"\nData split (80/10/10):")
    print(f"  Train: {X_train.shape}  fraud_rate={y_train.mean():.4f}")
    print(f"  Val:   {X_val.shape}  fraud_rate={y_val.mean():.4f}")
    print(f"  Test:  {X_test.shape}  fraud_rate={y_test.mean():.4f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 5) Build preprocessing pipeline
# ============================================================
def build_preprocessor(X_train):
    """
    Build and fit a ColumnTransformer on the training data.
    Returns (preprocessor, cat_cols, num_cols).
    """
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

    print(f"\nPreprocessor built and fitted:")
    print(f"  Numeric features ({len(num_cols)}): {num_cols}")
    print(f"  Categorical features ({len(cat_cols)}): {cat_cols}")

    return preprocessor, cat_cols, num_cols


def transform_data(preprocessor, X_train, X_val, X_test):
    """Transform all splits using the fitted preprocessor."""
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    print(f"  Transformed shapes: train={X_train_t.shape}, val={X_val_t.shape}, test={X_test_t.shape}")
    return X_train_t, X_val_t, X_test_t


def get_feature_names(preprocessor, num_cols, cat_cols):
    """Extract human-readable feature names from the fitted ColumnTransformer."""
    names = list(num_cols)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(cat_cols)
    names.extend(cat_names)
    return names


# ============================================================
# 6) Full pipeline convenience function
# ============================================================
def get_processed_data():
    """
    Run the entire preprocessing pipeline end-to-end.
    Returns a dict with all data objects needed by downstream modules.
    """
    df = load_data()
    X, y, policy_numbers = prepare_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)
    X_train_t, X_val_t, X_test_t = transform_data(preprocessor, X_train, X_val, X_test)
    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}")

    return {
        "df": df,
        "X": X, "y": y,
        "policy_numbers": policy_numbers,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_t": X_train_t, "X_val_t": X_val_t, "X_test_t": X_test_t,
        "preprocessor": preprocessor,
        "cat_cols": cat_cols, "num_cols": num_cols,
        "feature_names": feature_names,
        "scale_pos_weight": scale_pos_weight,
    }


# ============================================================
# Main: run standalone for EDA
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Data Preprocessing & EDA")
    print("=" * 60)

    df = load_data()
    findings = run_eda(df)
    X, y, pn = prepare_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)
    X_train_t, X_val_t, X_test_t = transform_data(preprocessor, X_train, X_val, X_test)
    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)

    print(f"\nTotal transformed features: {len(feature_names)}")
    print("\n[OK] Preprocessing complete.")
