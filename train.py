"""
Single Training Pipeline Entry Point
=====================================
Replaces running 4 scripts manually. One command runs everything.

Usage:
    python train.py                        # full pipeline with defaults
    python train.py --config my_config.yaml  # custom config
    python train.py --skip-improvement     # skip Optuna tuning (faster)
    python train.py --skip-rag             # skip RAG index build

All settings come from training_config.yaml.
Every run is logged to MLflow automatically.
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import yaml
import mlflow
import pandas as pd
import numpy as np


# ============================================================
# Load config
# ============================================================
def load_config(config_path="training_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# MLflow setup
# ============================================================
def setup_mlflow(cfg):
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    experiment   = cfg["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    print(f"MLflow tracking: {tracking_uri} | experiment: {experiment}")


# ============================================================
# Step 1 — Data validation
# ============================================================
def step_validate(cfg):
    print("\n" + "=" * 60)
    print("STEP 1: Data Validation")
    print("=" * 60)
    from data_validation import validate_raw_data, validate_and_report

    df = pd.read_csv(cfg["data"]["path"])
    report = validate_and_report(df)

    print(f"  Rows: {report['total_rows']:,} | Fraud rate: {report['fraud_rate']:.4f}")
    for c in report["checks"]:
        print(f"  {c['status']}  {c['check']}")

    if not report["all_passed"]:
        print("\n[WARNING] Some validation checks failed. Proceeding anyway.")

    return df, report


# ============================================================
# Step 2 — Base modeling (LR + XGBoost + LightGBM)
# ============================================================
def step_base_modeling(cfg):
    print("\n" + "=" * 60)
    print("STEP 2: Base Modeling (LR + XGBoost + LightGBM)")
    print("=" * 60)

    with mlflow.start_run(run_name="base_modeling", nested=True):
        from modeling import run_full_pipeline
        data = run_full_pipeline()

        # Log base model metrics
        mlflow.log_metrics({
            "base_xgb_val_pr_auc":  data.get("best_val_prob", [0]),
            "train_size":           len(data["y_train"]),
            "val_size":             len(data["y_val"]),
            "test_size":            len(data["y_test"]),
        })
        mlflow.log_param("random_state", cfg["data"]["random_state"])

    return data


# ============================================================
# Step 3 — SHAP explainability
# ============================================================
def step_shap(data):
    print("\n" + "=" * 60)
    print("STEP 3: SHAP Explainability")
    print("=" * 60)

    with mlflow.start_run(run_name="shap_analysis", nested=True):
        from shap_explainability import run_shap_analysis
        shap_results = run_shap_analysis(data)

        # Log top feature
        if shap_results["importance"] is not None:
            top_feature = shap_results["importance"].iloc[0]["feature"]
            top_shap    = shap_results["importance"].iloc[0]["mean_abs_shap"]
            mlflow.log_metric("top_feature_shap", float(top_shap))
            mlflow.log_param("top_feature", top_feature)

    return shap_results


# ============================================================
# Step 4 — Model improvement (Optuna + CatBoost + Stacking)
# ============================================================
def step_improvement(cfg):
    print("\n" + "=" * 60)
    print("STEP 4: Model Improvement (Optuna + CatBoost + Stacking)")
    print("=" * 60)

    with mlflow.start_run(run_name="model_improvement", nested=True):
        # Log Optuna settings
        mlflow.log_params({
            "optuna_n_trials":  cfg["optuna"]["n_trials"],
            "optuna_cv_folds":  cfg["optuna"]["cv_folds"],
            "optuna_timeout":   cfg["optuna"]["timeout"],
        })

        from model_improvement import main as run_improvement
        run_improvement()

        # Load and log final metrics from saved metadata
        metadata_path = os.path.join("outputs", "improvement", "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)

            mlflow.log_metrics({
                "val_pr_auc":         metadata["val_pr_auc"],
                "val_roc_auc":        metadata["val_roc_auc"],
                "test_pr_auc":        metadata["test_pr_auc"],
                "test_roc_auc":       metadata["test_roc_auc"],
                "val_precision_5pct": metadata["val_precision_5pct"],
                "val_recall_5pct":    metadata["val_recall_5pct"],
                "roi_x":              metadata["triage_roi"]["roi_x"],
                "net_benefit_usd":    metadata["triage_roi"]["net_benefit"],
                "fraud_caught":       metadata["triage_roi"]["fraud_caught"],
            })
            mlflow.log_param("best_model_name", metadata["best_model_name"])

            # Log model artifact
            model_path = os.path.join("outputs", "improvement", "best_model_improved.joblib")
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="model")

            print(f"\n  Best model:   {metadata['best_model_name']}")
            print(f"  Val PR-AUC:   {metadata['val_pr_auc']:.4f}")
            print(f"  Test PR-AUC:  {metadata['test_pr_auc']:.4f}")
            print(f"  ROI:          {metadata['triage_roi']['roi_x']:.1f}x")

    return metadata


# ============================================================
# Step 5 — RAG pipeline
# ============================================================
def step_rag():
    print("\n" + "=" * 60)
    print("STEP 5: RAG Pipeline (vector index)")
    print("=" * 60)

    from rag_pipeline import run_demo
    run_demo()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Fraud Triage Training Pipeline")
    parser.add_argument("--config",           default="training_config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--skip-base",        action="store_true",
                        help="Skip base modeling step")
    parser.add_argument("--skip-shap",        action="store_true",
                        help="Skip SHAP analysis step")
    parser.add_argument("--skip-improvement", action="store_true",
                        help="Skip Optuna tuning step (faster)")
    parser.add_argument("--skip-rag",         action="store_true",
                        help="Skip RAG index build")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("FRAUD TRIAGE — FULL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Config: {args.config}")

    # Load config
    cfg = load_config(args.config)

    # Setup MLflow
    setup_mlflow(cfg)

    # Master run — wraps all steps
    with mlflow.start_run(run_name="full_pipeline"):
        mlflow.log_param("config_file", args.config)
        mlflow.log_artifact(args.config, artifact_path="config")

        # Step 1: Validate data
        df, validation_report = step_validate(cfg)
        mlflow.log_metric("validation_passed", int(validation_report["all_passed"]))
        mlflow.log_metric("dataset_rows", validation_report["total_rows"])
        mlflow.log_metric("fraud_rate",   validation_report["fraud_rate"])

        # Step 2: Base modeling
        if not args.skip_base:
            data = step_base_modeling(cfg)
        else:
            print("\n[SKIP] Base modeling")
            data = None

        # Step 3: SHAP
        if not args.skip_shap and data is not None:
            shap_results = step_shap(data)
        else:
            print("\n[SKIP] SHAP analysis")

        # Step 4: Model improvement
        if not args.skip_improvement:
            metadata = step_improvement(cfg)
        else:
            print("\n[SKIP] Model improvement")

        # Step 5: RAG
        if not args.skip_rag:
            step_rag()
        else:
            print("\n[SKIP] RAG pipeline")

        elapsed = time.time() - start_time
        mlflow.log_metric("total_runtime_seconds", elapsed)

    print("\n" + "=" * 60)
    print(f"[OK] Full pipeline complete in {elapsed/60:.1f} minutes")
    print("=" * 60)
    print("\nNext steps:")
    print("  View MLflow runs:    mlflow ui")
    print("  Launch dashboard:    python -m streamlit run app.py")


if __name__ == "__main__":
    main()
