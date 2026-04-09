# Walkthrough: SHAP Explainability + Project Refactoring + Model Improvement

## Overview

Three major changes were made in this session:

1. **SHAP Explainability (Phase 4)** — computed SHAP values, global feature importances, per-claim explanations, and human-readable reason codes for the XGBoost fraud triage model
2. **Project Refactoring** — restructured the monolithic 60-cell Colab notebook into clean, importable Python modules to prepare for the RAG pipeline (Phase 5) and Streamlit app (Phase 6)
3. **Model Improvement** — feature engineering (34 new features), hyperparameter tuning (RandomizedSearchCV), and CatBoost as an alternative model

---

## New Project Structure

```
FSE 570 Capstone Project/
├── config.py                     # Shared constants, paths, hyperparams
├── data_preprocessing.py         # Load, EDA, split, preprocessing pipeline
├── modeling.py                   # Train LR/XGB/LGBM, evaluate, compare, triage
├── feature_engineering.py        # 34 engineered features (ordinal, interaction, risk flags)
├── model_improvement.py          # Feature eng + tuned XGBoost + CatBoost
├── shap_explainability.py        # SHAP values, importance, reason codes
├── fraud_oracle.csv              # Dataset
│
├── notebooks/
│   └── Automotive_Insurance_Claims_Risk_Triage.ipynb   # Original (preserved)
│
└── outputs/
    ├── eda/                      # EDA plots and summaries
    ├── plots/                    # PR curves, risk distributions
    ├── metrics/                  # Model comparison, triage results
    ├── models/                   # Saved model artifacts (.pkl)
    ├── shap/                     # SHAP outputs (CSVs + plots)
    └── improvement/              # Model improvement results
```

### How to Run

Each module works standalone **or** as part of the chain:

```bash
python data_preprocessing.py     # Just EDA + preprocessing
python modeling.py               # Preprocessing + all model training + evaluation
python shap_explainability.py    # Everything above + SHAP analysis
python model_improvement.py      # Feature engineering + tuning + CatBoost
```

Running `shap_explainability.py` automatically calls `modeling.py`, which calls `data_preprocessing.py` — so one command runs the base pipeline. `model_improvement.py` runs independently with its own enhanced pipeline.

---

## Files Created / Modified

### [NEW] [config.py](file:///c:/Users/mchowd42/Desktop/FSE%20570%20Capstone%20Project/config.py)
Centralizes all shared settings:
- Paths (`DATA_PATH`, `OUTPUTS_DIR`, subdirectories)
- Dataset constants (`TARGET`, `RANDOM_STATE`, `COLS_TO_DROP`)
- Triage thresholds (`PCT_SIU`, `PCT_MANUAL`)
- XGBoost hyperparameters (mirrored from the notebook)
- SHAP settings (`N_REASON_CODES`)

### [NEW] [data_preprocessing.py](file:///c:/Users/mchowd42/Desktop/FSE%20570%20Capstone%20Project/data_preprocessing.py)
Extracts all data handling from the notebook:
- `load_data()` — reads `fraud_oracle.csv`
- `run_eda()` — target distribution, missing values, duplicates, high-cardinality checks; saves plots to `outputs/eda/`
- `prepare_features()` — drops `PolicyNumber`, separates X/y
- `split_data()` — stratified 80/10/10 split
- `build_preprocessor()` — `ColumnTransformer` with median imputation + StandardScaler (numeric) and constant imputation + OneHotEncoder (categorical)
- `get_processed_data()` — convenience function that runs the full pipeline and returns a dict for downstream modules

### [NEW] [modeling.py](file:///c:/Users/mchowd42/Desktop/FSE%20570%20Capstone%20Project/modeling.py)
Training and evaluation logic:
- `evaluate_model()` — computes PR-AUC, Precision@K, Recall@K, confusion matrix, classification report, and saves PR curve plots
- `train_logistic_regression()` — LR baseline with balanced class weights
- `train_xgboost()` — XGBoost with early stopping (same hyperparams as notebook)
- `train_lightgbm()` — LightGBM with early stopping
- `compare_models()` — side-by-side model comparison table
- `triage_analysis()` — assigns claims to SIU / Manual Review / Approve buckets
- `precision_recall_at_k_sweep()` — evaluates at 1%, 2%, 5%, 10%, 20% review rates
- `save_best_model()` — pickles the best model to `outputs/models/`
- `run_full_pipeline()` — runs everything end-to-end

### [MODIFIED] [shap_explainability.py](file:///c:/Users/mchowd42/Desktop/FSE%20570%20Capstone%20Project/shap_explainability.py)
Refactored to import from `config` and `modeling` instead of duplicating code:
- `compute_shap_values()` — uses `shap.TreeExplainer`
- `global_feature_importance()` — mean |SHAP| bar chart + CSV
- `shap_summary_plot()` — beeswarm plot
- `explain_top_claims()` — per-claim SHAP drivers for SIU-tier claims
- `make_reason_code()` — converts SHAP values into human-readable strings
- `generate_reason_codes()` — batch reason code generation + CSV export

### [NEW] [feature_engineering.py](file:///c:/Users/mchowd42/Desktop/FSE%20570%20Capstone%20Project/feature_engineering.py)
Creates 34 new features from the raw dataset:
- **Ordinal encoding** — converts string categories to numeric for 9 columns (`Days_Policy_Accident`, `PastNumberOfClaims`, `AgeOfVehicle`, `VehiclePrice`, etc.)
- **Time features** — `WeekGap`, `MonthGap`, `SameDayOfWeek`
- **Risk flags** — `NoPoliceReport`, `NoWitness`, `NoReportNoWitness`, `PolicyHolderFault`, `ExternalAgent`, `HighDeductible`, `YoungDriver`, `HasPastClaims`, `RecentAddressChange`
- **Interactions** — `Age_x_Deductible`, `Fault_NoPolice`, `PriceToDeductible`, `Age_x_PastClaims`, `Cars_x_Suppliments`, `PolicyAge_x_ClaimDelay`
- **Composite** — `RiskFlagCount` (sum of all binary risk flags)

### [NEW] [model_improvement.py](file:///c:/Users/mchowd42/Desktop/FSE%20570%20Capstone%20Project/model_improvement.py)
Full model improvement pipeline:
- Loads data + applies feature engineering
- Trains XGBoost with engineered features (default params)
- Runs hyperparameter tuning via `RandomizedSearchCV` (40 iterations, 3-fold CV)
- Trains CatBoost (handles categoricals natively, no one-hot encoding needed)
- Compares all approaches and evaluates the best on the test set

### [PRESERVED] [Automotive_Insurance_Claims_Risk_Triage.ipynb](file:///c:/Users/mchowd42/Desktop/FSE%20570%20Capstone%20Project/notebooks/Automotive_Insurance_Claims_Risk_Triage.ipynb)
The original notebook was **copied** (not moved) to `notebooks/`. It remains unchanged with all 60 cells and their outputs intact — it tells the exploration story for grading purposes.

---

## All Generated Outputs

### `outputs/eda/`
| File | Description |
|------|-------------|
| `target_distribution.png` | Bar chart of fraud vs. non-fraud proportions |
| `missing_values.csv` | Missing value counts per column |

### `outputs/plots/`
| File | Description |
|------|-------------|
| `pr_curve_logistic_regression_(baseline).png` | PR curve for LR baseline |
| `pr_curve_xgboost.png` | PR curve for XGBoost (validation) |
| `pr_curve_lightgbm.png` | PR curve for LightGBM (validation) |
| `pr_curve_xgboost_(test).png` | PR curve for XGBoost (test set) |
| `risk_distribution_xgboost_(validation).png` | Risk score histogram |

### `outputs/metrics/`
| File | Description |
|------|-------------|
| `model_comparison.csv` | Side-by-side PR-AUC, Precision@5%, Recall@5% |
| `precision_recall_at_k.csv` | Performance at 1%, 2%, 5%, 10%, 20% review rates |
| `triage_results.csv` | Full validation set with risk scores + triage buckets |
| `triage_fraud_rates.csv` | Fraud rate per triage bucket |
| `test_summary.json` | Test set evaluation metrics |

### `outputs/models/`
| File | Description |
|------|-------------|
| `best_model_xgboost.pkl` | Serialized base XGBoost model |
| `best_model_xgboost_engineered.pkl` | Best improved model (XGBoost + feature engineering) |

### `outputs/shap/`
| File | Description |
|------|-------------|
| `global_feature_importance.csv` | Mean \|SHAP\| for all 147 features |
| `global_feature_importance.png` | Top 20 features bar chart |
| `shap_summary_beeswarm.png` | SHAP beeswarm plot |
| `siu_claim_explanations.csv` | Per-claim SHAP drivers (77 SIU claims) |
| `siu_reason_codes.csv` | Human-readable reason codes |

### `outputs/improvement/`
| File | Description |
|------|-------------|
| `pr_curve_xgboost_+_feature_eng.png` | PR curve for XGBoost with engineered features |
| `pr_curve_xgboost_tuned.png` | PR curve for tuned XGBoost |
| `pr_curve_catboost.png` | PR curve for CatBoost |
| `pr_curve_xgboost_+_feature_eng_(test).png` | Test set PR curve for best model |
| `model_comparison_chart.png` | Side-by-side bar chart of all improved models |
| `model_comparison_improved.csv` | Comparison table of all improved models |
| `xgb_tuning_results.csv` | Full RandomizedSearchCV results (40 iterations) |
| `xgb_best_params.json` | Best hyperparameters found |
| `catboost_feature_importance.csv` | CatBoost native feature importances |
| `test_summary_improved.json` | Test set metrics for best improved model |

---

## Critical Finding: PolicyNumber Leakage

> [!CAUTION]
> **PolicyNumber was included as a model feature in the original notebook.** This is a data leakage / identifier issue.

### The Problem
- `PolicyNumber` has **15,420 unique values across 15,420 rows** — it's a row ID
- The notebook's own integrity check flagged it: `"High-cardinality columns: ['PolicyNumber']"`
- Despite the flag, it was **not removed** and was fed through the numeric pipeline
- XGBoost can exploit high-cardinality identifiers to memorize training examples

### The Evidence
| Metric | Old Model (with PolicyNumber) | New Model (without) |
|--------|------|------|
| Val PR-AUC | **0.7166** | **0.2518** |
| Test PR-AUC | **0.7926** | **0.1951** |

> [!IMPORTANT]
> The dramatic drop confirms the original model was heavily memorizing PolicyNumber. The corrected metrics are lower but honest — they represent what the system would actually achieve on unseen claims.

### Resolution
- `PolicyNumber` is listed in `config.COLS_TO_DROP` and removed before modeling
- PolicyNumber is preserved as a separate lookup column for the Streamlit app
- `RepNumber` (16 unique values) is kept — it's a legitimate categorical feature

---

## SHAP Analysis Results

### Top 10 Global Feature Importances

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------|
| 1 | BasePolicy_Liability | 0.1081 |
| 2 | Fault_Policy Holder | 0.0956 |
| 3 | PolicyType_Sedan - Collision | 0.0139 |
| 4 | Age | 0.0105 |
| 5 | AddressChange_Claim_2 to 3 years | 0.0080 |
| 6 | BasePolicy_All Perils | 0.0057 |
| 7 | MonthClaimed_Nov | 0.0051 |
| 8 | Deductible | 0.0050 |
| 9 | Month_Dec | 0.0031 |
| 10 | VehicleCategory_Sport | 0.0031 |

### Example Reason Codes (Rank 1 SIU Claim)
```
Risk Score: 0.5703 | Actual Fraud: Yes

[MILD] High risk: AddressChange_Claim_2 to 3 years (SHAP: +0.1221)
[MILD] High risk: Deductible (SHAP: +0.1181)
[MILD] High risk: BasePolicy = 'Liability' (SHAP: +0.0670)
[MILD] Low risk: Fault = 'Policy Holder' (SHAP: -0.0468)
[MILD] High risk: Age (SHAP: +0.0146)
```

### Triage Bucket Performance
| Bucket | Count | Fraud Rate | Enrichment vs. Base (6.0%) |
|--------|-------|------------|---------------------------|
| SIU (top 5%) | 77 | 27.3% | 4.5x |
| Manual Review (next 15%) | 231 | 14.3% | 2.4x |
| Approve (remaining 80%) | 1,234 | 3.2% | 0.5x |

The triage system still provides useful risk stratification even with honest metrics.

---

## Model Improvement Results

### Feature Engineering (34 New Features)
The [feature_engineering.py](file:///c:/Users/mchowd42/Desktop/FSE%20570%20Capstone%20Project/feature_engineering.py) module creates 34 new features:

| Category | Count | Examples |
|----------|-------|----------|
| Ordinal encoding | 13 | `VehiclePrice_Num`, `PastNumberOfClaims_Num`, `AgeOfVehicle_Num` |
| Time-based | 3 | `WeekGap`, `MonthGap`, `SameDayOfWeek` |
| Risk flags | 9 | `NoPoliceReport`, `NoWitness`, `NoReportNoWitness`, `YoungDriver` |
| Interaction | 6 | `Age_x_Deductible`, `Fault_NoPolice`, `PriceToDeductible` |
| Composite | 1 | `RiskFlagCount` (sum of all binary risk flags) |
| Weekend flags | 2 | `WeekendAccident`, `WeekendClaim` |

The key insight: many columns that were treated as categorical strings (e.g., `"2 to 4"`, `"more than 30"`) are actually ordinal and benefit from numeric encoding.

### Improved Model Comparison (Validation Set)

| Model | PR-AUC | Precision@5% | Recall@5% | vs. Base |
|-------|--------|-------------|-----------|----------|
| **XGBoost + Feature Eng** | **0.3149** | 0.2727 | 0.2258 | **+25.1%** |
| CatBoost | 0.3048 | 0.3117 | 0.2581 | +21.1% |
| XGBoost Tuned | 0.2997 | **0.3377** | **0.2796** | +19.0% |
| Base XGBoost (no eng) | 0.2518 | 0.2857 | 0.2366 | baseline |

> [!TIP]
> **Feature engineering provided the biggest single improvement** (+25% PR-AUC). Converting ordinal categoricals to numeric values and adding interaction/risk features was more impactful than hyperparameter tuning or switching to CatBoost.

### Best Model Test Set Performance
| Metric | Base XGBoost | XGBoost + Feature Eng | Change |
|--------|-------------|----------------------|--------|
| PR-AUC | 0.1951 | **0.2586** | +32.5% |
| Precision@5% | 0.2078 | **0.2857** | +37.5% |
| Recall@5% | 0.1739 | **0.2391** | +37.5% |

### Improved Triage Bucket Performance
| Bucket | Count | Fraud Rate | Enrichment vs. Base (6.0%) |
|--------|-------|------------|---------------------------|
| SIU (top 5%) | 77 | 27.3% | 4.5x |
| Manual Review (next 15%) | 231 | 18.6% | 3.1x |
| Approve (remaining 80%) | 1,234 | 2.4% | 0.4x |

The Approve bucket's fraud rate dropped from 3.2% to 2.4%, meaning fewer fraudulent claims are slipping through.

### Hyperparameter Tuning Details
Best parameters found by RandomizedSearchCV (40 iterations, 3-fold stratified CV):
```json
{
  "subsample": 0.6,
  "reg_lambda": 2.0,
  "reg_alpha": 0,
  "n_estimators": 200,
  "min_child_weight": 7,
  "max_depth": 8,
  "learning_rate": 0.05,
  "gamma": 1.0,
  "colsample_bytree": 1.0
}
```

---

## Why the Refactoring Matters

| Concern | Before (single notebook) | After (modules) |
|---------|-------------------------|-----------------|
| **Reproducibility** | Must re-run 60+ cells in order | `python shap_explainability.py` runs everything |
| **Imports for Streamlit** | Can't `import` from a `.ipynb` | `from modeling import load_trained_model` works |
| **Collaboration** | Merge conflicts on every cell output | Clean git diffs |
| **Testing** | Hard to unit-test notebook cells | Each function is testable |
| **Outputs** | Inline-only, lost on restart | Saved as PNG/CSV files |

---

## Next Steps

### Phase 5: RAG Pipeline (Weeks 9-10)
Create `rag_pipeline.py`:
1. Curate insurance fraud guideline documents
2. Build a vector index (sentence-transformers + FAISS/ChromaDB)
3. Integrate an LLM to generate cited triage briefs using claim data + SHAP reason codes + retrieved passages

### Phase 6: Streamlit Application (Weeks 11-12)
Create `app.py`:
1. Risk-ranked review queue (sortable table)
2. Per-claim detail view (risk score, SHAP reasons, RAG brief)
3. Summary dashboard (fraud rates, precision@K curves, global feature importance)

All modules are already importable, so the Streamlit app can do:
```python
from data_preprocessing import get_processed_data
from feature_engineering import engineer_features
from modeling import evaluate_model, triage_analysis
from shap_explainability import explain_top_claims, make_reason_code
from rag_pipeline import generate_brief  # Phase 5
```
