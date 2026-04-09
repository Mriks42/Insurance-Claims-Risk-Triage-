# Automotive Insurance Claims Risk Triage

> **FSE 570 Capstone Project** — Arizona State University  
> **Team Connecticut:** Mriganko Chowdhury, Aryan Gonsalves, Ashish Raj Singh, Deborah Sheryl Veluvalli, Kshama Girish

An end-to-end fraud triage system for automotive insurance claims that:
1. **Predicts** a fraud risk score for each claim (XGBoost, LightGBM, CatBoost)
2. **Explains** the key risk drivers using SHAP values and human-readable reason codes
3. **Generates** a policy-grounded triage brief using Retrieval-Augmented Generation (RAG) with citations to internal fraud guidelines
4. **Routes** each claim to the appropriate triage bucket: **SIU Escalation**, **Manual Review**, or **Approve**

---

## Table of Contents

- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
- [Critical Finding: Data Leakage](#critical-finding-data-leakage)
- [Model Performance](#model-performance)
- [SHAP Explainability](#shap-explainability)
- [RAG Pipeline](#rag-pipeline)
- [Outputs](#outputs)

---

## Project Structure

```
FSE 570 Capstone Project/
│
├── config.py                     # Shared constants, paths, hyperparameters
├── data_preprocessing.py         # Data loading, EDA, splitting, preprocessing pipeline
├── modeling.py                   # Logistic Regression, XGBoost, LightGBM training & evaluation
├── feature_engineering.py        # 34 engineered features (ordinal, interaction, risk flags)
├── model_improvement.py          # Feature eng + hyperparameter tuning + CatBoost
├── shap_explainability.py        # SHAP values, global importance, per-claim reason codes
├── rag_pipeline.py               # RAG: document retrieval + triage brief generation
├── requirements.txt              # Python dependencies
│
├── fraud_oracle.csv              # Dataset (15,420 claims, 33 columns)
│
├── docs/
│   └── fraud_guidelines/         # Curated insurance fraud guideline documents
│       ├── fraud_red_flags.md
│       ├── triage_procedures.md
│       └── policy_coverage_standards.md
│
├── notebooks/
│   └── Automotive_Insurance_Claims_Risk_Triage.ipynb   # Original exploration notebook
│
└── outputs/
    ├── eda/                      # EDA plots and data quality summaries
    ├── plots/                    # Precision-Recall curves, risk score distributions
    ├── metrics/                  # Model comparison tables, triage analysis results
    ├── models/                   # Serialized model artifacts (.pkl)
    ├── shap/                     # SHAP importance CSVs, beeswarm plots, reason codes
    ├── improvement/              # Model improvement comparison results
    └── rag/                      # RAG-generated triage briefs (JSON + text)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

Each module can be run independently or as part of the chain:

```bash
# Phase 1-3: Preprocessing + model training + evaluation
python data_preprocessing.py       # Load data, run EDA, build preprocessing pipeline
python modeling.py                 # Train LR/XGBoost/LightGBM, evaluate, compare, triage

# Phase 4: Explainability
python shap_explainability.py      # Compute SHAP values, generate reason codes
                                   # (automatically runs modeling.py first)

# Model improvement
python model_improvement.py        # Feature engineering + hyperparameter tuning + CatBoost

# Phase 5: RAG pipeline
python rag_pipeline.py             # Load guidelines, build vector index, generate demo briefs
```

> **Note:** Running `shap_explainability.py` automatically calls `modeling.py`, which calls `data_preprocessing.py` — so a single command runs the full base pipeline.

### 3. (Optional) Enable LLM-powered briefs

Set your OpenAI API key to enable GPT-generated triage briefs:

```bash
# Windows
set OPENAI_API_KEY=sk-your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=sk-your-api-key-here
```

Without an API key, the RAG pipeline uses a structured template fallback that produces professional, well-formatted briefs.

---

## Module Reference

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `config.py` | Shared settings | Paths, `TARGET`, `RANDOM_STATE`, `COLS_TO_DROP`, XGB hyperparams |
| `data_preprocessing.py` | Data pipeline | `load_data()`, `run_eda()`, `split_data()`, `build_preprocessor()`, `get_processed_data()` |
| `modeling.py` | Model training | `train_xgboost()`, `train_lightgbm()`, `evaluate_model()`, `triage_analysis()` |
| `feature_engineering.py` | Feature creation | `engineer_features()` — 34 new features from ordinal encoding, interactions, risk flags |
| `model_improvement.py` | Model optimization | XGBoost + feature eng, RandomizedSearchCV tuning, CatBoost |
| `shap_explainability.py` | Explainability | `compute_shap_values()`, `global_feature_importance()`, `make_reason_code()` |
| `rag_pipeline.py` | RAG briefs | `RAGPipeline` class, `retrieve_passages()`, `generate_brief()` |

---

## Critical Finding: Data Leakage

> **⚠️ PolicyNumber was included as a model feature in the original notebook.**

### The Problem
- `PolicyNumber` has **15,420 unique values across 15,420 rows** — it is a row identifier
- The notebook's own integrity check flagged it: `"Potential ID/high-cardinality columns: ['PolicyNumber']"`
- Despite the flag, it was included in the numeric pipeline and fed to the model
- XGBoost memorized policy numbers from the training set, inflating metrics artificially

### The Evidence

| Metric | With PolicyNumber | Without (Corrected) |
|--------|:-:|:-:|
| **Val PR-AUC** | 0.7166 | 0.2518 |
| **Test PR-AUC** | 0.7926 | 0.1951 |

The dramatic drop confirms the original model was **memorizing row identifiers** rather than learning generalizable fraud patterns. The corrected metrics are lower but honest — they represent real-world production performance.

### Resolution
- `PolicyNumber` is permanently excluded via `config.COLS_TO_DROP`
- It is preserved as a lookup column (for display in the Streamlit app)
- `RepNumber` (16 unique values — claims representative ID) is kept as a legitimate feature

---

## Model Performance

### Base Model Comparison (without feature engineering)

| Model | Val PR-AUC | Precision@5% | Recall@5% |
|-------|:----------:|:------------:|:---------:|
| **XGBoost** | **0.2518** | 0.2857 | 0.2366 |
| LightGBM | 0.1846 | 0.2208 | 0.1828 |
| Logistic Regression | 0.1677 | 0.1818 | 0.1505 |

### Improved Models (with feature engineering)

| Model | Val PR-AUC | Precision@5% | Recall@5% | vs. Base |
|-------|:----------:|:------------:|:---------:|:--------:|
| **XGBoost + Feature Eng** | **0.3149** | 0.2727 | 0.2258 | **+25.1%** |
| CatBoost | 0.3048 | 0.3117 | 0.2581 | +21.1% |
| XGBoost Tuned | 0.2997 | **0.3377** | **0.2796** | +19.0% |

### Triage Bucket Performance (Best Model)

| Bucket | Count | Fraud Rate | Enrichment vs. Base (6.0%) |
|--------|:-----:|:----------:|:-:|
| **SIU** (top 5%) | 77 | 27.3% | 4.5x |
| **Manual Review** (next 15%) | 231 | 18.6% | 3.1x |
| **Approve** (remaining 80%) | 1,234 | 2.4% | 0.4x |

---

## SHAP Explainability

### Top 10 Global Feature Importances (Mean |SHAP|)

| Rank | Feature | Mean |SHAP| |
|:----:|---------|:------------:|
| 1 | BasePolicy = Liability | 0.1081 |
| 2 | Fault = Policy Holder | 0.0956 |
| 3 | PolicyType = Sedan - Collision | 0.0139 |
| 4 | Age | 0.0105 |
| 5 | AddressChange = 2 to 3 years | 0.0080 |
| 6 | BasePolicy = All Perils | 0.0057 |
| 7 | MonthClaimed = Nov | 0.0051 |
| 8 | Deductible | 0.0050 |
| 9 | Month = Dec | 0.0031 |
| 10 | VehicleCategory = Sport | 0.0031 |

### Example Reason Codes (Rank 1 SIU Claim)

```
Risk Score: 0.5703 | Actual Fraud: Yes

[MILD] High risk: AddressChange_Claim = '2 to 3 years' (SHAP: +0.1221)
[MILD] High risk: Deductible (SHAP: +0.1181)
[MILD] High risk: BasePolicy = 'Liability' (SHAP: +0.0670)
[MILD] Low risk: Fault = 'Policy Holder' (SHAP: -0.0468)
[MILD] High risk: Age (SHAP: +0.0146)
```

---

## RAG Pipeline

### Architecture

```
Claim Data + SHAP Reason Codes
        │
        ▼
  Query Builder ─────────────────────────────────────────┐
        │                                                │
        ▼                                                │
  ChromaDB Vector Index                                  │
  (36 chunks, all-MiniLM-L6-v2 embeddings)             │
        │                                                │
        ▼                                                │
  Top-5 Relevant Passages                                │
        │                                                │
        ▼                                                │
  Brief Generator ◄─────────────────────────────────────┘
  (OpenAI GPT-4o-mini or template fallback)
        │
        ▼
  Cited Triage Brief
```

### Guideline Corpus

| Document | Description |
|----------|-------------|
| `fraud_red_flags.md` | Common fraud indicators, vehicle risk factors, demographic patterns |
| `triage_procedures.md` | Three-tier triage framework, SIU protocols, manual review procedures |
| `policy_coverage_standards.md` | Coverage types, deductible patterns, agent oversight, claim validation |

### Brief Generation Modes

- **LLM mode** (if `OPENAI_API_KEY` is set): GPT-4o-mini generates natural language briefs with cited passages
- **Template mode** (default): Structured briefs with risk assessment, claim details, cited guidelines, and investigation steps

Each generated brief includes:
- Risk score and severity level (HIGH / ELEVATED / MODERATE / LOW)
- Itemized risk factors from SHAP analysis
- Cited guideline passages (e.g., `[Source 1: fraud_red_flags.md]`)
- Bucket-specific recommended investigation steps

---

## Outputs

All pipeline outputs are saved to the `outputs/` directory:

| Directory | Contents |
|-----------|----------|
| `outputs/eda/` | Target distribution plot, missing values summary |
| `outputs/plots/` | PR curves (4), risk score distribution |
| `outputs/metrics/` | Model comparison CSV, precision@K sweep, triage results |
| `outputs/models/` | Serialized best models (`.pkl`) |
| `outputs/shap/` | Global importance CSV/PNG, beeswarm plot, claim explanations, reason codes |
| `outputs/improvement/` | Improved model comparison, tuning results, CatBoost importances |
| `outputs/rag/` | Demo triage briefs (JSON + text) |

---

## Dataset

**Source:** `fraud_oracle.csv` — 15,420 automotive insurance claims with 33 columns  
**Target:** `FraudFound_P` (binary: 0 = legitimate, 1 = fraud)  
**Fraud rate:** 5.99% (923 of 15,420 claims)  
**Split:** 80/10/10 stratified (train: 12,336 / val: 1,542 / test: 1,542)

---

## License

This project was developed as part of the FSE 570 Data Science Capstone course at Arizona State University.
