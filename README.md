---
title: Insurance Fraud Triage
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: ML-powered automotive insurance fraud triage dashboard
---

# Automotive Insurance Claims Risk Triage

> **FSE 570 Capstone Project** — Arizona State University
> **Team Connecticut:** Mriganko Chowdhury, Aryan Gonsalves, Ashish Raj Singh, Deborah Sheryl Veluvalli, Kshama Girish

🚀 **Live Demo:** [https://huggingface.co/spaces/Mriks/fraud-triage](https://huggingface.co/spaces/Mriks/fraud-triage)

> The app is hosted on Hugging Face Spaces (free tier). If it's been inactive it may take ~30 seconds to wake up on first visit.

An end-to-end fraud triage system for automotive insurance claims that:
1. **Predicts** a fraud risk score for each claim (XGBoost, CatBoost, OOF Stacking Ensemble)
2. **Explains** the key risk drivers using SHAP values and human-readable reason codes
3. **Generates** a policy-grounded triage brief using Retrieval-Augmented Generation (RAG) with citations to internal fraud guidelines
4. **Routes** each claim to the appropriate triage bucket: **SIU Escalation**, **Manual Review**, or **Approve**
5. **Analyses fairness** across demographic groups (age, sex, marital status) using the 80% disparate impact rule
6. **Monitors** model behavior over time with batch drift detection
7. **Tracks** month-by-month performance to detect model decay
8. **Visualises** everything in an interactive Streamlit dashboard with 7 pages

---

## Screenshots

| Summary Dashboard | Review Queue |
|:-:|:-:|
| ![Summary Dashboard](images/1.png) | ![Review Queue](images/2.png) |

| Claim Detail + RAG Brief | Live Scoring |
|:-:|:-:|
| ![Claim Detail](images/3.png) | ![Live Scoring](images/4.png) |

---

## Table of Contents

- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Docker](#docker)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Module Reference](#module-reference)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Tests](#tests)
- [Critical Finding: Data Leakage](#critical-finding-data-leakage)
- [Model Performance](#model-performance)
- [Fairness Analysis](#fairness-analysis)
- [Model Monitoring](#model-monitoring)
- [Temporal Analysis](#temporal-analysis)
- [SHAP Explainability](#shap-explainability)
- [RAG Pipeline](#rag-pipeline)
- [Key Improvements](#key-improvements-over-original-notebook)
- [Outputs](#outputs)

---

## Project Structure

```
FSE 570 Capstone Project/
│
├── app.py                        # Streamlit dashboard (7 pages)
├── config.py                     # Shared constants, paths, hyperparameters
├── training_config.yaml          # All training settings in one YAML file
├── train.py                      # Single command runs the full pipeline
│
├── data_preprocessing.py         # Data loading, EDA, splitting, preprocessing pipeline
├── data_validation.py            # Pandera schema checks on raw data
├── feature_engineering.py        # 41 engineered features
├── modeling.py                   # LR baseline, XGBoost, LightGBM
├── model_improvement.py          # Optuna tuning + CatBoost + OOF stacking ensemble
├── shap_explainability.py        # SHAP values, global importance, per-claim reason codes
├── rag_pipeline.py               # RAG: persistent ChromaDB index + triage brief generation
│
├── fairness_analysis.py          # Disparate impact analysis (age, sex, marital status)
├── monitoring.py                 # Batch drift detection (Evidently + KS test fallback)
├── temporal_analysis.py          # Month-by-month performance analysis
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for OpenAI API key
├── pytest.ini                    # Test configuration
├── Dockerfile                    # Container definition
├── docker-compose.yml            # App + MLflow server
│
├── tests/
│   ├── test_feature_engineering.py   # 30+ feature engineering tests
│   ├── test_preprocessing.py         # Data split, preprocessor, validation tests
│   ├── test_modeling_utils.py        # Triage bucket, ROI, reason code tests
│   └── test_rag_pipeline.py          # Chunking, query building, brief generation tests
│
├── .github/
│   └── workflows/
│       ├── deploy.yml            # Auto-deploy to Hugging Face Spaces on push
│       └── tests.yml             # Run pytest on every push
│
├── docs/
│   └── fraud_guidelines/         # Curated insurance fraud guideline documents (4 files)
│       ├── fraud_red_flags.md
│       ├── triage_procedures.md
│       ├── policy_coverage_standards.md
│       └── staged_accident_patterns.md
│
├── notebooks/
│   └── Automotive_Insurance_Claims_Risk_Triage.ipynb
│
└── outputs/
    ├── eda/                      # EDA plots and data quality summaries
    ├── plots/                    # PR curves, risk score distributions, calibration curves
    ├── metrics/                  # Model comparison tables, triage analysis, ROI summary
    ├── models/                   # Serialized base model artifacts (.joblib)
    ├── shap/                     # SHAP importance CSVs, beeswarm plots, reason codes
    ├── improvement/              # Optuna results, stacking ensemble, model_metadata.json
    ├── rag/                      # Persistent ChromaDB index + demo triage briefs
    ├── fairness/                 # Disparate impact charts and CSVs
    ├── monitoring/               # Batch statistics, drift summary, trend charts
    └── temporal/                 # Monthly metrics CSV and performance charts
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/insurance-claims-risk-triage.git
cd insurance-claims-risk-triage
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `fraud_oracle.csv` from [Kaggle — Vehicle Insurance Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) and place it in the project root.

### 4. (Optional) Enable LLM-powered briefs

```bash
# Copy .env.example to .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-key-here
```

### 5. Run the full pipeline (one command)

```bash
python train.py
```

This runs data validation → base modeling → SHAP → Optuna tuning → RAG index, all logged to MLflow.

Or run steps individually:

```bash
python modeling.py           # base models
python shap_explainability.py
python model_improvement.py  # Optuna + CatBoost + stacking
python rag_pipeline.py       # build vector index
```

### 6. Launch the dashboard

```bash
python -m streamlit run app.py
```

### 7. View MLflow experiment runs

```bash
mlflow ui
# Opens at http://localhost:5000
```

### 8. Run tests

```bash
pytest tests/
```

---

## Docker

Run the entire app in a container — no local Python setup needed.

```bash
# Build and start
docker-compose up

# Dashboard at http://localhost:8501
# MLflow UI at http://localhost:5000 (with --profile mlflow)
docker-compose --profile mlflow up
```

### GitHub Codespaces (no admin rights needed)

1. Push repo to GitHub
2. Click green **Code** button → **Codespaces** → **Create codespace on main**
3. Docker is pre-installed in Codespaces — run `docker-compose up` directly

---

## Streamlit Dashboard

7 pages:

Each page has a collapsible **ℹ️** button at the top with a plain-English explanation of every metric and chart — useful for non-technical reviewers and investigators.

| Page | Description |
|------|-------------|
| **📊 Summary Dashboard** | KPI cards (SIU/Manual/Approve counts, PR-AUC, ROI), smoothed PR curve, risk score distribution, SHAP global importance, triage bucket fraud rates, cost-benefit ROI breakdown |
| **📋 Review Queue** | All 1,542 test claims ranked by risk score, color-coded by bucket, filterable by bucket and score range, jump-to-detail button |
| **🔎 Claim Detail** | Per-claim risk score banner, SHAP waterfall chart, styled reason code pills, on-demand RAG triage brief with cited guidelines |
| **⚡ Live Scoring** | Score a brand-new claim in real time — fill a form, get instant risk score, gauge chart, SHAP explanation, and triage brief |
| **⚖️ Fairness Analysis** | Disparate impact analysis across age groups, sex, and marital status using the 80% rule. Groups with fewer than 30 claims excluded from DI calculation |
| **📡 Monitoring** | Test set split into 6 time-ordered batches — tracks fraud rate, avg risk score, and feature drift (KS test) per batch |
| **📅 Temporal Analysis** | Month-by-month PR-AUC, fraud rate, Precision@5%, and avg risk score — highlights best and worst performing months |

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `train.py` | Single entry point — runs full pipeline with MLflow logging |
| `training_config.yaml` | All settings (split ratios, Optuna trials, thresholds, costs) |
| `config.py` | Shared paths and constants |
| `data_validation.py` | Pandera schema checks before any processing |
| `data_preprocessing.py` | Load, EDA, split, preprocess |
| `feature_engineering.py` | 41 engineered features |
| `modeling.py` | LR baseline, XGBoost, LightGBM |
| `model_improvement.py` | Optuna tuning, CatBoost, OOF stacking |
| `shap_explainability.py` | SHAP values and reason codes |
| `rag_pipeline.py` | ChromaDB vector index + triage brief generation |
| `fairness_analysis.py` | Disparate impact analysis across demographic groups |
| `monitoring.py` | Batch drift detection (Evidently + KS test fallback) |
| `temporal_analysis.py` | Month-by-month performance tracking |
| `app.py` | 7-page Streamlit dashboard |

---

## MLflow Experiment Tracking

Every training run is automatically logged:

```bash
python train.py        # logs all steps
mlflow ui              # view at http://localhost:5000
```

What gets logged per run:
- All Optuna hyperparameters
- Val PR-AUC, ROC-AUC, Precision@5%, Recall@5%
- Test PR-AUC, ROC-AUC
- ROI metrics (fraud caught, net benefit, ROI multiplier)
- Best model artifact (`.joblib`)
- Config file used

---

## Tests

```bash
pytest tests/           # run all tests
pytest tests/ -v        # verbose output
pytest tests/ --cov=.   # with coverage report
```

| Test File | What it covers |
|-----------|---------------|
| `test_feature_engineering.py` | 30+ tests: ordinal encoding, risk flags, guideline flags, interactions |
| `test_preprocessing.py` | Data split ratios, no overlap, preprocessor output, validation checks |
| `test_modeling_utils.py` | Precision@K, recall@K, triage bucket counts, ROI calculation, reason codes |
| `test_rag_pipeline.py` | Document chunking, query building, template brief generation |

Tests run automatically on every GitHub push via `.github/workflows/tests.yml`.

---

## Critical Finding: Data Leakage

> **⚠️ PolicyNumber was included as a model feature in the original notebook.**

| Metric | With PolicyNumber (leaked) | Without (corrected) |
|--------|:-:|:-:|
| **Val PR-AUC** | 0.7166 | 0.2518 |
| **Test PR-AUC** | 0.7926 | 0.1951 |

`PolicyNumber` has 15,420 unique values across 15,420 rows — it is a row identifier. Fixed via `config.COLS_TO_DROP`.

---

## Model Performance

### Base Models

| Model | Val PR-AUC | Val ROC-AUC | Precision@5% | Recall@5% |
|-------|:----------:|:-----------:|:------------:|:---------:|
| **XGBoost** | **0.2518** | 0.8346 | 0.2857 | 0.2366 |
| LightGBM | 0.1846 | 0.8097 | 0.2208 | 0.1828 |
| Logistic Regression | 0.1677 | 0.8108 | 0.1818 | 0.1505 |

### Improved Models (Optuna + 41 engineered features)

| Model | Val PR-AUC | Val ROC-AUC | Precision@5% | vs. Base |
|-------|:----------:|:-----------:|:------------:|:--------:|
| **XGBoost (Optuna)** | **0.3223** | 0.8595 | **0.3247** | **+28.0%** |
| OOF Stack (XGB + CatBoost) | 0.3144 | **0.8647** | 0.3117 | +24.9% |
| CatBoost (Optuna) | 0.2879 | 0.8630 | 0.2987 | +14.3% |

### Triage Bucket Performance

| Bucket | Count | Fraud Rate | Enrichment |
|--------|:-----:|:----------:|:----------:|
| **SIU** (top 5%) | 77 | **32.5%** | **5.4x** |
| **Manual Review** (next 15%) | 231 | 16.0% | 2.7x |
| **Approve** (remaining 80%) | 1,234 | 2.5% | 0.4x |

### ROI

| Metric | Value |
|--------|-------|
| Fraud claims caught | 62 |
| Losses prevented | $930,000 |
| Investigation costs | $61,600 |
| **Net benefit** | **$868,400** |
| **ROI** | **15.1x** |

---

## Fairness Analysis

Checks whether the model treats demographic groups equitably using the **80% disparate impact rule** — the standard in US insurance regulation.

Groups analysed:
- **Age groups**: 16-25, 26-35, 36-50, 51-65, 65+
- **Sex**: Male, Female
- **Marital Status**: Single, Married, Widow, Divorced

Metrics per group: flag rate, fraud rate, false positive rate, precision, disparate impact ratio.

**Groups with fewer than 30 claims are excluded** from the disparate impact calculation — small samples produce unreliable ratios (e.g. Widow, 65+).

### Key Finding
The model flags **Males at ~22%** vs **Females at ~12%**, giving a disparate impact ratio of **0.574** — below the 0.80 threshold. This does not necessarily mean the model is biased. It may reflect genuine fraud patterns in the data. However, features like `VehicleCategory` or `AgentType` may be acting as indirect proxies for sex (proxy discrimination), which would warrant further investigation before production deployment.

```bash
python fairness_analysis.py    # standalone run
# or view in dashboard: ⚖️ Fairness Analysis page
```

---

## Model Monitoring

Simulates production monitoring by splitting the test set into 6 time-ordered batches and checking for drift.

- **Data drift**: feature distributions shifting (Evidently or KS test fallback)
- **Prediction drift**: risk score distribution changing
- **Target drift**: actual fraud rate changing per batch

```bash
python monitoring.py    # standalone run
# or view in dashboard: 📡 Monitoring page
```

---

## Temporal Analysis

Evaluates model performance month-by-month using the `Month` column in the dataset. Shows how PR-AUC, fraud rate, and Precision@5% vary across the year.

```bash
python temporal_analysis.py    # standalone run
# or view in dashboard: 📅 Temporal Analysis page
```

---

## SHAP Explainability

### Top 5 Global Feature Importances (CatBoost)

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------|
| 1 | Fault | 0.823 |
| 2 | **Liability_NoPolice** *(guideline-grounded)* | 0.546 |
| 3 | PolicyHolderFault | 0.246 |
| 4 | Fault_NoPolice | 0.244 |
| 5 | BasePolicy | 0.218 |

---

## RAG Pipeline

4 guideline documents, 56 chunks, persistent ChromaDB index.

| Document | Description |
|----------|-------------|
| `fraud_red_flags.md` | Common fraud indicators, vehicle risk factors, demographic patterns |
| `triage_procedures.md` | Three-tier triage framework, SIU protocols, manual review procedures |
| `policy_coverage_standards.md` | Coverage types, deductible patterns, agent oversight |
| `staged_accident_patterns.md` | Staged accident schemes, investigation procedures |

- **LLM mode**: GPT-4o-mini generates natural language briefs with cited passages
- **Template mode**: Structured briefs with no API key required

---

## Key Improvements Over Original Notebook

| Area | Before | After |
|------|--------|-------|
| Data leakage | PolicyNumber in model (PR-AUC 0.79 fake) | Fixed — honest PR-AUC 0.25 → 0.32 |
| Features | 31 raw features | 41 engineered + 7 guideline-grounded flags |
| Hyperparameter tuning | RandomizedSearchCV | Optuna Bayesian TPE |
| Ensemble | None | OOF stacking (XGB + CatBoost → LR meta-learner) |
| Calibration | None | Isotonic calibration |
| Experiment tracking | None | MLflow — every run logged |
| Data validation | None | Pandera schema checks |
| Tests | None | 60+ pytest tests, CI via GitHub Actions |
| Fairness | Not measured | Disparate impact analysis (80% rule) |
| Monitoring | Not measured | Batch drift detection |
| Temporal analysis | Not measured | Month-by-month performance tracking |
| Deployment | Local only | Docker + Hugging Face Spaces (auto-deploy) |
| Pipeline | 4 manual scripts | Single `python train.py` command |
| Dashboard | None | 7-page Streamlit app |

---

## Outputs

| Directory | Contents |
|-----------|----------|
| `outputs/eda/` | Target distribution plot, missing values summary |
| `outputs/plots/` | PR curves, risk score distributions, calibration curves |
| `outputs/metrics/` | Model comparison CSV, triage results, ROI JSON |
| `outputs/models/` | Serialized base models |
| `outputs/shap/` | Global importance, beeswarm plot, reason codes |
| `outputs/improvement/` | Optuna results, best model, model_metadata.json |
| `outputs/rag/` | ChromaDB index, demo triage briefs |
| `outputs/fairness/` | Disparate impact charts and CSVs per demographic group |
| `outputs/monitoring/` | Batch statistics, drift summary, trend charts |
| `outputs/temporal/` | Monthly metrics CSV and performance charts |

---

## Dataset

**Source:** `fraud_oracle.csv` — 15,420 automotive insurance claims, 33 columns
**Target:** `FraudFound_P` (binary: 0 = legitimate, 1 = fraud)
**Fraud rate:** 5.99% (923 of 15,420 claims)
**Split:** 80/10/10 stratified (train: 12,336 / val: 1,542 / test: 1,542)

---

## License

Developed as part of the FSE 570 Data Science Capstone course at Arizona State University.
