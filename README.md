# Automotive Insurance Claims Risk Triage

> **FSE 570 Capstone Project** — Arizona State University
> **Team Connecticut:** Mriganko Chowdhury, Aryan Gonsalves, Ashish Raj Singh, Deborah Sheryl Veluvalli, Kshama Girish

An end-to-end fraud triage system for automotive insurance claims that:
1. **Predicts** a fraud risk score for each claim (XGBoost, CatBoost, OOF Stacking Ensemble)
2. **Explains** the key risk drivers using SHAP values and human-readable reason codes
3. **Generates** a policy-grounded triage brief using Retrieval-Augmented Generation (RAG) with citations to internal fraud guidelines
4. **Routes** each claim to the appropriate triage bucket: **SIU Escalation**, **Manual Review**, or **Approve**
5. **Visualises** everything in an interactive Streamlit dashboard with live claim scoring

---

## Table of Contents

- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Module Reference](#module-reference)
- [Critical Finding: Data Leakage](#critical-finding-data-leakage)
- [Model Performance](#model-performance)
- [SHAP Explainability](#shap-explainability)
- [RAG Pipeline](#rag-pipeline)
- [Key Improvements](#key-improvements-over-original-notebook)
- [Outputs](#outputs)

---

## Project Structure

```
FSE 570 Capstone Project/
│
├── app.py                        # Streamlit dashboard (4 pages)
├── config.py                     # Shared constants, paths, hyperparameters, Optuna settings
├── data_preprocessing.py         # Data loading, EDA, splitting, preprocessing pipeline
├── feature_engineering.py        # 41 engineered features (ordinal, interaction, risk flags, guideline-grounded)
├── modeling.py                   # LR baseline, XGBoost, LightGBM — with calibration, CV, ROI
├── model_improvement.py          # Optuna tuning + CatBoost + OOF stacking ensemble
├── shap_explainability.py        # SHAP values, global importance, per-claim reason codes
├── rag_pipeline.py               # RAG: persistent ChromaDB index + triage brief generation
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for OpenAI API key
│
├── docs/
│   └── fraud_guidelines/         # Curated insurance fraud guideline documents (4 files, 56 chunks)
│       ├── fraud_red_flags.md
│       ├── triage_procedures.md
│       ├── policy_coverage_standards.md
│       └── staged_accident_patterns.md
│
├── notebooks/
│   └── Automotive_Insurance_Claims_Risk_Triage.ipynb   # Original exploration notebook
│
└── outputs/
    ├── eda/                      # EDA plots and data quality summaries
    ├── plots/                    # PR curves, risk score distributions, calibration curves
    ├── metrics/                  # Model comparison tables, triage analysis, ROI summary
    ├── models/                   # Serialized base model artifacts (.joblib)
    ├── shap/                     # SHAP importance CSVs, beeswarm plots, reason codes
    ├── improvement/              # Optuna results, stacking ensemble, CatBoost SHAP, model_metadata.json
    └── rag/                      # Persistent ChromaDB index + demo triage briefs
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

The dataset is not included in this repository. Download `fraud_oracle.csv` from [Kaggle — Vehicle Insurance Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) and place it in the project root:

```
insurance-claims-risk-triage/
└── fraud_oracle.csv   ← place here
```

### 4. (Optional) Enable LLM-powered briefs

Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
```

Without a key the RAG pipeline uses a structured template fallback that produces professional, well-formatted briefs. The API key can also be entered directly in the Streamlit sidebar.

### 5. Run the full pipeline

```bash
# Phase 1-3: Base preprocessing + LR/XGBoost/LightGBM + calibration + ROI
python modeling.py

# Phase 4: Explainability (SHAP + reason codes)
python shap_explainability.py

# Phase 4+: Improved models (Optuna tuning + CatBoost + OOF stacking)
python model_improvement.py

# Phase 5: RAG pipeline (builds persistent vector index on first run, reuses on subsequent runs)
python rag_pipeline.py

# Phase 6: Launch the Streamlit dashboard
python -m streamlit run app.py
```

> **Note:** The RAG pipeline only rebuilds the vector index when guideline documents change. On subsequent runs it reuses the persisted index instantly.

---

## Streamlit Dashboard

Launch with:

```bash
python -m streamlit run app.py
```

The dashboard runs on the **test set** (1,542 completely unseen claims) for all live charts and the review queue. Validation set metrics are shown only in the model comparison table (where they were used for model selection).

### Pages

| Page | Description |
|------|-------------|
| **📊 Summary Dashboard** | KPI cards, smoothed PR curve, risk score distribution, SHAP importance, triage bucket fraud rates, cost-benefit ROI |
| **📋 Review Queue** | All 1,542 test claims ranked by risk score, color-coded by bucket, filterable by bucket and score range |
| **🔎 Claim Detail** | Per-claim risk score banner, SHAP waterfall chart, styled reason code pills, on-demand RAG triage brief |
| **⚡ Live Scoring** | Score a brand-new claim in real time — fill in a form, get instant risk score, SHAP explanation, and triage brief |

### Data Used Per Page

| Component | Data Source |
|-----------|-------------|
| PR curve, queue, claim detail, SHAP | Test set (unseen, honest) |
| SHAP global importance | Test set (computed live at startup) |
| ROI numbers | Test set (computed live from test predictions) |
| Model comparison table | Validation set (used for model selection) |
| Live Scoring | Brand-new claim (not from any split) |

---

## Module Reference

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `app.py` | Streamlit dashboard | 4-page interactive app with live scoring |
| `config.py` | Shared settings | Paths, `TARGET`, `RANDOM_STATE`, `COLS_TO_DROP`, XGB params, Optuna settings, ROI constants |
| `data_preprocessing.py` | Data pipeline | `load_data()`, `run_eda()`, `split_data()`, `build_preprocessor()`, `get_processed_data()` |
| `feature_engineering.py` | Feature creation | `engineer_features()` — 41 features: ordinal encoding, interactions, risk flags, guideline-grounded combos |
| `modeling.py` | Base model training | `train_xgboost()`, `train_lightgbm()`, `evaluate_model()`, `calibrate_model()`, `triage_analysis()` |
| `model_improvement.py` | Model optimization | Optuna XGBoost + CatBoost tuning, CatBoost SHAP, permutation importance, OOF stacking, model_metadata.json |
| `shap_explainability.py` | Explainability | `compute_shap_values()`, `global_feature_importance()`, `make_reason_code()` |
| `rag_pipeline.py` | RAG briefs | `RAGPipeline` class, persistent ChromaDB index, `generate_brief()` with LLM/template fallback |

---

## Critical Finding: Data Leakage

> **⚠️ PolicyNumber was included as a model feature in the original notebook.**

### The Problem
- `PolicyNumber` has **15,420 unique values across 15,420 rows** — it is a row identifier
- XGBoost memorized policy numbers from the training set, inflating metrics artificially

### The Evidence

| Metric | With PolicyNumber (leaked) | Without (corrected) |
|--------|:-:|:-:|
| **Val PR-AUC** | 0.7166 | 0.2518 |
| **Test PR-AUC** | 0.7926 | 0.1951 |

### Resolution
- `PolicyNumber` is permanently excluded via `config.COLS_TO_DROP`
- Preserved as a display-only lookup column in the Streamlit app

---

## Model Performance

### Base Model Comparison (no feature engineering)

| Model | Val PR-AUC | Val ROC-AUC | Precision@5% | Recall@5% |
|-------|:----------:|:-----------:|:------------:|:---------:|
| **XGBoost** | **0.2518** | 0.8346 | 0.2857 | 0.2366 |
| LightGBM | 0.1846 | 0.8097 | 0.2208 | 0.1828 |
| Logistic Regression | 0.1677 | 0.8108 | 0.1818 | 0.1505 |

### Improved Models (Optuna tuning + 41 engineered features)

| Model | Val PR-AUC | Val ROC-AUC | Precision@5% | Recall@5% | vs. Base |
|-------|:----------:|:-----------:|:------------:|:---------:|:--------:|
| **XGBoost (Optuna)** | **0.3223** | 0.8595 | **0.3247** | **0.2688** | **+28.0%** |
| OOF Stack (XGB + CatBoost) | 0.3144 | **0.8647** | 0.3117 | 0.2581 | +24.9% |
| CatBoost (Optuna) | 0.2879 | 0.8630 | 0.2987 | 0.2473 | +14.3% |

### Final Test Set Performance (Best Model — XGBoost Optuna)

| Metric | Value |
|--------|-------|
| **Test PR-AUC** | **0.2443** |
| Test ROC-AUC | 0.8331 |
| Precision@5% | 0.2727 |
| Recall@5% | 0.2283 |

> 5-fold CV PR-AUC: **0.2164 ± 0.0248** (statistically validated)

### Triage Bucket Performance

| Bucket | Count | Fraud Rate | Enrichment vs. Base (6.0%) |
|--------|:-----:|:----------:|:-:|
| **SIU** (top 5%) | 77 | **32.5%** | **5.4x** |
| **Manual Review** (next 15%) | 231 | 16.0% | 2.7x |
| **Approve** (remaining 80%) | 1,234 | 2.5% | 0.4x |

### Cost-Benefit ROI

| Metric | Value |
|--------|-------|
| Fraud claims caught (SIU + Manual) | 62 |
| Estimated fraud losses prevented | $930,000 |
| Investigation costs | $61,600 |
| **Net benefit** | **$868,400** |
| **ROI** | **15.1x** |

---

## SHAP Explainability

### Top 10 Global Feature Importances — CatBoost (Optuna, native SHAP)

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------|
| 1 | Fault | 0.823 |
| 2 | **Liability_NoPolice** *(guideline-grounded feature)* | 0.546 |
| 3 | PolicyHolderFault | 0.246 |
| 4 | Fault_NoPolice | 0.244 |
| 5 | BasePolicy | 0.218 |

### Example Reason Codes (SIU Claim)

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

### Guideline Corpus (4 documents, 56 chunks)

| Document | Description |
|----------|-------------|
| `fraud_red_flags.md` | Common fraud indicators, vehicle risk factors, demographic patterns |
| `triage_procedures.md` | Three-tier triage framework, SIU protocols, manual review procedures |
| `policy_coverage_standards.md` | Coverage types, deductible patterns, agent oversight, claim validation |
| `staged_accident_patterns.md` | Staged accident schemes (swoop & squat, phantom vehicle, paper accidents), investigation procedures |

### RAG Index Behavior
- **First run:** builds and persists the ChromaDB index to `outputs/rag/chroma_db/`
- **Subsequent runs:** reuses the persisted index instantly — no re-embedding
- **Rebuild trigger:** only when guideline documents are added, edited, or deleted

### Brief Generation Modes
- **LLM mode** (`OPENAI_API_KEY` set): GPT-4o-mini generates natural language briefs with cited passages
- **Template mode** (default): Structured briefs with risk assessment, claim details, cited guidelines, and investigation steps

---

## Key Improvements Over Original Notebook

| Area | Before | After |
|------|--------|-------|
| Data leakage | PolicyNumber in model (PR-AUC 0.79 fake) | Fixed — honest PR-AUC 0.25 → 0.32 |
| Features | 31 raw features | 41 engineered + 7 guideline-grounded flags |
| Hyperparameter tuning | RandomizedSearchCV (random) | Optuna Bayesian TPE (intelligent) |
| CatBoost | Default params, no SHAP | Optuna-tuned + native SHAP |
| Ensemble | None | OOF stacking (XGB + CatBoost → LR meta-learner) |
| Calibration | None | Isotonic calibration + reliability diagram |
| Evaluation threshold | Fixed 0.5 (wrong for imbalanced data) | Operational threshold (top 5% cutoff) |
| Metrics | PR-AUC only | PR-AUC + ROC-AUC + CV mean ± std |
| Business impact | Not measured | $868k net benefit, 15.1x ROI |
| RAG corpus | 3 docs, 36 chunks, rebuilt every run | 4 docs, 56 chunks, persistent index |
| SHAP reason codes | OHE decoding bug | Fixed with longest-prefix matching |
| Dashboard | None | 4-page Streamlit app with live scoring |

---

## Outputs

| Directory | Contents |
|-----------|----------|
| `outputs/eda/` | Target distribution plot, missing values summary |
| `outputs/plots/` | PR curves, risk score distributions, calibration curves |
| `outputs/metrics/` | Model comparison CSV, precision@K sweep, triage results, ROI JSON |
| `outputs/models/` | Serialized base models (`.joblib`) |
| `outputs/shap/` | Global importance CSV/PNG, beeswarm plot, claim explanations, reason codes |
| `outputs/improvement/` | Optuna trial CSVs, best params JSONs, CatBoost SHAP, stacking calibration, `model_metadata.json` |
| `outputs/rag/` | Persistent ChromaDB index, demo triage briefs (JSON + text) |

---

## Dataset

**Source:** `fraud_oracle.csv` — 15,420 automotive insurance claims with 33 columns
**Target:** `FraudFound_P` (binary: 0 = legitimate, 1 = fraud)
**Fraud rate:** 5.99% (923 of 15,420 claims)
**Split:** 80/10/10 stratified (train: 12,336 / val: 1,542 / test: 1,542)

---

## License

This project was developed as part of the FSE 570 Data Science Capstone course at Arizona State University.
