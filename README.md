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

| Fairness Analysis | Model Monitoring |
|:-:|:-:|
| ![Fairness Analysis](images/5.png) | ![Model Monitoring](images/6.png) |

| Temporal Analysis |  |
|:-:|:-:|
| ![Temporal Analysis](images/7.png) |  |

| MLflow Experiment Tracking |  |
|:-:|:-:|
| ![MLflow — model_improvement run](images/8.png) |  |

| Live on Hugging Face Spaces |  |
|:-:|:-:|
| ![Hugging Face Spaces deployment](images/9.png) |  |

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
git clone https://github.com/Mriks42/insurance-claims-risk-triage.git
cd insurance-claims-risk-triage
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `fraud_oracle.csv` from [Kaggle — Vehicle Insurance Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) and place it in the project root.

### 4. Enable LLM-powered briefs (GPT-4o-mini)

Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
```

Get a key at [platform.openai.com](https://platform.openai.com/api-keys). GPT-4o-mini costs ~$0.0003 per brief — $5 of credits lasts essentially forever for portfolio use.

**For the live Hugging Face app:** Add `OPENAI_API_KEY` as a secret in your Space settings (Settings → Variables and secrets → New secret).

Without a key the RAG pipeline automatically falls back to the structured template mode — fully functional, no API needed.

### 5. Run the full pipeline (one command)

```bash
python train.py
```

This single command replaces running 4 scripts manually. It executes the full pipeline in order:
1. **Data validation** — Pandera schema checks on `fraud_oracle.csv`
2. **Base modeling** — LR baseline, XGBoost, LightGBM
3. **SHAP analysis** — global importance, per-claim reason codes
4. **Model improvement** — Optuna Bayesian tuning, CatBoost, OOF stacking
5. **RAG pipeline** — builds persistent ChromaDB vector index

Every step is automatically logged to MLflow. Use flags to skip steps:

```bash
python train.py --skip-base --skip-shap --skip-rag   # Optuna tuning only (fastest)
python train.py --skip-improvement --skip-rag         # base models + SHAP only
```

All settings (Optuna trials, split ratios, cost assumptions) live in `training_config.yaml` — edit that file to change any parameter without touching the code.

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

The app is fully containerized using Docker. This means anyone can run it on any machine without installing Python, managing package versions, or dealing with environment conflicts — just one command and it starts.

### Two Dockerfiles

| File | Purpose |
|------|---------|
| `Dockerfile` | Full image — includes all ML training + dashboard dependencies. Used by Hugging Face Spaces for live deployment |
| `Dockerfile.slim` | Lightweight image — dashboard-only dependencies, smaller size. Ideal for local testing when disk space is limited |

### Run with Docker Desktop

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and run:

```bash
# Full stack — dashboard + MLflow server
docker-compose up

# Dashboard opens at http://localhost:8501
```

To also start the MLflow tracking UI:
```bash
docker-compose --profile mlflow up
# Dashboard at http://localhost:8501
# MLflow UI at http://localhost:5000
```

### Run the slim image (lower disk usage)

```bash
# Build
docker build -f Dockerfile.slim -t fraud-triage-slim .

# Run
docker run -p 8501:8501 fraud-triage-slim

# Dashboard at http://localhost:8501
```

### Why Docker matters
In production, your model runs on a server, not your laptop. Docker guarantees the server runs the exact same environment as your development machine — same Python version, same package versions, same configuration. Without Docker, "works on my machine" is a common and costly problem. Every major cloud provider (AWS, GCP, Azure) deploys applications as Docker containers.

### No Docker Desktop? Use GitHub Codespaces
If you don't have admin rights to install Docker Desktop:
1. Go to your GitHub repo
2. Click green **Code** button → **Codespaces** → **Create codespace on main**
3. Docker is pre-installed — run the same commands above directly in the browser terminal

---

## Deployment — Hugging Face Spaces

🚀 **Live URL:** [https://huggingface.co/spaces/Mriks/fraud-triage](https://huggingface.co/spaces/Mriks/fraud-triage)

The app is deployed to Hugging Face Spaces using the `Dockerfile` in this repo. Hugging Face builds and runs the container on their infrastructure — no server management needed.

### How auto-deploy works
Every push to the `main` branch on GitHub automatically triggers a deployment:

```
Push to GitHub → GitHub Action runs → Code pushed to HF Space → HF rebuilds Docker container → App live
```

This is set up via `.github/workflows/deploy.yml`. The workflow uses three GitHub secrets:
- `HF_TOKEN` — Hugging Face API token with write access
- `HF_USERNAME` — your Hugging Face username
- `HF_SPACE_NAME` — name of your Space (`fraud-triage`)

### Sleep behavior
Hugging Face free tier puts Spaces to sleep after 48 hours of inactivity. When someone visits after it's been sleeping, it takes ~30 seconds to wake up. This is normal for free-tier hosting — open the link a minute before any live demo.

---

## CI/CD — GitHub Actions

Two workflows run automatically on every push to `main`:

| Workflow | File | What it does |
|----------|------|-------------|
| **Run Tests** | `.github/workflows/tests.yml` | Runs `pytest tests/` — fails the build if any test breaks |
| **Deploy to HF** | `.github/workflows/deploy.yml` | Pushes latest code to Hugging Face Spaces |

This means every code change is automatically tested and deployed. If a test fails, the deployment is blocked. This is standard CI/CD (Continuous Integration / Continuous Deployment) practice used at every software company.

---

## Streamlit Dashboard

7 pages:

Each page has a collapsible **ℹ️** button at the top with a plain-English explanation of every metric and chart — useful for non-technical reviewers and investigators.

| Page | Description |
|------|-------------|
| **📊 Summary Dashboard** | KPI cards (SIU/Manual/Approve counts, PR-AUC, ROI), smoothed PR curve (test set), model comparison table, risk score distribution, SHAP global importance, triage bucket fraud rates, cost-benefit ROI breakdown, confusion matrix at operational threshold, calibration curve (reliability diagram) |
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

Every training run is automatically logged to MLflow — parameters, metrics, runtime, and model artifacts. This creates a permanent, comparable record of every model version ever trained.

```bash
python train.py        # runs pipeline and logs everything to MLflow
mlflow ui              # open dashboard at http://localhost:5000
```

### What gets logged per run

| Category | What's logged |
|----------|--------------|
| **Parameters** | Config file used, Optuna trial settings, random state |
| **Dataset metrics** | Row count (15,420), fraud rate (5.99%), validation pass/fail |
| **Model metrics** | Val PR-AUC, Val ROC-AUC, Precision@5%, Recall@5% |
| **Test metrics** | Test PR-AUC, Test ROC-AUC (honest, unseen data) |
| **Business metrics** | Fraud caught, losses prevented, investigation costs, net benefit, ROI |
| **Artifacts** | Best model `.joblib` file |
| **Runtime** | Total pipeline duration in seconds |

### Nested runs (parent + child)
The pipeline uses nested MLflow runs to keep things organized:
- **`full_pipeline`** (parent) — logs dataset stats and overall pipeline metrics
- **`model_improvement`** (child) — logs all model-specific metrics from Optuna tuning

![MLflow model_improvement run](images/8.png)

### Why experiment tracking matters
Without MLflow, every training run's results exist only in the terminal — close it and they're gone. With MLflow you can:
- Compare 10 different runs side by side to find the best settings
- Prove to stakeholders which model is in production and why it was chosen
- Reproduce any past run exactly by checking what parameters were used
- Track model performance over time as new data arrives

In regulated industries like insurance, an audit trail of model decisions is often legally required.

---

## Tests

The project has 60+ automated tests covering all core modules. Tests run automatically on every GitHub push via CI.

```bash
pytest tests/           # run all tests
pytest tests/ -v        # verbose output with test names
pytest tests/ --cov=.   # with coverage report
```

### Test files

| Test File | What it covers |
|-----------|---------------|
| `test_feature_engineering.py` | 30+ tests: ordinal encoding correctness, binary risk flags, guideline-grounded flags, interaction features, REPLACED_CATEGORICALS list |
| `test_preprocessing.py` | Data split ratios (80/10/10), no overlap between splits, stratification, preprocessor output shape, no NaN after transform, PolicyNumber dropped |
| `test_modeling_utils.py` | Precision@K, Recall@K, triage bucket counts (5%/15%/80%), ROI calculation, reason code generation, OHE decoding |
| `test_rag_pipeline.py` | Document chunking, chunk uniqueness, query building, template brief generation for all 3 triage buckets, API key fallback |

### Why tests matter
Without tests, every code change risks silently breaking something. For example:
- A change to `engineer_features()` might accidentally drop a guideline flag
- A change to the triage bucket logic might shift the 5%/15% thresholds
- A change to reason code generation might break OHE decoding for columns with underscores

Tests catch these instantly. The GitHub Action blocks deployment if any test fails — so broken code can never reach the live app.

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

| Model | Val PR-AUC | Val ROC-AUC | Precision@5% | Test PR-AUC | vs. Base |
|-------|:----------:|:-----------:|:------------:|:-----------:|:--------:|
| **XGBoost (Optuna)** | **0.3223** | 0.8595 | **0.3247** | **0.2443** | **+28.0%** |
| OOF Stack (XGB + CatBoost) | 0.3144 | **0.8647** | 0.3117 | — | +24.9% |
| CatBoost (Optuna) | 0.2879 | 0.8630 | 0.2987 | — | +14.3% |

> Test PR-AUC is reported only for the selected best model (XGBoost Optuna) — the test set is touched once to avoid selection bias.

### Triage Bucket Performance

| Bucket | Count | Fraud Rate | Enrichment |
|--------|:-----:|:----------:|:----------:|
| **SIU** (top 5%) | 77 | **32.5%** | **5.4x** |
| **Manual Review** (next 15%) | 231 | 16.0% | 2.7x |
| **Approve** (remaining 80%) | 1,234 | 2.5% | 0.4x |

### ROI

| Metric | Value |
|--------|-------|
| Fraud claims caught | 54 |
| Losses prevented | $810,000 |
| Investigation costs | $61,600 |
| **Net benefit** | **$748,400** |
| **ROI** | **13.1x** |

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

4 guideline documents, 56 chunks, persistent ChromaDB index. Powers the AI-generated triage briefs on the Claim Detail and Live Scoring pages.

| Document | Description |
|----------|-------------|
| `fraud_red_flags.md` | Common fraud indicators, vehicle risk factors, demographic patterns |
| `triage_procedures.md` | Three-tier triage framework, SIU protocols, manual review procedures |
| `policy_coverage_standards.md` | Coverage types, deductible patterns, agent oversight |
| `staged_accident_patterns.md` | Staged accident schemes, investigation procedures |

### Brief generation modes

| Mode | How it works | When it's used |
|------|-------------|----------------|
| **🤖 GPT-4o-mini** | Retrieves relevant guideline passages via ChromaDB, sends them + claim data + SHAP reason codes to GPT-4o-mini, returns a natural language brief with citations | When `OPENAI_API_KEY` is set in `.env` or HF Spaces secrets |
| **📋 Template** | Structured brief built from claim data and retrieved passages — no LLM needed | When no API key is present |

Both modes retrieve guideline passages from ChromaDB and cite them in the brief. The template mode is fully professional and suitable for production use.

### Setup
```bash
# Local — add to .env file
OPENAI_API_KEY=sk-your-key-here

# Hugging Face — add as Space secret
# Settings → Variables and secrets → New secret → OPENAI_API_KEY
```

### RAG index behavior
- **First run:** builds and persists the ChromaDB index to `outputs/rag/chroma_db/`
- **Subsequent runs:** reuses the persisted index instantly — no re-embedding
- **Rebuild trigger:** only when guideline documents change

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
| RAG corpus | 3 docs, 36 chunks, rebuilt every run | 4 docs, 56 chunks, persistent index, GPT-4o-mini integrated |
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
