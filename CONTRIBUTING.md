# Contributing to Automotive Insurance Claims Risk Triage

Thank you for your interest in contributing. This guide explains how to set up the development environment, run the project, and submit changes.

---

## Prerequisites

- Python 3.11+
- Git
- Docker Desktop (optional — for containerized runs)

---

## Setup

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

Download `fraud_oracle.csv` from [Kaggle](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) and place it in the project root.

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key (optional)
```

---

## Running the project

```bash
# Run the full training pipeline
python train.py

# Launch the dashboard
python -m streamlit run app.py

# View MLflow experiment runs
mlflow ui
```

---

## Running tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

All tests must pass before submitting a pull request. The CI pipeline will automatically run tests on every push.

---

## Project structure

| File/Folder | Purpose |
|-------------|---------|
| `app.py` | 7-page Streamlit dashboard |
| `train.py` | Single command pipeline entry point |
| `training_config.yaml` | All training settings |
| `feature_engineering.py` | 41 engineered features |
| `model_improvement.py` | Optuna tuning + CatBoost + OOF stacking |
| `fairness_analysis.py` | Disparate impact analysis |
| `monitoring.py` | Batch drift detection |
| `temporal_analysis.py` | Month-by-month performance |
| `rag_pipeline.py` | ChromaDB RAG + GPT-4o-mini briefs |
| `tests/` | pytest test suite |
| `docs/fraud_guidelines/` | Internal fraud guideline documents |

---

## Code style

- Follow existing code style — functions are documented with docstrings
- Keep functions focused and single-purpose
- Add tests for any new feature or bug fix
- Update `README.md` if you add a new module or change behavior

---

## Submitting changes

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest tests/`
4. Commit: `git commit -m "Brief description of change"`
5. Push: `git push origin feature/your-feature-name`
6. Open a Pull Request on GitHub

---

## Reporting issues

Open a GitHub Issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
