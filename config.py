"""
Shared configuration for the Automotive Insurance Claims Risk Triage project.
FSE 570 Capstone - Team Connecticut
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "fraud_oracle.csv")

# Output directories
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
EDA_DIR = os.path.join(OUTPUTS_DIR, "eda")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
SHAP_DIR = os.path.join(OUTPUTS_DIR, "shap")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")

for d in [OUTPUTS_DIR, EDA_DIR, METRICS_DIR, PLOTS_DIR, SHAP_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset settings
# ──────────────────────────────────────────────
TARGET = "FraudFound_P"
RANDOM_STATE = 42

# Columns to DROP before modeling (identifiers / leakage risk).
# PolicyNumber has 15,420 unique values across 15,420 rows — it is
# essentially a row ID. The notebook's own integrity check (Cell 13)
# flagged it as a high-cardinality column. Including it lets the model
# memorize specific policy numbers from the training set, inflating
# metrics artificially while degrading real-world generalization.
COLS_TO_DROP = ["PolicyNumber"]

# ──────────────────────────────────────────────
# Triage settings
# ──────────────────────────────────────────────
PCT_SIU = 0.05          # top 5% → SIU escalation
PCT_MANUAL = 0.15        # next 15% → Manual Review
# remaining 80% → Approve

# ──────────────────────────────────────────────
# SHAP settings
# ──────────────────────────────────────────────
N_REASON_CODES = 5       # top SHAP drivers per claim

# ──────────────────────────────────────────────
# Business cost assumptions (for ROI analysis)
# ──────────────────────────────────────────────
AVG_FRAUD_LOSS = 15_000   # average loss per undetected fraud claim ($)
SIU_COST       = 500      # cost per SIU investigation ($)
MANUAL_COST    = 100      # cost per manual review ($)

# ──────────────────────────────────────────────
# Optuna tuning settings
# ──────────────────────────────────────────────
OPTUNA_N_TRIALS   = 20    # number of Bayesian optimisation trials
OPTUNA_CV_FOLDS   = 3     # stratified k-fold folds used inside each trial
OPTUNA_TIMEOUT    = 300   # seconds wall-clock limit per study (5 min each)

# ──────────────────────────────────────────────
# XGBoost base hyperparameters (from notebook)
# n_estimators is intentionally high — early stopping controls actual count
# ──────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
    early_stopping_rounds=50,
)
