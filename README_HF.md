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

An end-to-end fraud detection system for automotive insurance claims.

## Features
- XGBoost + CatBoost + OOF Stacking Ensemble (Val PR-AUC: 0.32)
- SHAP explainability with per-claim reason codes
- RAG-powered triage briefs (GPT-4o-mini or template fallback)
- Fairness analysis across age, sex, marital status (80% disparate impact rule)
- Model monitoring and drift detection
- Temporal month-by-month performance analysis
- 7-page Streamlit dashboard with ℹ️ explanations on every page

## Pages
| Page | Description |
|------|-------------|
| 📊 Summary Dashboard | KPIs, PR curve, SHAP importance, triage bucket fraud rates, ROI |
| 📋 Review Queue | 1,542 test claims ranked by risk score, filterable |
| 🔎 Claim Detail | Per-claim SHAP waterfall, reason codes, RAG triage brief |
| ⚡ Live Scoring | Score a new claim in real time |
| ⚖️ Fairness Analysis | Disparate impact across age, sex, marital status |
| 📡 Monitoring | Batch drift detection across time-ordered claim batches |
| 📅 Temporal Analysis | Month-by-month PR-AUC and fraud rate |

## Dataset
Uses the [Vehicle Insurance Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) dataset from Kaggle (not included — download separately).

## Run locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
