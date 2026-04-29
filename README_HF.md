---
title: Insurance Fraud Triage
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
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
- Fairness analysis across age, sex, marital status
- Model monitoring and drift detection
- Temporal performance analysis

## Dataset
Uses the [Vehicle Insurance Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) dataset from Kaggle.
