# AI Risk Manager — Fraud-Spike Detector

Razorpay AI Buildathon — Track 02 (AI Risk Manager)

A cost-sensitive fraud detector that flags transactions, explains why
(SHAP) and what would have cleared it (counterfactual), routes them
(auto-clear / auto-escalate / hold-for-review), and logs human review
decisions for an audit trail.

## Setup

1. python -m venv venv && source venv/bin/activate
2. pip install -r requirements.txt
3. Download creditcard.csv from
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   and place it in data/raw/
4. cd src && python data_prep.py

## Results

(fill in after Day 3-4: PR-AUC, precision/recall at chosen threshold,
false-positive cost saved vs a naive 0.5 threshold)
