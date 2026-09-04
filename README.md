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
# 🛡️ AI Risk Manager — Fraud Detection

### Razorpay AI Buildathon — Track 02: AI Risk Manager

A cost-sensitive, transaction-level fraud-risk management system that goes beyond a simple fraud prediction.

The system:

- Predicts fraud risk using XGBoost
- Uses a cost-sensitive decision threshold instead of blindly using 0.5
- Explains individual predictions using SHAP
- Generates counterfactual explanations showing what change could have moved a transaction below the risk threshold
- Routes transactions into `AUTO-CLEAR`, `HOLD-FOR-REVIEW`, or `AUTO-ESCALATE`
- Allows a human reviewer to verify the decision
- Logs review decisions for an audit trail

---

## 🎥 5-Minute Demo

[Watch the 5-minute project demo](https://youtu.be/LVoyHEVfVZU)

---

## 🚨 Problem

Fraud detection is not only a classification problem.

A fraud-risk system has to answer several operational questions:

1. How risky is this transaction?
2. When should the system flag it?
3. What caused the model to flag it?
4. What change could have prevented the flag?
5. Should the transaction be cleared automatically or sent to a human?
6. Can the decision be reviewed later?

This project builds a complete risk-management pipeline around the fraud prediction model.

---

## 🧠 Solution

The system follows this pipeline:

```text
Transaction
     │
     ▼
XGBoost Fraud Model
     │
     ▼
Fraud Risk Probability
     │
     ▼
Cost-Sensitive Decision Policy
     │
     ├───────────────┬──────────────────┐
     ▼               ▼                  ▼
AUTO-CLEAR     HOLD-FOR-REVIEW    AUTO-ESCALATE
                     │
                     ▼
               Human Review
                     │
                     ▼
                Audit Log


       ┌─────────────────────────┐
       │      Explainability     │
       │                         │
       │  SHAP → Why?            │
       │  Counterfactual → What  │
       │  would have changed?    │
       └─────────────────────────┘


