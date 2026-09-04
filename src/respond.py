"""Day 6 — auto-responder. Takes a transaction's score + SHAP reason +
counterfactual, and routes it: auto-clear / auto-escalate / hold-for-review.
Also logs a human reviewer's verdict for an audit trail (same pattern as
the cricket project's "coach verification" step).
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap

from config import VAL_PATH, TARGET_COL, MODELS_DIR, PROCESSED_DIR

REVIEW_LOG_PATH = PROCESSED_DIR / "review_log.csv"

# two thresholds define three zones:
#   prob < CLEAR_BELOW              -> auto-clear (very low risk)
#   CLEAR_BELOW <= prob < ESCALATE_ABOVE -> hold-for-review (uncertain)
#   prob >= ESCALATE_ABOVE          -> auto-escalate (very high risk)
ESCALATE_ABOVE = 0.80


def load_val():
    df = pd.read_csv(VAL_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def route(prob, clear_below):
    if prob >= ESCALATE_ABOVE:
        return "AUTO-ESCALATE"
    elif prob >= clear_below:
        return "HOLD-FOR-REVIEW"
    else:
        return "AUTO-CLEAR"


def explain_transaction(model, explainer, X_val, idx):
    row = X_val.iloc[idx]
    row_shap = explainer.shap_values(row.to_frame().T)[0]
    top_feature = X_val.columns[np.argmax(row_shap)]
    return top_feature, row_shap.max()


def main():
    model = joblib.load(MODELS_DIR / "xgb_fraud_model.joblib")
    with open(MODELS_DIR / "threshold.json") as f:
        threshold = json.load(f)["threshold"]

    X_val, y_val = load_val()
    probs = model.predict_proba(X_val)[:, 1]
    explainer = shap.TreeExplainer(model)

    # look at the 5 highest-confidence transactions as a demo batch
        # pick a mixed sample: one clearly escalated, some genuinely borderline
    # (hold-for-review), and one clearly cleared — shows all 3 routing zones
    sorted_idx = np.argsort(probs)[::-1]
    escalate_example = sorted_idx[:1]                      # most confident
    hold_zone = np.where((probs >= threshold) & (probs < ESCALATE_ABOVE))[0]
    hold_examples = hold_zone[:3] if len(hold_zone) >= 3 else hold_zone
    clear_example = np.where(probs < threshold)[0][:1]     # a clean case

    top5_idx = np.concatenate([escalate_example, hold_examples, clear_example])

    log_rows = []
    print(f"Routing top 5 highest-risk transactions "
          f"(hold-for-review threshold: {threshold:.2f}, "
          f"auto-escalate threshold: {ESCALATE_ABOVE:.2f})\n")

    for idx in top5_idx:
        prob = probs[idx]
        decision = route(prob, threshold)
        top_feature, contribution = explain_transaction(model, explainer, X_val, idx)
        actual_label = "FRAUD" if y_val.iloc[idx] == 1 else "legit"

        print(f"Transaction #{idx} — confidence {prob:.2f} — {decision}")
        print(f"  Top reason: '{top_feature}' (SHAP: +{contribution:.2f})")
        print(f"  Ground truth: {actual_label}")
        print()

        log_rows.append({
            "transaction_id": idx,
            "confidence": round(float(prob), 4),
            "decision": decision,
            "top_reason_feature": top_feature,
            "shap_contribution": round(float(contribution), 4),
            "ground_truth": actual_label,
            "reviewer_verdict": "",   # filled in by a human, see below
        })

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(REVIEW_LOG_PATH, index=False)
    print(f"Saved review log to {REVIEW_LOG_PATH}")
    print("\nOpen that CSV and fill in 'reviewer_verdict' for each row "
          "(agree / disagree) — this is your audit trail.")


if __name__ == "__main__":
    main()
