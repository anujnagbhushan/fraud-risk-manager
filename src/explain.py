"""Day 5 — explainability. For a handful of flagged transactions, show
(1) SHAP reasons: which features pushed this toward "fraud", and
(2) a counterfactual: how much would the top feature need to change to
    flip this transaction back under the threshold.

This is the audit-trail layer — it turns "the model flagged this" into
"the model flagged this, here's why, and here's what would have cleared it."
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from config import VAL_PATH, TARGET_COL, MODELS_DIR, FIGURES_DIR


def load_val():
    df = pd.read_csv(VAL_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def counterfactual_for_row(model, row, top_feature, threshold, direction):
    """Nudge `top_feature` on this single row, step by step, until the
    model's predicted probability drops below `threshold`. Returns how
    much the feature had to change, or None if it never flips within
    a reasonable search range."""
    row = row.copy()
    original_value = row[top_feature]
    step = abs(original_value) * 0.05 + 0.01  # 5% steps (with a floor)

    for i in range(1, 200):
        row[top_feature] = original_value - direction * step * i
        prob = model.predict_proba(row.to_frame().T)[0, 1]
        if prob < threshold:
            change = row[top_feature] - original_value
            return change, prob
    return None, None


def main():
    model = joblib.load(MODELS_DIR / "xgb_fraud_model.joblib")
    with open(MODELS_DIR / "threshold.json") as f:
        threshold = json.load(f)["threshold"]

    X_val, y_val = load_val()
    probs = model.predict_proba(X_val)[:, 1]
    flagged_idx = np.where(probs >= threshold)[0]

    print(f"Threshold in use: {threshold:.2f}")
    print(f"{len(flagged_idx)} transactions flagged out of {len(X_val)}\n")

    # SHAP: explains *why* the model made each prediction, feature by feature
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # save a global summary plot (which features matter most, overall)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_val, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP summary to {FIGURES_DIR / 'shap_summary.png'}\n")

    # walk through 3 example flagged transactions in detail
    sample_idx = flagged_idx[:3]
    for i in sample_idx:
        row = X_val.iloc[i]
        row_shap = shap_values[i]

        # which feature pushed this toward fraud the most?
        top_feature_pos = X_val.columns[np.argmax(row_shap)]
        top_contribution = row_shap.max()

        print(f"--- Transaction #{i} (model confidence: {probs[i]:.2f}) ---")
        print(f"Top reason flagged: '{top_feature_pos}' "
              f"(SHAP contribution: +{top_contribution:.3f})")

        # counterfactual: how much would this feature need to drop to clear it?
        change, new_prob = counterfactual_for_row(
            model, row, top_feature_pos, threshold, direction=1
        )
        if change is not None:
            print(f"Counterfactual: if '{top_feature_pos}' had been "
                  f"{change:+.2f} different (new confidence: {new_prob:.2f}), "
                  f"this would NOT have been flagged.")
        else:
            print("Counterfactual: no single-feature change found "
                  "within search range that clears this transaction.")
        print()


if __name__ == "__main__":
    main()
