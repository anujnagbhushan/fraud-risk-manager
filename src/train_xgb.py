"""Day 3-4 — the real model. XGBoost, evaluated with a cost-sensitive
threshold instead of the default 0.5. Saves the trained model and the
chosen threshold so later scripts (explain.py, respond.py, app.py) can
reuse them.
"""

import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from config import (
    TRAIN_PATH,
    VAL_PATH,
    TARGET_COL,
    RANDOM_SEED,
    MODELS_DIR,
    FIGURES_DIR,
    COST_FALSE_POSITIVE,
    COST_FALSE_NEGATIVE,
)


def load_split(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def find_best_threshold(y_true, probs):
    """Sweep thresholds 0.01-0.99, compute total cost at each, return the
    threshold that minimizes cost — plus the full curve for plotting."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE
        costs.append(cost)

    costs = np.array(costs)
    best_idx = costs.argmin()
    return thresholds[best_idx], thresholds, costs


def main():
    X_train, y_train = load_split(TRAIN_PATH)
    X_val, y_val = load_split(VAL_PATH)

    # scale_pos_weight tells XGBoost how imbalanced the classes are
    # (ratio of legit:fraud in the training set)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]

    # default 0.5 threshold, for comparison
    default_preds = (val_probs >= 0.5).astype(int)
    print("=== XGBoost @ default threshold (0.5) ===")
    print(f"Precision: {precision_score(y_val, default_preds):.3f}")
    print(f"Recall:    {recall_score(y_val, default_preds):.3f}")
    print(f"PR-AUC:    {average_precision_score(y_val, val_probs):.3f}")

    # cost-sensitive threshold
    best_threshold, thresholds, costs = find_best_threshold(y_val, val_probs)
    best_preds = (val_probs >= best_threshold).astype(int)

    print(f"\n=== XGBoost @ cost-optimal threshold ({best_threshold:.2f}) ===")
    print(f"Precision: {precision_score(y_val, best_preds):.3f}")
    print(f"Recall:    {recall_score(y_val, best_preds):.3f}")
    print(f"Total cost at this threshold: {costs.min():,.0f}")
    print(f"Total cost at 0.5 threshold:  "
          f"{costs[np.argmin(np.abs(thresholds - 0.5))]:,.0f}")

    # save the cost curve as a figure for your README/video
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, costs)
    plt.axvline(best_threshold, color="red", linestyle="--",
                label=f"chosen threshold = {best_threshold:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Total cost (FP × ₹5 + FN × ₹100)")
    plt.title("Cost vs threshold")
    plt.legend()
    plt.savefig(FIGURES_DIR / "cost_curve.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved cost curve to {FIGURES_DIR / 'cost_curve.png'}")

    # save model + chosen threshold so later scripts can reuse them
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "xgb_fraud_model.joblib")
    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": float(best_threshold)}, f)
    print(f"Saved model + threshold to {MODELS_DIR}")


if __name__ == "__main__":
    main()
    