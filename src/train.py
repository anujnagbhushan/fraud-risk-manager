"""Day 2 — baseline model. A simple logistic regression, just to have a
number to beat before bringing in XGBoost. Evaluated on val.csv only —
test.csv stays untouched until final evaluation.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, average_precision_score

from config import TRAIN_PATH, VAL_PATH, TARGET_COL, RANDOM_SEED


def load_split(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def main():
    X_train, y_train = load_split(TRAIN_PATH)
    X_val, y_val = load_split(VAL_PATH)

    # logistic regression is sensitive to feature scale, so standardize first
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # accounts for the huge class imbalance
        random_state=RANDOM_SEED,
    )
    model.fit(X_train_scaled, y_train)

    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    val_preds = model.predict(X_val_scaled)

    precision = precision_score(y_val, val_preds)
    recall = recall_score(y_val, val_preds)
    pr_auc = average_precision_score(y_val, val_probs)

    print("=== Baseline: Logistic Regression ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"PR-AUC:    {pr_auc:.3f}")


if __name__ == "__main__":
    main()
    
