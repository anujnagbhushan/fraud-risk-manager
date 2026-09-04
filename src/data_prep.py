from sklearn.model_selection import train_test_split
import pandas as pd

from config import (
    RAW_DATA_PATH,
    PROCESSED_DIR,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    TARGET_COL,
    RANDOM_SEED,
)


def load_raw() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected dataset at {RAW_DATA_PATH}.\n"
            "Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in data/raw/."
        )
    return pd.read_csv(RAW_DATA_PATH)


def report_class_balance(df: pd.DataFrame, name: str) -> None:
    counts = df[TARGET_COL].value_counts()
    fraud_pct = 100 * counts.get(1, 0) / len(df)
    print(f"[{name}] n={len(df):,}  fraud={counts.get(1, 0):,} "
          f"({fraud_pct:.3f}%)  legit={counts.get(0, 0):,}")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    report_class_balance(df, "full dataset")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[TARGET_COL], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[TARGET_COL], random_state=RANDOM_SEED
    )

    report_class_balance(train_df, "train")
    report_class_balance(val_df, "val")
    report_class_balance(test_df, "test (held out - do not touch until final eval)")

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"\nSaved splits to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()

