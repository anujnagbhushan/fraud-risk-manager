from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

TARGET_COL = "Class"

RANDOM_SEED = 42

COST_FALSE_POSITIVE = 5
COST_FALSE_NEGATIVE = 100
