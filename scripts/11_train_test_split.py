from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml"
    / "ev_ml_dataset.csv"
)

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml"
    / "train_base.csv"
)

TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml"
    / "test.csv"
)

# ============================================================
# 1. LOAD FINAL ML DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))

# ============================================================
# 2. SORT BY TIME
# ============================================================

df = df.sort_values("timestamp").reset_index(drop=True)

# ============================================================
# 3. GET UNIQUE TIMESTAMPS
# ============================================================

unique_times = (
    df["timestamp"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

print("\nUnique timestamps:", len(unique_times))

# ============================================================
# 4. 80:20 CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(len(unique_times) * 0.80)

split_time = unique_times.iloc[split_index]

train = df[df["timestamp"] < split_time].copy()
test = df[df["timestamp"] >= split_time].copy()

# ============================================================
# 5. CHECK SIZES
# ============================================================

print("\n" + "=" * 60)
print("80:20 CHRONOLOGICAL SPLIT")
print("=" * 60)

print("\nTRAIN")
print("Rows:", len(train))
print("Columns:", len(train.columns))
print("Start:", train["timestamp"].min())
print("End:", train["timestamp"].max())

print("\nTEST")
print("Rows:", len(test))
print("Columns:", len(test.columns))
print("Start:", test["timestamp"].min())
print("End:", test["timestamp"].max())

print("\nPercentages:")
print(
    "Train:",
    round(len(train) / len(df) * 100, 2),
    "%"
)

print(
    "Test:",
    round(len(test) / len(df) * 100, 2),
    "%"
)

# ============================================================
# 6. CHECK FOR TIME OVERLAP
# ============================================================

overlap = set(train["timestamp"]).intersection(
    set(test["timestamp"])
)

print("\nTimestamp overlap:", len(overlap))

# ============================================================
# 7. SAVE
# ============================================================

train.to_csv(TRAIN_FILE, index=False)
test.to_csv(TEST_FILE, index=False)

print("\n Saved:")
print(TRAIN_FILE)
print(TEST_FILE)