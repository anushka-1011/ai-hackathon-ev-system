from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml"
    / "train_base.csv"
)

CV_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml"
    / "cv"
)

OUTPUT_FILE = (
    CV_DIR
    / "dataset_IV_summary.csv"
)

CV_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)

train["timestamp"] = pd.to_datetime(
    train["timestamp"]
)

train = train.sort_values(
    "timestamp"
).reset_index(drop=True)

# ============================================================
# UNIQUE TIMESTAMPS
# ============================================================

unique_times = (
    train["timestamp"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

# ============================================================
# STANDARD 10-FOLD
# ============================================================

kf = KFold(
    n_splits=10,
    shuffle=False
)

results = []

# ============================================================
# CREATE DATASET IV SUMMARY
# ============================================================

for fold, (train_idx, validation_idx) in enumerate(
    kf.split(unique_times),
    start=1
):

    train_times = unique_times.iloc[train_idx]
    validation_times = unique_times.iloc[validation_idx]

    fold_train = train[
        train["timestamp"].isin(train_times)
    ]

    fold_validation = train[
        train["timestamp"].isin(validation_times)
    ]

    results.append({
        "fold": fold,

        "train_rows": len(fold_train),
        "train_columns": len(fold_train.columns),

        "validation_rows": len(fold_validation),
        "validation_columns": len(fold_validation.columns),

        "train_timestamps": len(train_times),
        "validation_timestamps": len(validation_times),

        "train_start": fold_train["timestamp"].min(),
        "train_end": fold_train["timestamp"].max(),

        "validation_start": fold_validation["timestamp"].min(),
        "validation_end": fold_validation["timestamp"].max()
    })

# ============================================================
# SAVE
# ============================================================

summary = pd.DataFrame(results)

summary.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

print("=" * 60)
print("DATASET IV — STANDARD 10-FOLD WITHOUT SMOTER")
print("=" * 60)

print(
    summary[
        [
            "fold",
            "train_rows",
            "validation_rows",
            "train_timestamps",
            "validation_timestamps"
        ]
    ].to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n Dataset IV summary created")