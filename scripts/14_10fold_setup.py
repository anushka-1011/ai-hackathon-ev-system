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

CV_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_FILE = CV_DIR / "10fold_summary.csv"


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)

train["timestamp"] = pd.to_datetime(
    train["timestamp"]
)

# Sort timestamps
train = train.sort_values(
    "timestamp"
).reset_index(drop=True)

print("=" * 60)
print("STANDARD 10-FOLD CROSS-VALIDATION")
print("=" * 60)

print("Training rows:", len(train))
print("Training columns:", len(train.columns))


# ============================================================
# UNIQUE TIMESTAMPS
# ============================================================

unique_times = (
    train["timestamp"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

print("Unique timestamps:", len(unique_times))


# ============================================================
# STANDARD 10-FOLD
# ============================================================

kf = KFold(
    n_splits=10,
    shuffle=False
)

results = []


# ============================================================
# CREATE FOLDS
# ============================================================

for fold, (train_idx, val_idx) in enumerate(
    kf.split(unique_times),
    start=1
):

    # Timestamps belonging to each part
    train_times = unique_times.iloc[train_idx]
    val_times = unique_times.iloc[val_idx]

    # All station rows belonging to those timestamps
    fold_train = train[
        train["timestamp"].isin(train_times)
    ]

    fold_val = train[
        train["timestamp"].isin(val_times)
    ]

    results.append({
        "fold": fold,

        "train_rows": len(fold_train),
        "train_columns": len(fold_train.columns),

        "validation_rows": len(fold_val),
        "validation_columns": len(fold_val.columns),

        "train_timestamps": len(train_times),
        "validation_timestamps": len(val_times),

        "train_start": fold_train["timestamp"].min(),
        "train_end": fold_train["timestamp"].max(),

        "validation_start": fold_val["timestamp"].min(),
        "validation_end": fold_val["timestamp"].max()
    })


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(results)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 60)
print("10-FOLD SIZES")
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

print("\n" + "=" * 60)
print("DATASET IV — WITHOUT SMOTER")
print("=" * 60)

print(
    "Each fold uses the 9 training folds directly."
)

print("\n" + "=" * 60)
print("DATASET III — WITH SMOTER")
print("=" * 60)

print(
    "Each fold will apply SMOTER ONLY to its 9-fold "
    "training portion."
)

print(
    "The validation fold remains untouched."
)

print("\nSaved summary to:")
print(SUMMARY_FILE)

print("\n Standard 10-fold setup complete")