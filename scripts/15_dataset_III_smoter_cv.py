from pathlib import Path

import numpy as np
import pandas as pd
import resreg
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


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

OUTPUT_FILE = (
    CV_DIR
    / "dataset_III_summary.csv"
)


# ============================================================
# FEATURES USED FOR MODELING
# ============================================================

feature_columns = [
    "energy_kwh",
    "demand_kw",
    "active_sessions",

    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",

    "electricity_price",
    "price_imputed",

    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",

    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_4",
    "demand_lag_12",
    "demand_lag_96",

    "rolling_mean_4",
    "rolling_mean_12",
    "rolling_mean_96"
]

target_column = "target_demand_kw"


# ============================================================
# LOAD 80% TRAINING DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)

train["timestamp"] = pd.to_datetime(
    train["timestamp"]
)

train = train.sort_values(
    "timestamp"
).reset_index(drop=True)

print("=" * 60)
print("DATASET III — STANDARD 10-FOLD + SMOTER")
print("=" * 60)

print("Training rows:", len(train))
print("Training columns:", len(train.columns))


# ============================================================
# SPLIT BY UNIQUE TIMESTAMP
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
# PROCESS EACH FOLD
# ============================================================

for fold, (train_idx, validation_idx) in enumerate(
    kf.split(unique_times),
    start=1
):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    # --------------------------------------------------------
    # Get timestamps belonging to each fold
    # --------------------------------------------------------

    train_times = unique_times.iloc[train_idx]
    validation_times = unique_times.iloc[validation_idx]

    fold_train = train[
        train["timestamp"].isin(train_times)
    ].copy()

    fold_validation = train[
        train["timestamp"].isin(validation_times)
    ].copy()

    print(
        "Original fold-training rows:",
        len(fold_train)
    )

    print(
        "Validation rows:",
        len(fold_validation)
    )

    # --------------------------------------------------------
    # X and y
    # --------------------------------------------------------

    X = fold_train[
        feature_columns
    ].copy()

    y = fold_train[
        target_column
    ].copy()

    # --------------------------------------------------------
    # Rare high-demand threshold
    # calculated ONLY from this fold's training data
    # --------------------------------------------------------

    positive_target = y[y > 0]

    rare_threshold = float(
        positive_target.quantile(0.95)
    )

    relevance = resreg.sigmoid_relevance(
        y.to_numpy(),
        cl=None,
        ch=rare_threshold
    )

    rare_mask = (
        relevance >= 0.5
    )

    original_rare = int(
        rare_mask.sum()
    )

    original_normal = int(
        (~rare_mask).sum()
    )

    print(
        "Rare threshold:",
        round(rare_threshold, 6),
        "kW"
    )

    print(
        "Original normal:",
        original_normal
    )

    print(
        "Original rare:",
        original_rare
    )

    # --------------------------------------------------------
    # Same feature handling used for Dataset I
    # --------------------------------------------------------

    nominal_columns = [
        "weather_code",
        "price_imputed",
        "hour",
        "minute",
        "day_of_week",
        "is_weekend"
    ]

    continuous_columns = [
        col
        for col in feature_columns
        if col not in nominal_columns
    ]

    # --------------------------------------------------------
    # Scale continuous features
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = X.copy()

    X_scaled[
        continuous_columns
    ] = scaler.fit_transform(
        X[continuous_columns]
    )

    X_array = X_scaled.to_numpy()
    y_array = y.to_numpy()

    nominal_indices = [
        feature_columns.index(col)
        for col in nominal_columns
    ]

    # --------------------------------------------------------
    # SMOTER
    # --------------------------------------------------------

    print("Applying SMOTER...")

    X_resampled, y_resampled = resreg.smoter(
        X_array,
        y_array,
        relevance=relevance,
        relevance_threshold=0.5,
        k=5,
        over="balance",
        nominal=np.array(nominal_indices),
        random_state=42
    )

    # --------------------------------------------------------
    # SMOTER size
    # --------------------------------------------------------

    smoter_rows = len(
        X_resampled
    )

    smoter_rare = int(
        (y_resampled >= rare_threshold).sum()
    )

    smoter_zero = int(
        (y_resampled == 0).sum()
    )

    smoter_positive = int(
        (y_resampled > 0).sum()
    )

    print(
        "After SMOTER:",
        smoter_rows
    )

    print(
        "Rare rows after SMOTER:",
        smoter_rare
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "fold": fold,

        "original_train_rows":
            len(fold_train),

        "original_train_columns":
            len(fold_train.columns),

        "smoter_train_rows":
            smoter_rows,

        "smoter_model_columns":
            len(feature_columns) + 1,

        "validation_rows":
            len(fold_validation),

        "validation_columns":
            len(fold_validation.columns),

        "original_normal_rows":
            original_normal,

        "original_rare_rows":
            original_rare,

        "smoter_zero_rows":
            smoter_zero,

        "smoter_positive_rows":
            smoter_positive,

        "smoter_rare_rows":
            smoter_rare,

        "rare_threshold_kw":
            rare_threshold,

        "validation_start":
            fold_validation["timestamp"].min(),

        "validation_end":
            fold_validation["timestamp"].max()
    })


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    results
)

summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY FINAL TABLE
# ============================================================

print("\n" + "=" * 60)
print("DATASET III SUMMARY")
print("=" * 60)

print(
    summary[
        [
            "fold",
            "original_train_rows",
            "smoter_train_rows",
            "validation_rows",
            "original_rare_rows",
            "smoter_rare_rows"
        ]
    ].to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n Dataset III summary created")