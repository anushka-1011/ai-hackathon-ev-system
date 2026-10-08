from pathlib import Path

import numpy as np
import pandas as pd
import resreg

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

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml"
    / "dataset_I_SMOTER_train.csv"
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)


# ============================================================
# MODEL FEATURES
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
# PREPARE X AND y
# ============================================================

X = train[feature_columns].copy()
y = train[target_column].copy()


# ============================================================
# DEFINE RARE HIGH-DEMAND REGION
# ============================================================

positive_target = y[y > 0]

rare_threshold = float(
    positive_target.quantile(0.95)
)

print("=" * 60)
print("SMOTER SETUP")
print("=" * 60)

print("Rare-demand threshold:",
      round(rare_threshold, 6), "kW")


# ============================================================
# COMPUTE RELEVANCE
# ============================================================

relevance = resreg.sigmoid_relevance(
    y.to_numpy(),
    cl=None,
    ch=rare_threshold
)

relevance_threshold = 0.5

rare_mask = relevance >= relevance_threshold

rare_count = rare_mask.sum()
normal_count = (~rare_mask).sum()

print("Normal samples:", normal_count)
print("Rare samples:", rare_count)

print(
    "Rare percentage:",
    round(rare_count / len(y) * 100, 3),
    "%"
)


# ============================================================
# CONTINUOUS / NOMINAL FEATURES
# ============================================================

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


# ============================================================
# SCALE CONTINUOUS FEATURES FOR DISTANCE CALCULATION
# ============================================================

scaler = StandardScaler()

X_scaled = X.copy()

X_scaled[continuous_columns] = scaler.fit_transform(
    X[continuous_columns]
)

X_array = X_scaled.to_numpy()
y_array = y.to_numpy()


# Column indices of nominal features
nominal_indices = [
    feature_columns.index(col)
    for col in nominal_columns
]


# ============================================================
# EXPECTED DATASET SIZE WITH BALANCE
# ============================================================

expected_domain_size = (
    normal_count + rare_count
) // 2

expected_total = (
    expected_domain_size * 2
)

print("\nExpected SMOTER normal samples:",
      expected_domain_size)

print("Expected SMOTER rare samples:",
      expected_domain_size)

print("Expected total rows:",
      expected_total)


# ============================================================
# APPLY SMOTER
# ============================================================

print("\nRunning SMOTER...")
print("This may take some time because the rare-demand region")
print("is being oversampled using nearest-neighbour interpolation.")

X_resampled, y_resampled = resreg.smoter(
    X_array,
    y_array,
    relevance=relevance,
    relevance_threshold=relevance_threshold,
    k=5,
    over="balance",
    nominal=np.array(nominal_indices),
    random_state=42
)


# ============================================================
# CONVERT BACK TO DATAFRAME
# ============================================================

X_resampled = pd.DataFrame(
    X_resampled,
    columns=feature_columns
)

# Undo scaling
X_resampled[continuous_columns] = scaler.inverse_transform(
    X_resampled[continuous_columns]
)

dataset_I = X_resampled.copy()

dataset_I[target_column] = y_resampled


# ============================================================
# CLEAN NUMERIC PRECISION
# ============================================================

# Keep nominal columns as integer values
for column in nominal_columns:
    dataset_I[column] = (
        dataset_I[column]
        .round()
        .astype(int)
    )


# ============================================================
# SAVE
# ============================================================

dataset_I.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 60)
print("DATASET I — SMOTER")
print("=" * 60)

print("Rows:", len(dataset_I))
print("Columns:", len(dataset_I.columns))

print("\nZero demand:",
      (dataset_I[target_column] == 0).sum())

print("Positive demand:",
      (dataset_I[target_column] > 0).sum())

print(
    "Rare/high-demand:",
    (dataset_I[target_column] >= rare_threshold).sum()
)

print("\nTarget statistics:")
print(
    dataset_I[target_column].describe()
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nDataset I created")