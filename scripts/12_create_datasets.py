from pathlib import Path
import pandas as pd

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

DATASET_II_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml"
    / "dataset_II_train.csv"
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

model_columns = feature_columns + [target_column]

# ============================================================
# CREATE DATASET II
# ============================================================

dataset_II = train[model_columns].copy()

# Save
dataset_II.to_csv(
    DATASET_II_FILE,
    index=False
)

# ============================================================
# CHECK
# ============================================================

print("=" * 60)
print("DATASET II — WITHOUT SMOTER")
print("=" * 60)

print("Rows:", len(dataset_II))
print("Columns:", len(dataset_II.columns))

print("\nColumns:")
print(dataset_II.columns.tolist())

print("\nTarget distribution:")
print(
    "Zero:",
    (dataset_II[target_column] == 0).sum()
)

print(
    "Positive:",
    (dataset_II[target_column] > 0).sum()
)

print("\nSaved to:")
print(DATASET_II_FILE)

print("\nDataset II created")