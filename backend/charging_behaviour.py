import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "acn",
    "acn_clean_sessions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "behavior"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "user_behavior_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("CHARGING-BEHAVIOR SEGMENTATION")
print("=" * 70)

print("\nLoading ACN charging sessions...")

df = pd.read_csv(INPUT_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "userID",
    "connectionTime",
    "disconnectTime",
    "kWhDelivered"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# DATETIME CONVERSION
# ============================================================

df["connectionTime"] = pd.to_datetime(
    df["connectionTime"],
    errors="coerce"
)

df["disconnectTime"] = pd.to_datetime(
    df["disconnectTime"],
    errors="coerce"
)

df["kWhDelivered"] = pd.to_numeric(
    df["kWhDelivered"],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID USER RECORDS
# ============================================================

df = df.dropna(
    subset=[
        "userID",
        "connectionTime",
        "disconnectTime",
        "kWhDelivered"
    ]
).copy()

print(
    f"\nValid sessions used: {len(df):,}"
)

print(
    f"Unique users: {df['userID'].nunique():,}"
)


# ============================================================
# SESSION DURATION
# ============================================================

df["session_duration_min"] = (
    (
        df["disconnectTime"]
        - df["connectionTime"]
    ).dt.total_seconds()
    / 60
)

# Remove impossible durations
df = df[
    (df["session_duration_min"] > 0)
    & (df["session_duration_min"] <= 24 * 60)
].copy()


# ============================================================
# TIME-OF-DAY FEATURES
# ============================================================

df["arrival_hour"] = (
    df["connectionTime"].dt.hour
    + df["connectionTime"].dt.minute / 60
)

df["day_of_week"] = (
    df["connectionTime"].dt.dayofweek
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ============================================================
# PEAK-HOUR DEFINITION
# ============================================================

# Peak charging period:
# 07:00–10:00 and 17:00–21:00

df["is_peak_hour"] = (
    (
        (df["arrival_hour"] >= 7)
        & (df["arrival_hour"] < 10)
    )
    |
    (
        (df["arrival_hour"] >= 17)
        & (df["arrival_hour"] < 21)
    )
).astype(int)


# ============================================================
# USER-LEVEL AGGREGATION
# ============================================================

user_features = (
    df.groupby("userID")
    .agg(
        session_count=(
            "userID",
            "size"
        ),

        avg_session_duration_min=(
            "session_duration_min",
            "mean"
        ),

        avg_energy_kwh=(
            "kWhDelivered",
            "mean"
        ),

        total_energy_kwh=(
            "kWhDelivered",
            "sum"
        ),

        avg_arrival_hour=(
            "arrival_hour",
            "mean"
        ),

        peak_hour_percentage=(
            "is_peak_hour",
            "mean"
        ),

        weekend_percentage=(
            "is_weekend",
            "mean"
        ),

        first_session=(
            "connectionTime",
            "min"
        ),

        last_session=(
            "connectionTime",
            "max"
        )
    )
    .reset_index()
)


# ============================================================
# CONVERT PERCENTAGES
# ============================================================

user_features["peak_hour_percentage"] *= 100

user_features["weekend_percentage"] *= 100


# ============================================================
# OBSERVATION PERIOD
# ============================================================

user_features["observation_days"] = (
    (
        user_features["last_session"]
        - user_features["first_session"]
    ).dt.total_seconds()
    / 86400
)

# A single-session user has 0 observation days.
# Use 1 day to avoid division by zero.

user_features["observation_days"] = (
    user_features["observation_days"]
    .clip(lower=1)
)


# ============================================================
# CHARGING FREQUENCY
# ============================================================

user_features["sessions_per_30_days"] = (
    user_features["session_count"]
    / user_features["observation_days"]
    * 30
)


# ============================================================
# SAVE
# ============================================================

user_features.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("USER BEHAVIOR DATASET CREATED")
print("=" * 70)

print(
    f"\nOutput file:\n{OUTPUT_PATH}"
)

print(
    f"\nUsers: {len(user_features):,}"
)

print(
    f"Features: {len(user_features.columns)}"
)

print("\nBehavior features:")

for column in user_features.columns:
    print(f" - {column}")

print("\nPreview:")
print(
    user_features.head(10).to_string(
        index=False
    )
)

print("\nDone.")