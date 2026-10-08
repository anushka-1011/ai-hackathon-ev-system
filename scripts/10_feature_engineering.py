import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD MASTER DATASET
# ============================================================

df = pd.read_csv("ev_master_dataset.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ============================================================
# 2. SORT BY STATION + TIME
# ============================================================

df = df.sort_values(
    ["site_name", "stationID", "timestamp"]
).reset_index(drop=True)

# ============================================================
# 3. TIME FEATURES
# ============================================================

df["hour"] = df["timestamp"].dt.hour

df["minute"] = df["timestamp"].dt.minute

df["day_of_week"] = df["timestamp"].dt.dayofweek

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# Hour expressed cyclically
decimal_hour = (
    df["hour"] +
    df["minute"] / 60
)

df["hour_sin"] = np.sin(
    2 * np.pi * decimal_hour / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * decimal_hour / 24
)

# Day-of-week cyclic feature
df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

# ============================================================
# 4. DEMAND LAG FEATURES
# ============================================================

group = df.groupby(
    ["site_name", "stationID"]
)["demand_kw"]

# Previous 15 minutes
df["demand_lag_1"] = group.shift(1)

# Previous 30 minutes
df["demand_lag_2"] = group.shift(2)

# Previous 1 hour
df["demand_lag_4"] = group.shift(4)

# Previous 3 hours
df["demand_lag_12"] = group.shift(12)

# Previous 24 hours
df["demand_lag_96"] = group.shift(96)

## ============================================================
# 5. ROLLING DEMAND FEATURES
# ============================================================

df["rolling_mean_4"] = (
    df.groupby(["site_name", "stationID"])["demand_kw"]
    .transform(lambda s: s.shift(1).rolling(4).mean())
)

df["rolling_mean_12"] = (
    df.groupby(["site_name", "stationID"])["demand_kw"]
    .transform(lambda s: s.shift(1).rolling(12).mean())
)

df["rolling_mean_96"] = (
    df.groupby(["site_name", "stationID"])["demand_kw"]
    .transform(lambda s: s.shift(1).rolling(96).mean())
)

# ============================================================
# 6. CREATE NEXT-15-MINUTE TARGET
# ============================================================

df["target_demand_kw"] = (
    group.shift(-1)
)

# ============================================================
# 7. REMOVE ROWS WITHOUT ENOUGH HISTORY / TARGET
# ============================================================

before = len(df)

df = df.dropna(
    subset=[
        "demand_lag_1",
        "demand_lag_2",
        "demand_lag_4",
        "demand_lag_12",
        "demand_lag_96",
        "rolling_mean_4",
        "rolling_mean_12",
        "rolling_mean_96",
        "target_demand_kw"
    ]
).reset_index(drop=True)

after = len(df)

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

print("Rows before:", before)
print("Rows after:", after)
print("Rows removed:", before - after)

print("\nNumber of columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(
    df.isna().sum()
)

print("\nTarget statistics:")
print(
    df["target_demand_kw"].describe()
)

# ============================================================
# 8. SAVE
# ============================================================

df.to_csv(
    "ev_ml_dataset.csv",
    index=False
)

print("\n✅ Saved as ev_ml_dataset.csv")