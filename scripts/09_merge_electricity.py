import pandas as pd

# ============================================================
# 1. LOAD ACN + WEATHER DATA
# ============================================================

df = pd.read_csv(
    "acn_15min_with_weather.csv"
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

# Extract local calendar date
df["date"] = df["timestamp"].dt.date


# ============================================================
# 2. LOAD ELECTRICITY PRICE DATA
# ============================================================

price = pd.read_csv(
    "sp15_final_daily_price_apr_may_2019.csv"
)

price["date"] = pd.to_datetime(
    price["date"]
).dt.date


# ============================================================
# 3. MERGE PRICE INTO 15-MINUTE DATA
# ============================================================

df = df.merge(
    price,
    on="date",
    how="left"
)


# ============================================================
# 4. REMOVE TEMPORARY DATE COLUMN
# ============================================================

df = df.drop(
    columns=["date"]
)


# ============================================================
# 5. SORT
# ============================================================

df = df.sort_values(
    [
        "site_name",
        "stationID",
        "timestamp"
    ]
).reset_index(drop=True)


# ============================================================
# 6. CHECK RESULT
# ============================================================

print("=" * 60)
print("MASTER DATASET")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nElectricity price statistics:")
print(df["electricity_price"].describe())

print("\nPrice-imputed records:")
print(
    df["price_imputed"].value_counts()
)

print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# 7. SAVE MASTER DATASET
# ============================================================

df.to_csv(
    "ev_master_dataset.csv",
    index=False
)

print("\n✅ Saved as ev_master_dataset.csv")