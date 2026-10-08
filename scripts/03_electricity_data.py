import pandas as pd

# ============================================================
# 1. LOAD CURRENT 15-MINUTE DATA
# ============================================================

df = pd.read_csv("acn_15min_station_demand.csv")

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

# ============================================================
# 2. DEFINE EXACT STUDY PERIOD
# ============================================================

start = pd.Timestamp(
    "2019-04-01 00:00:00",
    tz="America/Los_Angeles"
)

end = pd.Timestamp(
    "2019-05-31 23:45:00",
    tz="America/Los_Angeles"
)

# 15-minute timestamps
timestamps = pd.date_range(
    start=start,
    end=end,
    freq="15min"
)

print("Number of timestamps:", len(timestamps))


# ============================================================
# 3. GET ALL SITE + STATION COMBINATIONS
# ============================================================

stations = (
    df[["site_name", "stationID"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print("Number of stations:", len(stations))


# ============================================================
# 4. CREATE FULL STATION × TIME GRID
# ============================================================

stations["key"] = 1

time_df = pd.DataFrame({
    "timestamp": timestamps,
    "key": 1
})

full_grid = stations.merge(
    time_df,
    on="key"
).drop(columns="key")

print("Complete grid rows:", len(full_grid))


# ============================================================
# 5. MERGE ACTUAL DEMAND DATA
# ============================================================

complete = full_grid.merge(
    df[
        [
            "timestamp",
            "site_name",
            "stationID",
            "energy_kwh",
            "demand_kw",
            "active_sessions"
        ]
    ],
    on=[
        "timestamp",
        "site_name",
        "stationID"
    ],
    how="left"
)


# ============================================================
# 6. FILL NON-CHARGING PERIODS WITH ZERO
# ============================================================

complete["energy_kwh"] = (
    complete["energy_kwh"].fillna(0)
)

complete["demand_kw"] = (
    complete["demand_kw"].fillna(0)
)

complete["active_sessions"] = (
    complete["active_sessions"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# 7. SORT CHRONOLOGICALLY
# ============================================================

complete = complete.sort_values(
    [
        "site_name",
        "stationID",
        "timestamp"
    ]
).reset_index(drop=True)


# ============================================================
# 8. SAVE
# ============================================================

complete.to_csv(
    "acn_complete_15min_station_demand.csv",
    index=False
)


# ============================================================
# 9. CHECK
# ============================================================

print("\n" + "=" * 60)
print("COMPLETE 15-MINUTE DATASET")
print("=" * 60)

print("Rows:", len(complete))
print("Columns:", len(complete.columns))

print("\nColumns:")
print(complete.columns.tolist())

print("\nMissing values:")
print(complete.isna().sum())

print("\nZero-demand rows:")
print((complete["demand_kw"] == 0).sum())

print("\nFirst 10 rows:")
print(complete.head(10))

print("\nLast 10 rows:")
print(complete.tail(10))

print("\n✅ Saved as acn_complete_15min_station_demand.csv")