import pandas as pd

# ============================================================
# 1. LOAD COMPLETE 15-MINUTE ACN DATA
# ============================================================

acn = pd.read_csv(
    "acn_complete_15min_station_demand.csv"
)

acn["timestamp"] = pd.to_datetime(
    acn["timestamp"]
)

# Create hourly timestamp for matching weather
acn["weather_hour"] = acn["timestamp"].dt.floor("h")


# ============================================================
# 2. LOAD WEATHER FILES
# ============================================================

caltech_weather = pd.read_csv(
    "caltech_weather.csv"
)

jpl_weather = pd.read_csv(
    "jpl_weather.csv"
)

office_weather = pd.read_csv(
    "office001_weather.csv"
)


# ============================================================
# 3. COMBINE WEATHER FILES
# ============================================================

weather = pd.concat(
    [
        caltech_weather,
        jpl_weather,
        office_weather
    ],
    ignore_index=True
)

# Convert weather time
weather["time"] = pd.to_datetime(
    weather["time"]
)

# Make weather time timezone-aware
weather["time"] = (
    weather["time"]
    .dt.tz_localize("America/Los_Angeles")
)

# Rename for merging
weather = weather.rename(
    columns={
        "time": "weather_hour"
    }
)


# ============================================================
# 4. KEEP ONLY REQUIRED WEATHER COLUMNS
# ============================================================

weather = weather[
    [
        "weather_hour",
        "site_name",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "weather_code"
    ]
]


# ============================================================
# 5. CHECK WEATHER DUPLICATES
# ============================================================

print("=" * 60)
print("WEATHER CHECK")
print("=" * 60)

print(
    "Duplicate weather records:",
    weather.duplicated(
        subset=["weather_hour", "site_name"]
    ).sum()
)

print(
    "Weather rows:",
    len(weather)
)


# ============================================================
# 6. MERGE WEATHER WITH ACN DATA
# ============================================================

combined = acn.merge(
    weather,
    on=[
        "weather_hour",
        "site_name"
    ],
    how="left"
)


# ============================================================
# 7. REMOVE TEMPORARY MATCHING COLUMN
# ============================================================

combined = combined.drop(
    columns=["weather_hour"]
)


# ============================================================
# 8. SORT
# ============================================================

combined = combined.sort_values(
    [
        "site_name",
        "stationID",
        "timestamp"
    ]
).reset_index(drop=True)


# ============================================================
# 9. CHECK RESULT
# ============================================================

print("\n" + "=" * 60)
print("ACN + WEATHER DATASET")
print("=" * 60)

print("Rows:", len(combined))
print("Columns:", len(combined.columns))

print("\nColumns:")
print(combined.columns.tolist())

print("\nMissing values:")
print(
    combined[
        [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "weather_code"
        ]
    ].isna().sum()
)

print("\nFirst 5 rows:")
print(combined.head())

# ============================================================
# 10. SAVE
# ============================================================

combined.to_csv(
    "acn_15min_with_weather.csv",
    index=False
)

print("\n✅ Saved as acn_15min_with_weather.csv")