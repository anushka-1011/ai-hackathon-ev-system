from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "acn"
    / "acn_all_sessions.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "acn"
    / "acn_clean_sessions.csv"
)

# ==========================================
# 1. LOAD COMBINED ACN DATA
# ==========================================

acn = pd.read_csv(INPUT_FILE)
# Convert timestamps
acn["connectionTime"] = pd.to_datetime(
    acn["connectionTime"],
    errors="coerce"
)

acn["disconnectTime"] = pd.to_datetime(
    acn["disconnectTime"],
    errors="coerce"
)

acn["doneChargingTime"] = pd.to_datetime(
    acn["doneChargingTime"],
    errors="coerce"
)

# Make energy numeric
acn["kWhDelivered"] = pd.to_numeric(
    acn["kWhDelivered"],
    errors="coerce"
)


# ==========================================
# 2. KEEP ONLY OUR EXACT STUDY PERIOD
# ==========================================

acn = acn[
    (acn["connectionTime"] >= "2019-04-01 00:00:00-07:00") &
    (acn["connectionTime"] <  "2019-06-01 00:00:00-07:00")
].copy()


# ==========================================
# 3. CALCULATE ORIGINAL CHARGING DURATION
# ==========================================

acn["charging_duration_hours"] = (
    acn["doneChargingTime"] -
    acn["connectionTime"]
).dt.total_seconds() / 3600


# ==========================================
# 4. IDENTIFY MISSING / INVALID END TIMES
# ==========================================

invalid_end = (
    acn["doneChargingTime"].isna() |
    (acn["charging_duration_hours"] <= 0)
)

print("Sessions needing fallback:", invalid_end.sum())


# ==========================================
# 5. USE DISCONNECT TIME AS FALLBACK
# ==========================================

acn["charging_end_time"] = acn["doneChargingTime"]

acn.loc[invalid_end, "charging_end_time"] = (
    acn.loc[invalid_end, "disconnectTime"]
)

# Keep track of what we did
acn["charging_end_source"] = "doneChargingTime"

acn.loc[
    invalid_end,
    "charging_end_source"
] = "disconnectTime_fallback"


# ==========================================
# 6. RECALCULATE CLEAN DURATION
# ==========================================

acn["charging_duration_hours"] = (
    acn["charging_end_time"] -
    acn["connectionTime"]
).dt.total_seconds() / 3600


# ==========================================
# 7. FINAL QUALITY CHECK
# ==========================================

print("\n" + "=" * 60)
print("CLEANED ACN DATA")
print("=" * 60)

print("Rows:", len(acn))
print("Columns:", len(acn.columns))

print("\nDate range:")
print("Start:", acn["connectionTime"].min())
print("End:", acn["connectionTime"].max())

print("\nCharging end source:")
print(acn["charging_end_source"].value_counts())

print("\nMissing charging end:")
print(acn["charging_end_time"].isna().sum())

print(
    "Zero/negative charging duration:",
    (acn["charging_duration_hours"] <= 0).sum()
)

print("\nSessions by site:")
print(acn["site_name"].value_counts())


# ==========================================
# 8. SAVE CLEAN DATASET
# ==========================================

acn.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n Saved as:")
print(OUTPUT_FILE)