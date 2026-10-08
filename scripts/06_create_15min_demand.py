from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "acn"
    / "acn_clean_sessions.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "acn"
    / "acn_15min_station_demand.csv"
)

acn = pd.read_csv(INPUT_FILE)

acn["connectionTime"] = pd.to_datetime(
    acn["connectionTime"],
    errors="coerce"
)

acn["charging_end_time"] = pd.to_datetime(
    acn["charging_end_time"],
    errors="coerce"
)

acn["kWhDelivered"] = pd.to_numeric(
    acn["kWhDelivered"],
    errors="coerce"
)

# Charging duration
acn["charging_duration_hours"] = (
    acn["charging_end_time"]
    - acn["connectionTime"]
).dt.total_seconds() / 3600

# Average charging power
acn["avg_power_kw"] = (
    acn["kWhDelivered"]
    / acn["charging_duration_hours"]
)

acn = acn[
    (acn["charging_duration_hours"] > 0) &
    (acn["avg_power_kw"] > 0)
].copy()

records = []

for _, row in acn.iterrows():

    start = row["connectionTime"]
    end = row["charging_end_time"]
    power = row["avg_power_kw"]

    interval_start = start.floor("15min")

    while interval_start < end:

        interval_end = (
            interval_start + pd.Timedelta(minutes=15)
        )

        overlap_start = max(start, interval_start)
        overlap_end = min(end, interval_end)

        overlap_minutes = (
            overlap_end - overlap_start
        ).total_seconds() / 60

        if overlap_minutes > 0:

            energy_kwh = (
                power * overlap_minutes / 60
            )

            records.append({
                "timestamp": interval_start,
                "site_name": row["site_name"],
                "stationID": row["stationID"],
                "energy_kwh": energy_kwh
            })

        interval_start = interval_end

load = pd.DataFrame(records)

station_load = (
    load
    .groupby(
        ["timestamp", "site_name", "stationID"],
        as_index=False
    )
    .agg(
        energy_kwh=("energy_kwh", "sum")
    )
)

station_load["demand_kw"] = (
    station_load["energy_kwh"] / 0.25
)

# Active sessions
active_records = []

for _, row in acn.iterrows():

    current = row["connectionTime"].floor("15min")
    end = row["charging_end_time"]

    while current < end:

        active_records.append({
            "timestamp": current,
            "site_name": row["site_name"],
            "stationID": row["stationID"]
        })

        current += pd.Timedelta(minutes=15)

active_df = pd.DataFrame(active_records)

active_df = (
    active_df
    .groupby(
        ["timestamp", "site_name", "stationID"],
        as_index=False
    )
    .size()
    .rename(columns={"size": "active_sessions"})
)

station_load = station_load.merge(
    active_df,
    on=["timestamp", "site_name", "stationID"],
    how="left"
)

station_load["active_sessions"] = (
    station_load["active_sessions"]
    .fillna(0)
    .astype(int)
)

station_load = station_load.sort_values(
    ["site_name", "stationID", "timestamp"]
).reset_index(drop=True)

print("Rows:", len(station_load))
print("Columns:", len(station_load.columns))
print(station_load.columns.tolist())

print(
    "\nThis script is configured to create:",
    OUTPUT_FILE
)

# IMPORTANT:
# Do not run this script to overwrite the existing dataset
# unless you intentionally want to regenerate it.