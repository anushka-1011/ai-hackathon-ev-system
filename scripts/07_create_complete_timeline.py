from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "acn"
    / "acn_15min_station_demand.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "acn"
    / "acn_complete_15min_station_demand.csv"
)

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

start = pd.Timestamp(
    "2019-04-01 00:00:00",
    tz="America/Los_Angeles"
)

end = pd.Timestamp(
    "2019-05-31 23:45:00",
    tz="America/Los_Angeles"
)

timestamps = pd.date_range(
    start=start,
    end=end,
    freq="15min"
)

stations = (
    df[["site_name", "stationID"]]
    .drop_duplicates()
)

stations["key"] = 1

time_df = pd.DataFrame({
    "timestamp": timestamps,
    "key": 1
})

full_grid = (
    stations
    .merge(time_df, on="key")
    .drop(columns="key")
)

complete = full_grid.merge(
    df,
    on=["timestamp", "site_name", "stationID"],
    how="left"
)

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

complete = complete.sort_values(
    ["site_name", "stationID", "timestamp"]
).reset_index(drop=True)

print("Rows:", len(complete))
print("Columns:", len(complete.columns))
print(complete.columns.tolist())

print(
    "\n⚠️ This script is configured to create:",
    OUTPUT_FILE
)

# IMPORTANT:
# Do not run this script to overwrite the existing dataset
# unless you intentionally want to regenerate it.