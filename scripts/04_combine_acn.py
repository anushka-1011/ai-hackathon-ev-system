from pathlib import Path
import pandas as pd

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Input folder
RAW_ACN = BASE_DIR / "data" / "raw" / "acn"

# Output file
OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "acn"
    / "acn_all_sessions.csv"
)

# Load the three ACN files
caltech = pd.read_csv(RAW_ACN / "caltech_sessions.csv")
jpl = pd.read_csv(RAW_ACN / "jpl_sessions.csv")
office = pd.read_csv(RAW_ACN / "office001_sessions.csv")

# Add site name
caltech["site_name"] = "Caltech"
jpl["site_name"] = "JPL"
office["site_name"] = "Office001"

# Combine
acn = pd.concat(
    [caltech, jpl, office],
    ignore_index=True
)

# Convert timestamps
for column in [
    "connectionTime",
    "disconnectTime",
    "doneChargingTime"
]:
    acn[column] = pd.to_datetime(
        acn[column],
        errors="coerce"
    )

# Sort chronologically
acn = acn.sort_values(
    "connectionTime"
).reset_index(drop=True)

# Save
acn.to_csv(OUTPUT_FILE, index=False)

# Check
print("=" * 50)
print("ACN COMBINATION")
print("=" * 50)

print("Rows:", len(acn))
print("Columns:", len(acn.columns))

print("\nSessions by site:")
print(acn["site_name"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n ACN combination complete")