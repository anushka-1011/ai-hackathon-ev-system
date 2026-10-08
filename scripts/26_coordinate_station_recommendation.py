"""Recommend a charging site and station from coordinates and a time request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "backend"))

from backend.site_optimizer import optimize_site
from backend.station_allocation import allocate_station_load

PREDICTIONS_FILE = (
    BASE_DIR / "outputs" / "predictions" / "hybrid_station_predictions.csv"
)


def nearest_timestamp(timestamps: pd.Series, requested: pd.Timestamp) -> pd.Timestamp:
    return min(timestamps.drop_duplicates(), key=lambda value: abs(value - requested))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend a charging site and available station using GPS coordinates."
    )
    parser.add_argument("--date", required=True, help="Arrival date, e.g. 2019-05-20")
    parser.add_argument("--time", required=True, help="Arrival time, e.g. 08:00")
    parser.add_argument("--latitude", required=True, type=float)
    parser.add_argument("--longitude", required=True, type=float)
    parser.add_argument("--energy-kwh", required=True, type=float)
    parser.add_argument("--max-wait-minutes", type=float, default=60.0)
    parser.add_argument("--predictions", default=str(PREDICTIONS_FILE))
    parser.add_argument("--output", help="Optional JSON output path")
    arguments = parser.parse_args()

    if arguments.energy_kwh <= 0:
        raise ValueError("--energy-kwh must be greater than zero")
    if not -90 <= arguments.latitude <= 90:
        raise ValueError("--latitude must be between -90 and 90")
    if not -180 <= arguments.longitude <= 180:
        raise ValueError("--longitude must be between -180 and 180")

    predictions = pd.read_csv(arguments.predictions)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    timezone = predictions["timestamp"].dt.tz
    requested = pd.to_datetime(f"{arguments.date} {arguments.time}")
    if requested.tzinfo is None and timezone is not None:
        requested = requested.tz_localize(timezone)
    used_timestamp = nearest_timestamp(predictions["timestamp"], requested)
    selected = predictions[predictions["timestamp"] == used_timestamp].copy()
    if selected.empty:
        raise ValueError("No station predictions are available for the requested time.")

    site_table, site_recommendation = optimize_site(
        selected,
        user_latitude=arguments.latitude,
        user_longitude=arguments.longitude,
        energy_required_kwh=arguments.energy_kwh,
        max_wait_minutes=arguments.max_wait_minutes,
    )
    distances = site_table.set_index("site_name")["distance_km"]
    selected["distance_km"] = selected["site_name"].str.lower().map(distances)
    recommended_site = site_recommendation["site_name"].lower()
    station_candidates = selected[
        selected["site_name"].str.lower() == recommended_site
    ].copy()
    station_table = allocate_station_load(
        station_candidates[[
            "stationID", "site_name", "predicted_demand_kw", "active_sessions",
            "distance_km",
        ]],
        requested_power_kw=7.2,
    )
    station_table["requested_energy_kwh"] = arguments.energy_kwh
    station_table["estimated_charging_hours"] = station_table.apply(
        lambda row: (
            arguments.energy_kwh / row["allocated_power_kw"]
            if row["allocated_power_kw"] > 0 else 0.0
        ),
        axis=1,
    )
    station_table["energy_request_fulfilled"] = (
        station_table["allocated_power_kw"] > 0
    )
    selected_model = (
        str(selected["selected_model"].dropna().iloc[0])
        if "selected_model" in selected and selected["selected_model"].notna().any()
        else "unknown"
    )
    result = {
        "requested_timestamp": str(requested),
        "prediction_timestamp_used": str(used_timestamp),
        "selected_model": selected_model,
        "input_latitude": arguments.latitude,
        "input_longitude": arguments.longitude,
        "site_recommendation": site_recommendation,
        "site_comparison": site_table.to_dict(orient="records"),
        "station_recommendations": station_table.to_dict(orient="records"),
    }

    print(json.dumps(result, indent=2, default=str))
    if arguments.output:
        Path(arguments.output).write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )


if __name__ == "__main__":
    main()