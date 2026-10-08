"""Station-level load allocation utilities."""

from __future__ import annotations

import pandas as pd


def allocate_station_load(
    station_predictions: pd.DataFrame,
    requested_power_kw: float = 7.2,
    max_recommendations: int = 5,
) -> pd.DataFrame:
    """Rank stations and allocate a request to the least congested stations.

    ``station_predictions`` must contain ``stationID``, ``site_name``,
    ``predicted_demand_kw``, and ``active_sessions``. One free EVSE is
    represented by an active-session count of zero. The returned table is
    intentionally station-level so it can be consumed by an API or UI.
    """

    required = {
        "stationID",
        "site_name",
        "predicted_demand_kw",
        "active_sessions",
    }
    missing = required.difference(station_predictions.columns)
    if missing:
        raise ValueError(f"Missing station allocation columns: {sorted(missing)}")
    if requested_power_kw <= 0:
        raise ValueError("requested_power_kw must be greater than zero")

    result = station_predictions.copy()
    result["predicted_demand_kw"] = pd.to_numeric(
        result["predicted_demand_kw"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    result["active_sessions"] = pd.to_numeric(
        result["active_sessions"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    result["available"] = result["active_sessions"] <= 0

    demand_scale = max(float(result["predicted_demand_kw"].max()), 1.0)
    result["demand_score"] = result["predicted_demand_kw"] / demand_scale
    result["occupancy_score"] = result["active_sessions"].clip(upper=1.0)
    if "distance_km" in result:
        distance_scale = max(float(result["distance_km"].max()), 1.0)
        result["distance_score"] = result["distance_km"] / distance_scale
        result["allocation_score"] = (
            0.45 * result["demand_score"]
            + 0.25 * result["occupancy_score"]
            + 0.30 * result["distance_score"]
        )
    else:
        result["allocation_score"] = (
            0.65 * result["demand_score"] + 0.35 * result["occupancy_score"]
        )
    result = result.sort_values(
        ["available", "allocation_score", "predicted_demand_kw"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    result["allocated_power_kw"] = 0.0
    remaining_power = float(requested_power_kw)

    for index, row in result.iterrows():
        if remaining_power <= 0 or not row["available"]:
            continue
        assigned = min(7.2, remaining_power)
        result.at[index, "allocated_power_kw"] = assigned
        remaining_power -= assigned

    result["power_request_fulfilled"] = (
        (result["allocated_power_kw"] > 0) & (remaining_power <= 1e-9)
    )
    result["unmet_power_kw"] = max(remaining_power, 0.0)
    result["recommendation_rank"] = range(1, len(result) + 1)
    return result.head(max_recommendations).copy()