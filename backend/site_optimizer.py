import math

import pandas as pd

from site_locations import SITE_LOCATIONS


# Total EVSEs at each physical site

SITE_EVSE_COUNTS = {
    "caltech": 52,
    "jpl": 52,
    "office001": 8,
}


# Historical average charging-session duration (minutes)
SITE_AVG_SESSION_MINUTES = {
    "caltech": 389.12,
    "jpl": 427.27,
    "office001": 356.83,
}


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in km."""

    R = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_site_conditions(predictions):
    """
    Calculate site-level demand, availability,
    utilization, and congestion.
    """

    site_rows = []

    for site in ["caltech", "jpl", "office001"]:

        site_data = predictions[
            predictions["site_name"].str.lower() == site.lower()
        ].copy()

        total_evse = SITE_EVSE_COUNTS[site]

        # Predicted demand across all EVSEs
        predicted_demand_kw = (
            site_data["predicted_demand_kw"].sum()
        )

        # Current demand across all EVSEs
        current_demand_kw = (
            site_data["current_demand_kw"].sum()
        )

        # Number of currently occupied EVSEs
        occupied_evse = (
            site_data["active_sessions"] > 0
        ).sum()

        # Number of available EVSEs
        available_evse = max(
            total_evse - occupied_evse,
            0
        )

        # Current EVSE utilization
        utilization_percentage = (
            occupied_evse / total_evse
        ) * 100

        # Congestion classification
        if utilization_percentage < 40:
            congestion = "Low"

        elif utilization_percentage < 70:
            congestion = "Medium"

        else:
            congestion = "High"

        site_rows.append({
            "site_name": site,
            "predicted_demand_kw": predicted_demand_kw,
            "current_demand_kw": current_demand_kw,
            "total_evse": total_evse,
            "occupied_evse": occupied_evse,
            "available_evse": available_evse,
            "utilization_percentage": utilization_percentage,
            "congestion": congestion,
        })

    return pd.DataFrame(site_rows)


def estimate_wait_time(
    site,
    occupied_evse,
    total_evse,
    available_evse
):
    """
    Estimate waiting time based on current EVSE availability
    and the site's historical average charging-session duration.

    If an EVSE is immediately available, waiting time is 0.

    If all EVSEs are occupied, the historical average charging
    session duration is used as a site-level waiting-time estimate.
    """

    # If an EVSE is immediately available,
    # no waiting is required.
    if available_evse > 0:
        return 0.0

    # All EVSEs are occupied.
    # Use the site's historical average charging-session
    # duration as an estimated waiting-time proxy.
    return SITE_AVG_SESSION_MINUTES[site]


def optimize_site(
    predictions,
    user_latitude,
    user_longitude,
    energy_required_kwh,
    max_wait_minutes
):

    # ---------------------------------------------------------
    # 1. Calculate current site conditions
    # ---------------------------------------------------------

    site_conditions = calculate_site_conditions(
        predictions
    )

    site_rows = []

    for site in ["caltech", "jpl", "office001"]:

        location = SITE_LOCATIONS[site]

        matching = site_conditions[
            site_conditions["site_name"].str.lower()
            == site.lower()
        ]

        row = matching.iloc[0]

        # -----------------------------------------------------
        # GPS distance
        # -----------------------------------------------------

        distance_km = haversine_distance_km(
            user_latitude,
            user_longitude,
            location["latitude"],
            location["longitude"]
        )

        # -----------------------------------------------------
        # Estimated waiting time
        # -----------------------------------------------------

        estimated_wait_minutes = estimate_wait_time(
            site=site,
            occupied_evse=int(row["occupied_evse"]),
            total_evse=int(row["total_evse"]),
            available_evse=int(row["available_evse"])
        )

        site_rows.append({
            "site_name": site,
            "site_display_name": location["name"],
            "latitude": location["latitude"],
            "longitude": location["longitude"],

            "predicted_demand_kw":
                float(row["predicted_demand_kw"]),

            "current_demand_kw":
                float(row["current_demand_kw"]),

            "total_evse":
                int(row["total_evse"]),

            "occupied_evse":
                int(row["occupied_evse"]),

            "available_evse":
                int(row["available_evse"]),

            "utilization_percentage":
                float(row["utilization_percentage"]),

            "congestion":
                row["congestion"],

            "distance_km":
                distance_km,

            "estimated_wait_minutes":
                estimated_wait_minutes,
        })

    results = pd.DataFrame(site_rows)

    # ---------------------------------------------------------
    # 2. Normalize demand and distance
    # ---------------------------------------------------------

    max_demand = results["predicted_demand_kw"].max()
    max_distance = results["distance_km"].max()

    results["demand_score"] = (
        results["predicted_demand_kw"] / max_demand
        if max_demand > 0
        else 0.0
    )

    results["distance_score"] = (
        results["distance_km"] / max_distance
        if max_distance > 0
        else 0.0
    )

    # Lower utilization = better
    results["availability_score"] = (
        results["available_evse"]
        / results["total_evse"]
    )

    # Lower waiting time = better
    max_wait_observed = results[
        "estimated_wait_minutes"
    ].max()

    if max_wait_observed > 0:

        results["wait_score"] = (
            results["estimated_wait_minutes"]
            / max_wait_observed
        )

    else:

        results["wait_score"] = 0.0

    # ---------------------------------------------------------
    # 3. Apply maximum-wait constraint
    # ---------------------------------------------------------

    results["wait_constraint"] = (
        results["estimated_wait_minutes"]
        <= max_wait_minutes
    )

    # ---------------------------------------------------------
    # 4. Optimization score
    #
    # Lower score = better
    #
    # 35% predicted demand
    # 30% GPS distance
    # 20% availability
    # 15% waiting time
    # ---------------------------------------------------------

    results["optimization_score"] = (
        0.35 * results["demand_score"]
        + 0.30 * results["distance_score"]
        + 0.20 * (
            1 - results["availability_score"]
        )
        + 0.15 * results["wait_score"]
    )

    # ---------------------------------------------------------
    # 5. Prefer sites within user's maximum wait
    # ---------------------------------------------------------

    feasible_sites = results[
        results["wait_constraint"]
    ].copy()

    if len(feasible_sites) > 0:

        feasible_sites = feasible_sites.sort_values(
            "optimization_score"
        ).reset_index(drop=True)

        recommended = feasible_sites.iloc[0]

    else:

        # If no site satisfies the requested maximum wait,
        # choose the site with the shortest estimated wait.
        recommended = results.sort_values(
            [
                "estimated_wait_minutes",
                "optimization_score"
            ]
        ).iloc[0]

    # ---------------------------------------------------------
    # 6. Sort complete result table
    # ---------------------------------------------------------

    results = results.sort_values(
        "optimization_score"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 7. Final recommendation
    # ---------------------------------------------------------

    recommendation = {

        "recommended_site":
            recommended["site_display_name"],

        "site_name":
            recommended["site_name"],

        "distance_km":
            round(
                float(recommended["distance_km"]),
                2
            ),

        "predicted_demand_kw":
            round(
                float(
                    recommended["predicted_demand_kw"]
                ),
                2
            ),

        "current_demand_kw":
            round(
                float(
                    recommended["current_demand_kw"]
                ),
                2
            ),

        "total_evse":
            int(recommended["total_evse"]),

        "occupied_evse":
            int(recommended["occupied_evse"]),

        "available_evse":
            int(recommended["available_evse"]),

        "utilization_percentage":
            round(
                float(
                    recommended[
                        "utilization_percentage"
                    ]
                ),
                2
            ),

        "estimated_wait_minutes":
            round(
                float(
                    recommended[
                        "estimated_wait_minutes"
                    ]
                ),
                2
            ),

        "wait_within_limit":
            bool(
                recommended["wait_constraint"]
            ),

        "congestion":
            recommended["congestion"],

        "optimization_score":
            round(
                float(
                    recommended["optimization_score"]
                ),
                4
            ),

        "energy_required_kwh":
            energy_required_kwh,

        "max_wait_minutes":
            max_wait_minutes,
    }

    return results, recommendation