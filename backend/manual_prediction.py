import os
import joblib
import pandas as pd
import numpy as np
from site_optimizer import optimize_site

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml",
    "test.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "target_demand_kw"

METADATA_COLS = [
    "site_name",
    "stationID",
    "timestamp"
]

# Best model combination from model-selection stage
MODEL_NAMES = [
    "XGBoost",
    "LightGBM"
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_model_path(model_name):
    """
    Convert model name to the actual saved model filename.
    """

    filename = (
        f"dataset_II_"
        f"{model_name.lower()}_model.pkl"
    )

    return os.path.join(
        MODEL_DIR,
        filename
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("EV CHARGING DEMAND PREDICTION")
print("=" * 70)

print("\nLoading test dataset...")

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"\nTest dataset not found:\n{TEST_PATH}"
    )

test_df = pd.read_csv(TEST_PATH)

test_df["timestamp"] = pd.to_datetime(
    test_df["timestamp"]
)

print(
    f"Test dataset loaded: "
    f"{len(test_df):,} rows"
)


# ============================================================
# IDENTIFY EXACT MODEL FEATURES
# ============================================================

feature_cols = [
    col
    for col in test_df.columns
    if col not in METADATA_COLS + [TARGET]
]

print(
    f"Number of model features: "
    f"{len(feature_cols)}"
)

if len(feature_cols) != 25:
    raise ValueError(
        "\nERROR: Expected exactly 25 model features, "
        f"but found {len(feature_cols)}.\n"
        f"Features found:\n{feature_cols}"
    )

print("\nModel features:")

for i, feature in enumerate(feature_cols, start=1):
    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# USER INPUT
# ============================================================

print("\n" + "=" * 70)
print("EV REQUEST")
print("=" * 70)

print(
    "\nFor this first backend version, "
    "the prediction timestamp must exist in the "
    "trained test dataset."
)

print(
    "\nAvailable prediction period:"
)

print(
    f"From: {test_df['timestamp'].min()}"
)

print(
    f"To:   {test_df['timestamp'].max()}"
)


# ------------------------------------------------------------
# DATE AND TIME
# ------------------------------------------------------------

while True:

    date_input = input(
        "\nArrival date (YYYY-MM-DD): "
    ).strip()

    time_input = input(
        "Arrival time (HH:MM): "
    ).strip()

    timestamp_text = (
        f"{date_input} {time_input}"
    )

    try:

        prediction_timestamp = pd.to_datetime(
            timestamp_text
        )

        # Match the timezone used by the test dataset
        dataset_timezone = test_df["timestamp"].dt.tz

        if dataset_timezone is not None:
            prediction_timestamp = prediction_timestamp.tz_localize(
                dataset_timezone
            )

        break

    except Exception:

        print(
            "\nInvalid date/time."
            "\nUse format: YYYY-MM-DD HH:MM"
        )


# ------------------------------------------------------------
# EV LOCATION
# ------------------------------------------------------------

while True:

    try:

        latitude = float(
            input(
                "\nEV latitude: "
            )
        )

        longitude = float(
            input(
                "EV longitude: "
            )
        )

        break

    except ValueError:

        print(
            "\nPlease enter valid numeric coordinates."
        )


# ------------------------------------------------------------
# ENERGY REQUIREMENT
# ------------------------------------------------------------

while True:

    try:

        energy_required = float(
            input(
                "\nEnergy required (kWh): "
            )
        )

        if energy_required <= 0:
            raise ValueError

        break

    except ValueError:

        print(
            "\nEnergy required must be greater than 0."
        )


# ------------------------------------------------------------
# MAXIMUM WAIT
# ------------------------------------------------------------

while True:

    try:

        maximum_wait = float(
            input(
                "\nMaximum acceptable wait (minutes): "
            )
        )

        if maximum_wait < 0:
            raise ValueError

        break

    except ValueError:

        print(
            "\nWait time cannot be negative."
        )


# ============================================================
# DISPLAY REQUEST
# ============================================================

print("\n" + "=" * 70)
print("REQUEST SUMMARY")
print("=" * 70)

print(
    f"Arrival time:       "
    f"{prediction_timestamp}"
)

print(
    f"EV latitude:        "
    f"{latitude}"
)

print(
    f"EV longitude:       "
    f"{longitude}"
)

print(
    f"Energy required:    "
    f"{energy_required:.2f} kWh"
)

print(
    f"Maximum wait:       "
    f"{maximum_wait:.0f} minutes"
)


# ============================================================
# FIND TIMESTAMP
# ============================================================

matching_rows = test_df[
    test_df["timestamp"] == prediction_timestamp
].copy()


if matching_rows.empty:

    print("\n" + "=" * 70)
    print("TIMESTAMP NOT AVAILABLE")
    print("=" * 70)

    print(
        "\nThe requested timestamp does not exist "
        "in test.csv."
    )

    print(
        "\nUse one of the 15-minute timestamps "
        "between:"
    )

    print(
        f"{test_df['timestamp'].min()}"
    )

    print(
        f"{test_df['timestamp'].max()}"
    )

    raise SystemExit


print(
    f"\nStations available at this timestamp: "
    f"{len(matching_rows)}"
)


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

X = matching_rows[
    feature_cols
].copy()


# Check for missing values
if X.isnull().any().any():

    missing_cols = X.columns[
        X.isnull().any()
    ].tolist()

    raise ValueError(
        "\nMissing values found in model input:\n"
        f"{missing_cols}"
    )


# ============================================================
# LOAD BEST MODEL COMBINATION
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL COMBINATION")
print("=" * 70)

print("XGBoost + LightGBM")
print("Ensemble method: 50:50 averaging")
print("Composite Score: 75.00/100")


print("\n" + "=" * 70)
print("LOADING TRAINED MODELS")
print("=" * 70)

models = {}

for model_name in MODEL_NAMES:

    model_path = get_model_path(
        model_name
    )

    print(
        f"\nLoading {model_name}..."
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"\nModel not found:\n{model_path}"
        )

    models[model_name] = joblib.load(
        model_path
    )

    print(
        f"{model_name} loaded successfully."
    )


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING STATION DEMAND PREDICTIONS")
print("=" * 70)

predictions = {}

for model_name, model in models.items():

    print(
        f"\nRunning {model_name}..."
    )

    prediction = model.predict(X)

    predictions[model_name] = prediction


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results = matching_rows[
    [
        "site_name",
        "stationID",
        "timestamp"
    ]
].copy()


results["xgboost_prediction_kw"] = (
    predictions["XGBoost"]
)

results["lightgbm_prediction_kw"] = (
    predictions["LightGBM"]
)


# ============================================================
# BEST MODEL COMBINATION: XGBoost + LightGBM
# ============================================================

# 50:50 averaging
results["ensemble_prediction_kw"] = (
    results["xgboost_prediction_kw"]
    + results["lightgbm_prediction_kw"]
) / 2


# ============================================================
# PREVENT NEGATIVE REGRESSION OUTPUTS
# ============================================================

prediction_columns = [
    "xgboost_prediction_kw",
    "lightgbm_prediction_kw",
    "ensemble_prediction_kw"
]

for column in prediction_columns:

    results[column] = results[column].clip(
        lower=0
    )


# ============================================================
# DISPLAY EVSE-LEVEL PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("EVSE DEMAND RESULTS")
print("=" * 70)

display_columns = [
    "site_name",
    "stationID",
    "ensemble_prediction_kw"
]

display_results = results[
    display_columns
].copy()

display_results = display_results.sort_values(
    "ensemble_prediction_kw"
)

for _, row in display_results.iterrows():

    print(
        f"{str(row['site_name']):12s} | "
        f"{str(row['stationID']):25s} | "
        f"{row['ensemble_prediction_kw']:8.3f} kW"
    )


# ============================================================
# SITE-LEVEL DEMAND OPTIMIZATION
# ============================================================
print("\n" + "=" * 70)
print("SITE DEMAND OPTIMIZATION")
print("=" * 70)

# Create the dataframe expected by the optimizer.
site_prediction_input = results[
    [
        "site_name",
        "stationID",
        "ensemble_prediction_kw"
    ]
].copy()

# Add current conditions from the original timestamp rows.
site_prediction_input["current_demand_kw"] = (
    matching_rows["demand_kw"].values
)

site_prediction_input["active_sessions"] = (
    matching_rows["active_sessions"].values
)

# Rename ensemble prediction for the optimizer.
site_prediction_input = site_prediction_input.rename(
    columns={
        "ensemble_prediction_kw": "predicted_demand_kw"
    }
)


# ------------------------------------------------------------
# Calculate site recommendation
# ------------------------------------------------------------

site_results, recommendation = optimize_site(
    predictions=site_prediction_input,
    user_latitude=latitude,
    user_longitude=longitude,
    energy_required_kwh=energy_required,
    max_wait_minutes=maximum_wait
)


# ============================================================
# DISPLAY SITE RESULTS
# ============================================================
print("\n" + "-" * 70)
print("SITE COMPARISON")
print("-" * 70)

for _, row in site_results.iterrows():

    print(f"\nSite: {row['site_display_name']}")
    print(f"Current Demand: {row['current_demand_kw']:.3f} kW")
    print(f"Predicted Demand: {row['predicted_demand_kw']:.3f} kW")
    print(f"Total EVSEs: {int(row['total_evse'])}")
    print(f"Occupied EVSEs: {int(row['occupied_evse'])}")
    print(f"Available EVSEs: {int(row['available_evse'])}")
    print(
        f"Utilization: "
        f"{row['utilization_percentage']:.2f}%"
    )
    print(f"Distance: {row['distance_km']:.2f} km")
    print(f"Congestion: {row['congestion']}")
    print(
        f"Optimization Score: "
        f"{row['optimization_score']:.4f}"
    )

# ============================================================
# FINAL SITE RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("SMART CHARGING SITE RECOMMENDATION")
print("=" * 70)

print(
    f"\nRecommended Site: "
    f"{recommendation['recommended_site']}"
)

print(
    f"Distance: "
    f"{recommendation['distance_km']:.2f} km"
)

print(
    f"Predicted Site Demand: "
    f"{recommendation['predicted_demand_kw']:.3f} kW"
)

print(
    f"Estimated Congestion: "
    f"{recommendation['congestion']}"
)

print(
    f"Optimization Score: "
    f"{recommendation['optimization_score']:.4f}"
)

print(
    "\nEnergy Required: "
    f"{recommendation['energy_required_kwh']:.2f} kWh"
)

print(
    "Maximum Wait Allowed: "
    f"{recommendation['max_wait_minutes']} minutes"
)
print(
    f"Estimated Wait: "
    f"{recommendation['estimated_wait_minutes']:.2f} minutes"
)

print(
    f"Within Wait Limit: "
    f"{'Yes' if recommendation['wait_within_limit'] else 'No'}"
)

print("\n" + "=" * 70)
print("PREDICTION + SITE OPTIMIZATION COMPLETE")
print("=" * 70)