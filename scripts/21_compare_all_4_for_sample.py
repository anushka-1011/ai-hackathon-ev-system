import os
import joblib
import pandas as pd
import numpy as np


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "predictions"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(
    TEST_PATH
)

print(
    f"Test dataset shape: {test_df.shape}"
)


# ============================================================
# FEATURES
# ============================================================

TARGET = "target_demand_kw"

metadata_cols = [
    "site_name",
    "stationID",
    "timestamp"
]

feature_cols = [
    col
    for col in test_df.columns
    if col not in metadata_cols + [TARGET]
]

print(
    f"Number of features: {len(feature_cols)}"
)


# ============================================================
# SELECT SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("SELECT TEST SAMPLE")
print("=" * 60)

print(
    f"Available rows: 1 to {len(test_df):,}"
)

while True:

    try:

        row_number = int(
            input(
                "\nEnter test row number: "
            )
        )

        if 1 <= row_number <= len(test_df):
            break

        print(
            f"Please enter a number between "
            f"1 and {len(test_df):,}."
        )

    except ValueError:

        print(
            "Please enter a valid integer."
        )


# Convert 1-based row number to pandas 0-based index
sample_index = row_number - 1

sample = test_df.iloc[
    sample_index
]

# IMPORTANT:
# Select directly from the DataFrame so numeric dtypes
# remain numeric instead of becoming object dtype.
X_sample = test_df.loc[
    [sample_index],
    feature_cols
].copy()

actual_target = sample[
    TARGET
]


# ============================================================
# DISPLAY SAMPLE INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("SELECTED SAMPLE")
print("=" * 60)

print(
    f"Test row:       {row_number}"
)

print(
    f"Site:            {sample['site_name']}"
)

print(
    f"Station:         {sample['stationID']}"
)

print(
    f"Timestamp:       {sample['timestamp']}"
)

print(
    f"Actual demand:   {actual_target:.6f} kW"
)


# ============================================================
# MODEL NAMES
# ============================================================

model_names = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "RandomForest",
    "ExtraTrees"
]


# ============================================================
# RESULT STORAGE
# ============================================================

predictions = {}


# ============================================================
# DATASET I
# 80:20 + SMOTER
# ============================================================

print("\n" + "=" * 60)
print("DATASET I - 80:20 + SMOTER")
print("=" * 60)

dataset_I_predictions = {}


for model_name in model_names:

    model_path = os.path.join(
        MODEL_DIR,
        f"dataset_I_{model_name.lower()}_model.pkl"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Dataset I model not found:\n{model_path}"
        )

    model = joblib.load(
        model_path
    )

    prediction = float(
        model.predict(X_sample)[0]
    )

    dataset_I_predictions[
        model_name
    ] = prediction

    print(
        f"{model_name:15s}: "
        f"{prediction:.6f} kW"
    )


predictions[
    "Dataset I"
] = dataset_I_predictions


# ============================================================
# DATASET II
# 80:20 + NO SMOTER
# ============================================================

print("\n" + "=" * 60)
print("DATASET II - 80:20 + NO SMOTER")
print("=" * 60)

dataset_II_predictions = {}


for model_name in model_names:

    model_path = os.path.join(
        MODEL_DIR,
        f"dataset_II_{model_name.lower()}_model.pkl"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Dataset II model not found:\n{model_path}"
        )

    model = joblib.load(
        model_path
    )

    prediction = float(
        model.predict(X_sample)[0]
    )

    dataset_II_predictions[
        model_name
    ] = prediction

    print(
        f"{model_name:15s}: "
        f"{prediction:.6f} kW"
    )


predictions[
    "Dataset II"
] = dataset_II_predictions


# ============================================================
# DATASET III
# 10-FOLD + SMOTER
# ============================================================

print("\n" + "=" * 60)
print("DATASET III - 10-FOLD + SMOTER")
print("=" * 60)

dataset_III_predictions = {}


dataset_III_dir = os.path.join(
    MODEL_DIR,
    "dataset_III_cv"
)


for model_name in model_names:

    fold_predictions = []

    print(
        f"\n{model_name}:"
    )

    for fold_number in range(
        1,
        11
    ):

        model_path = os.path.join(
            dataset_III_dir,
            f"fold_{fold_number:02d}_"
            f"{model_name.lower()}.pkl"
        )

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Dataset III fold model not found:\n"
                f"{model_path}"
            )

        model = joblib.load(
            model_path
        )

        prediction = float(
            model.predict(X_sample)[0]
        )

        fold_predictions.append(
            prediction
        )

    mean_prediction = float(
        np.mean(
            fold_predictions
        )
    )

    std_prediction = float(
        np.std(
            fold_predictions
        )
    )

    dataset_III_predictions[
        model_name
    ] = mean_prediction

    print(
        f"  Mean prediction: "
        f"{mean_prediction:.6f} kW"
    )

    print(
        f"  Fold std:        "
        f"{std_prediction:.6f} kW"
    )


predictions[
    "Dataset III"
] = dataset_III_predictions


# ============================================================
# DATASET IV
# 10-FOLD + NO SMOTER
# ============================================================

print("\n" + "=" * 60)
print("DATASET IV - 10-FOLD + NO SMOTER")
print("=" * 60)

dataset_IV_predictions = {}


dataset_IV_dir = os.path.join(
    MODEL_DIR,
    "dataset_IV_cv"
)


for model_name in model_names:

    fold_predictions = []

    print(
        f"\n{model_name}:"
    )

    for fold_number in range(
        1,
        11
    ):

        model_path = os.path.join(
            dataset_IV_dir,
            f"fold_{fold_number:02d}_"
            f"{model_name.lower()}.pkl"
        )

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Dataset IV fold model not found:\n"
                f"{model_path}"
            )

        model = joblib.load(
            model_path
        )

        prediction = float(
            model.predict(X_sample)[0]
        )

        fold_predictions.append(
            prediction
        )

    mean_prediction = float(
        np.mean(
            fold_predictions
        )
    )

    std_prediction = float(
        np.std(
            fold_predictions
        )
    )

    dataset_IV_predictions[
        model_name
    ] = mean_prediction

    print(
        f"  Mean prediction: "
        f"{mean_prediction:.6f} kW"
    )

    print(
        f"  Fold std:        "
        f"{std_prediction:.6f} kW"
    )


predictions[
    "Dataset IV"
] = dataset_IV_predictions


# ============================================================
# CREATE 20-PREDICTION TABLE
# ============================================================

comparison_rows = []


for dataset_name in [
    "Dataset I",
    "Dataset II",
    "Dataset III",
    "Dataset IV"
]:

    row = {
        "test_row": row_number,
        "site_name": sample["site_name"],
        "stationID": sample["stationID"],
        "timestamp": sample["timestamp"],
        "actual_demand_kw": actual_target,
        "dataset": dataset_name
    }

    for model_name in model_names:

        row[
            model_name
        ] = predictions[
            dataset_name
        ][
            model_name
        ]

    comparison_rows.append(
        row
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# DISPLAY FINAL TABLE
# ============================================================

print("\n" + "=" * 80)
print("ALL 4 DATASETS × 5 MODELS")
print("=" * 80)

display_columns = [
    "dataset",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "RandomForest",
    "ExtraTrees"
]

print(
    comparison_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# ERROR COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("ABSOLUTE ERROR FROM ACTUAL DEMAND")
print("=" * 80)

error_df = comparison_df.copy()

for model_name in model_names:

    error_df[
        f"{model_name}_error"
    ] = np.abs(
        error_df[model_name]
        - actual_target
    )


error_columns = [
    "dataset"
]

for model_name in model_names:

    error_columns.append(
        f"{model_name}_error"
    )


print(
    error_df[
        error_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    f"sample_{row_number}_all_4_predictions.csv"
)


comparison_df.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(
    f"Results saved to:\n{output_path}"
)