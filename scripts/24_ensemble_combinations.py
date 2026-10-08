import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


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
    "metrics"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading untouched test set...")

test_df = pd.read_csv(
    TEST_PATH
)

print(
    f"Test shape: {test_df.shape}"
)


# ============================================================
# FEATURES / TARGET
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

X_test = test_df[
    feature_cols
]

y_test = test_df[
    TARGET
].to_numpy()

print(
    f"Number of features: {len(feature_cols)}"
)

print(
    f"Test rows: {len(X_test):,}"
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    # Zero-safe MAPE
    non_zero = y_true != 0

    if non_zero.sum() > 0:

        mape = np.mean(
            np.abs(
                (
                    y_true[non_zero]
                    - y_pred[non_zero]
                )
                / y_true[non_zero]
            )
        ) * 100

    else:

        mape = np.nan

    return (
        mae,
        rmse,
        mape,
        r2
    )


# ============================================================
# TOP 3 MODELS
# ============================================================

top_models = [
    "XGBoost",
    "CatBoost",
    "LightGBM"
]


# ============================================================
# PAIRWISE COMBINATIONS
# ============================================================

combinations = [
    ("XGBoost", "CatBoost"),
    ("XGBoost", "LightGBM"),
    ("CatBoost", "LightGBM")
]


# ============================================================
# RESULT STORAGE
# ============================================================

all_results = []


# ============================================================
# GET PREDICTIONS FOR ONE MODEL
# ============================================================

def get_predictions(
    dataset,
    model_name
):

    # --------------------------------------------------------
    # Dataset I
    # --------------------------------------------------------

    if dataset == "Dataset I":

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

        prediction = model.predict(
            X_test
        )

        del model

        return prediction


    # --------------------------------------------------------
    # Dataset II
    # --------------------------------------------------------

    if dataset == "Dataset II":

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

        prediction = model.predict(
            X_test
        )

        del model

        return prediction


    # --------------------------------------------------------
    # Dataset III
    # 10 folds + SMOTER
    # --------------------------------------------------------

    if dataset == "Dataset III":

        dataset_dir = os.path.join(
            MODEL_DIR,
            "dataset_III_cv"
        )

        fold_predictions = []

        for fold_number in range(
            1,
            11
        ):

            model_path = os.path.join(
                dataset_dir,
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

            prediction = model.predict(
                X_test
            )

            fold_predictions.append(
                prediction
            )

            del model

        return np.mean(
            np.vstack(
                fold_predictions
            ),
            axis=0
        )


    # --------------------------------------------------------
    # Dataset IV
    # 10 folds + No SMOTER
    # --------------------------------------------------------

    if dataset == "Dataset IV":

        dataset_dir = os.path.join(
            MODEL_DIR,
            "dataset_IV_cv"
        )

        fold_predictions = []

        for fold_number in range(
            1,
            11
        ):

            model_path = os.path.join(
                dataset_dir,
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

            prediction = model.predict(
                X_test
            )

            fold_predictions.append(
                prediction
            )

            del model

        return np.mean(
            np.vstack(
                fold_predictions
            ),
            axis=0
        )


    raise ValueError(
        f"Unknown dataset: {dataset}"
    )


# ============================================================
# TEST ALL FOUR DATASET CONDITIONS
# ============================================================

datasets = [
    "Dataset I",
    "Dataset II",
    "Dataset III",
    "Dataset IV"
]


for dataset in datasets:

    print(
        "\n" + "=" * 80
    )

    print(
        f"{dataset}"
    )

    print(
        "=" * 80
    )


    # --------------------------------------------------------
    # Get predictions of the three selected models
    # --------------------------------------------------------

    model_predictions = {}


    for model_name in top_models:

        print(
            f"Loading predictions: "
            f"{model_name}"
        )

        model_predictions[
            model_name
        ] = get_predictions(
            dataset,
            model_name
        )


    # --------------------------------------------------------
    # Test all 3 pairwise combinations
    # --------------------------------------------------------

    for model_a, model_b in combinations:

        print(
            "\n" + "-" * 60
        )

        print(
            f"Combination: "
            f"{model_a} + {model_b}"
        )

        print(
            "-" * 60
        )


        prediction_a = model_predictions[
            model_a
        ]

        prediction_b = model_predictions[
            model_b
        ]


        # 50:50 equal-weight ensemble
        ensemble_prediction = (
            prediction_a
            + prediction_b
        ) / 2.0


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mae, rmse, mape, r2 = calculate_metrics(
            y_test,
            ensemble_prediction
        )


        print(
            f"MAE  : {mae:.6f}"
        )

        print(
            f"RMSE : {rmse:.6f}"
        )

        print(
            f"MAPE : {mape:.6f}%"
        )

        print(
            f"R²   : {r2:.6f}"
        )


        all_results.append({

            "dataset":
                dataset,

            "combination":
                f"{model_a} + {model_b}",

            "model_1":
                model_a,

            "model_2":
                model_b,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "MAPE":
                mape,

            "R2":
                r2,

            "ensemble_method":
                "50:50 average"
        })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# DISPLAY ALL 12 RESULTS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "ALL PAIRWISE ENSEMBLE RESULTS"
)

print(
    "=" * 100
)

print(
    results_df[
        [
            "dataset",
            "combination",
            "MAE",
            "RMSE",
            "MAPE",
            "R2"
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE RAW ENSEMBLE RESULTS
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "top3_pairwise_ensemble_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# COMPOSITE SCORE
# ============================================================
#
# Same criterion used for model selection:
#
# R²           = 50%
# MAE          = 25%
# RMSE         = 15%
# Consistency  = 10%
#
# Higher R² is better.
# Lower MAE is better.
# Lower RMSE is better.
# Lower R² standard deviation is better.
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "ENSEMBLE COMPOSITE SCORE"
)

print(
    "=" * 100
)


R2_WEIGHT = 0.50
MAE_WEIGHT = 0.25
RMSE_WEIGHT = 0.15
CONSISTENCY_WEIGHT = 0.10


print(
    "\nWeights:"
)

print(
    f"R²          : {R2_WEIGHT:.0%}"
)

print(
    f"MAE         : {MAE_WEIGHT:.0%}"
)

print(
    f"RMSE        : {RMSE_WEIGHT:.0%}"
)

print(
    f"Consistency  : {CONSISTENCY_WEIGHT:.0%}"
)


# ============================================================
# AVERAGE METRICS ACROSS ALL FOUR DATASETS
# ============================================================

ensemble_summary = (
    results_df
    .groupby("combination")
    .agg(
        average_R2=("R2", "mean"),
        average_MAE=("MAE", "mean"),
        average_RMSE=("RMSE", "mean"),
        R2_std=("R2", lambda x: x.std(ddof=0))
    )
)


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def higher_is_better(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            1.0,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


def lower_is_better(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            1.0,
            index=series.index
        )

    return (
        (maximum - series)
        / (maximum - minimum)
    )


# ============================================================
# NORMALIZED SCORES
# ============================================================

# R²: higher is better
ensemble_summary[
    "R2_score"
] = higher_is_better(
    ensemble_summary[
        "average_R2"
    ]
)


# MAE: lower is better
ensemble_summary[
    "MAE_score"
] = lower_is_better(
    ensemble_summary[
        "average_MAE"
    ]
)


# RMSE: lower is better
ensemble_summary[
    "RMSE_score"
] = lower_is_better(
    ensemble_summary[
        "average_RMSE"
    ]
)


# Consistency:
# lower R² standard deviation is better
ensemble_summary[
    "Consistency_score"
] = lower_is_better(
    ensemble_summary[
        "R2_std"
    ]
)


# ============================================================
# CALCULATE COMPOSITE SCORE
# ============================================================

ensemble_summary[
    "Composite_score"
] = (

    R2_WEIGHT
    * ensemble_summary["R2_score"]

    + MAE_WEIGHT
    * ensemble_summary["MAE_score"]

    + RMSE_WEIGHT
    * ensemble_summary["RMSE_score"]

    + CONSISTENCY_WEIGHT
    * ensemble_summary["Consistency_score"]
)


# Convert to score out of 100
ensemble_summary[
    "Composite_score_100"
] = (
    ensemble_summary[
        "Composite_score"
    ] * 100
)


# ============================================================
# RANK COMBINATIONS
# ============================================================

ensemble_summary = ensemble_summary.sort_values(
    by="Composite_score",
    ascending=False
)

ensemble_summary[
    "Rank"
] = range(
    1,
    len(ensemble_summary) + 1
)


# ============================================================
# DISPLAY COMPOSITE RESULTS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "COMBINATION RANKING"
)

print(
    "=" * 100
)


display_columns = [
    "Rank",
    "average_R2",
    "average_MAE",
    "average_RMSE",
    "R2_std",
    "R2_score",
    "MAE_score",
    "RMSE_score",
    "Consistency_score",
    "Composite_score_100"
]


print(
    ensemble_summary[
        display_columns
    ]
    .round(6)
    .to_string()
)


# ============================================================
# BEST COMBINATION
# ============================================================

best_combination = ensemble_summary.iloc[
    0
]

best_combination_name = (
    ensemble_summary.index[0]
)


print(
    "\n" + "=" * 100
)

print(
    "BEST COMBINATION"
)

print(
    "=" * 100
)

print(
    f"Combination:     "
    f"{best_combination_name}"
)

print(
    f"Composite Score: "
    f"{best_combination['Composite_score_100']:.2f}/100"
)

print(
    f"Average R²:      "
    f"{best_combination['average_R2']:.6f}"
)

print(
    f"Average MAE:     "
    f"{best_combination['average_MAE']:.6f}"
)

print(
    f"Average RMSE:    "
    f"{best_combination['average_RMSE']:.6f}"
)

print(
    f"R² Std:          "
    f"{best_combination['R2_std']:.6f}"
)


# ============================================================
# SAVE COMPOSITE SCORES
# ============================================================

ensemble_summary_path = os.path.join(
    OUTPUT_DIR,
    "ensemble_composite_scores.csv"
)

ensemble_summary.reset_index().to_csv(
    ensemble_summary_path,
    index=False
)


# ============================================================
# FINAL OUTPUT PATHS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "FILES SAVED"
)

print(
    "=" * 100
)

print(
    f"Raw ensemble results:\n{output_path}"
)

print(
    f"\nComposite scores:\n{ensemble_summary_path}"
)