import os
import time
import gc
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml",
    "train_base.csv"
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

# Folder for saving all 10 x 5 fold models
CV_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "dataset_IV_cv"
)

os.makedirs(
    CV_MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD 80% TRAINING DATA
# ============================================================

print("\nLoading 80% training data...")

df = pd.read_csv(
    TRAIN_PATH
)

print(
    f"Training data shape: {df.shape}"
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
    for col in df.columns
    if col not in metadata_cols + [TARGET]
]

print(
    f"Number of features: {len(feature_cols)}"
)


# ============================================================
# SORT BY TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)

unique_timestamps = np.sort(
    df["timestamp"].unique()
)

print(
    f"Unique timestamps: {len(unique_timestamps)}"
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
# MODEL FACTORY
# ============================================================

def create_models():

    return {

        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ),

        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ),

        "CatBoost": CatBoostRegressor(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            thread_count=-1
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
    }


# ============================================================
# STANDARD 10-FOLD CROSS-VALIDATION
# ============================================================

kf = KFold(
    n_splits=10,
    shuffle=False
)

all_results = []


for fold_number, (
    train_idx,
    val_idx
) in enumerate(
    kf.split(unique_timestamps),
    start=1
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"FOLD {fold_number}/10"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Select timestamps
    # --------------------------------------------------------

    train_times = unique_timestamps[
        train_idx
    ]

    val_times = unique_timestamps[
        val_idx
    ]


    # --------------------------------------------------------
    # Select rows
    # --------------------------------------------------------

    train_mask = df[
        "timestamp"
    ].isin(
        train_times
    )

    val_mask = df[
        "timestamp"
    ].isin(
        val_times
    )

    fold_train = df.loc[
        train_mask
    ]

    fold_val = df.loc[
        val_mask
    ]


    print(
        f"Training rows:   "
        f"{len(fold_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(fold_val):,}"
    )


    # ========================================================
    # X / y
    # ========================================================

    X_train = fold_train[
        feature_cols
    ]

    y_train = fold_train[
        TARGET
    ]

    X_val = fold_val[
        feature_cols
    ]

    y_val = fold_val[
        TARGET
    ]


    # ========================================================
    # TRAIN FIVE MODELS
    # ========================================================

    models = create_models()


    for model_name, model in models.items():

        print(
            "\n" + "-" * 60
        )

        print(
            f"Fold {fold_number} - "
            f"Training {model_name}"
        )

        print(
            "-" * 60
        )


        start_time = time.time()


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )


        training_time = (
            time.time()
            - start_time
        )


        # ----------------------------------------------------
        # Validation prediction
        # ----------------------------------------------------

        y_pred = model.predict(
            X_val
        )


        # ----------------------------------------------------
        # SAVE TRAINED FOLD MODEL
        # ----------------------------------------------------

        model_path = os.path.join(
            CV_MODEL_DIR,
            f"fold_{fold_number:02d}_"
            f"{model_name.lower()}.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"Model saved: {model_path}"
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mae, rmse, mape, r2 = calculate_metrics(
            y_val.to_numpy(),
            y_pred
        )


        print(
            f"Training time: "
            f"{training_time:.2f} seconds"
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


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        all_results.append({

            "dataset":
                "Dataset IV - 10Fold + No SMOTER",

            "fold":
                fold_number,

            "model":
                model_name,

            "train_rows":
                len(fold_train),

            "validation_rows":
                len(fold_val),

            "MAE":
                mae,

            "RMSE":
                rmse,

            "MAPE":
                mape,

            "R2":
                r2,

            "training_time_seconds":
                training_time,

            "model_path":
                model_path
        })


        # Release model
        del model

        gc.collect()


    # ========================================================
    # CLEAN FOLD MEMORY
    # ========================================================

    del (
        fold_train,
        fold_val,
        X_train,
        y_train,
        X_val,
        y_val,
        models
    )

    gc.collect()


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


fold_results_path = os.path.join(
    OUTPUT_DIR,
    "dataset_IV_cv_results.csv"
)


results_df.to_csv(
    fold_results_path,
    index=False
)


# ============================================================
# MODEL SUMMARY
# ============================================================

summary_df = (
    results_df
    .groupby("model")
    .agg(

        mean_MAE=("MAE", "mean"),
        std_MAE=("MAE", "std"),

        mean_RMSE=("RMSE", "mean"),
        std_RMSE=("RMSE", "std"),

        mean_MAPE=("MAPE", "mean"),
        std_MAPE=("MAPE", "std"),

        mean_R2=("R2", "mean"),
        std_R2=("R2", "std")
    )
    .reset_index()
)


summary_df = summary_df.sort_values(
    by="mean_R2",
    ascending=False
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "dataset_IV_cv_model_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DATASET IV - FINAL 10-FOLD RESULTS"
)

print(
    "=" * 70
)


print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\nFold results saved to:"
)

print(
    fold_results_path
)


print(
    "\nModel summary saved to:"
)

print(
    summary_path
)


print(
    "\nSaved fold models in:"
)

print(
    CV_MODEL_DIR
)