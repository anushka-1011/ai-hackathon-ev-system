import os
import time
import gc
import pandas as pd
import numpy as np
import joblib
import resreg

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Folder for saving all 10 x 5 fold models
CV_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "dataset_III_cv"
)

os.makedirs(CV_MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading 80% training data...")

df = pd.read_csv(TRAIN_PATH)

print(f"Training data shape: {df.shape}")


# ============================================================
# FEATURES / TARGET
# ============================================================

TARGET = "target_demand_kw"

# Metadata is NOT used as a model feature
metadata_cols = [
    "site_name",
    "stationID",
    "timestamp"
]

feature_cols = [
    col for col in df.columns
    if col not in metadata_cols + [TARGET]
]

print(f"Number of features: {len(feature_cols)}")


# ============================================================
# SORT BY TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

unique_timestamps = np.sort(
    df["timestamp"].unique()
)

print(f"Unique timestamps: {len(unique_timestamps)}")


# ============================================================
# STANDARD 10-FOLD CV
# ============================================================

kf = KFold(
    n_splits=10,
    shuffle=False
)


# ============================================================
# NOMINAL FEATURES
# ============================================================

nominal_cols = [
    "weather_code",
    "price_imputed",
    "hour",
    "minute",
    "day_of_week",
    "is_weekend"
]

nominal_indices = [
    feature_cols.index(col)
    for col in nominal_cols
]

continuous_cols = [
    col for col in feature_cols
    if col not in nominal_cols
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

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

    return mae, rmse, mape, r2


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
# CROSS-VALIDATION
# ============================================================

all_results = []


for fold_number, (train_time_idx, val_time_idx) in enumerate(
    kf.split(unique_timestamps),
    start=1
):

    print("\n" + "=" * 70)
    print(f"FOLD {fold_number}/10")
    print("=" * 70)


    # --------------------------------------------------------
    # Select timestamps for this fold
    # --------------------------------------------------------

    train_times = unique_timestamps[
        train_time_idx
    ]

    val_times = unique_timestamps[
        val_time_idx
    ]


    # --------------------------------------------------------
    # Select rows
    # --------------------------------------------------------

    train_mask = df["timestamp"].isin(
        train_times
    )

    val_mask = df["timestamp"].isin(
        val_times
    )

    fold_train = df.loc[
        train_mask
    ].copy()

    fold_val = df.loc[
        val_mask
    ].copy()


    print(
        f"Original fold training rows: "
        f"{len(fold_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(fold_val):,}"
    )


    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X_train = fold_train[
        feature_cols
    ].copy()

    y_train = fold_train[
        TARGET
    ].copy()

    X_val = fold_val[
        feature_cols
    ].copy()

    y_val = fold_val[
        TARGET
    ].copy()


    # --------------------------------------------------------
    # Rare threshold = upper 5% of positive demand
    # --------------------------------------------------------

    positive_train = y_train[
        y_train > 0
    ]

    rare_threshold = np.percentile(
        positive_train,
        95
    )

    print(
        f"Rare-demand threshold: "
        f"{rare_threshold:.6f} kW"
    )


    # --------------------------------------------------------
    # Standardize continuous variables for SMOTER
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()

    X_train_scaled[
        continuous_cols
    ] = scaler.fit_transform(
        X_train[
            continuous_cols
        ]
    )

    X_val_scaled = X_val.copy()

    X_val_scaled[
        continuous_cols
    ] = scaler.transform(
        X_val[
            continuous_cols
        ]
    )


    # --------------------------------------------------------
    # SMOTER on TRAINING FOLD ONLY
    # --------------------------------------------------------

    print("Applying SMOTER...")

    relevance = resreg.sigmoid_relevance(
        y_train.to_numpy(),
        cl=None,
        ch=rare_threshold
    )

    X_resampled, y_resampled = resreg.smoter(
        X_train_scaled,
        y_train.to_numpy(),
        relevance=relevance,
        relevance_threshold=0.5,
        k=5,
        over="balance",
        random_state=42,
        nominal=nominal_indices
    )


    # --------------------------------------------------------
    # Convert back to DataFrame
    # --------------------------------------------------------

    X_resampled = pd.DataFrame(
        X_resampled,
        columns=feature_cols
    )

    y_resampled = np.asarray(
        y_resampled
    )


    # --------------------------------------------------------
    # Reverse scaling
    # --------------------------------------------------------

    X_resampled[
        continuous_cols
    ] = scaler.inverse_transform(
        X_resampled[
            continuous_cols
        ]
    )


    print(
        f"SMOTER training rows: "
        f"{len(X_resampled):,}"
    )

    print(
        f"Rare rows after SMOTER: "
        f"{(
            y_resampled >= rare_threshold
        ).sum():,}"
    )


    # --------------------------------------------------------
    # Validation data stays untouched
    # --------------------------------------------------------

    X_val_final = X_val.copy()


    # --------------------------------------------------------
    # Train five models
    # --------------------------------------------------------

    models = create_models()


    for model_name, model in models.items():

        print("\n" + "-" * 60)

        print(
            f"Fold {fold_number} - "
            f"Training {model_name}"
        )

        print("-" * 60)


        start_time = time.time()


        # Train
        model.fit(
            X_resampled,
            y_resampled
        )


        training_time = (
            time.time()
            - start_time
        )


        # Validation prediction
        y_pred = model.predict(
            X_val_final
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

        print(f"MAE  : {mae:.6f}")
        print(f"RMSE : {rmse:.6f}")
        print(f"MAPE : {mape:.6f}%")
        print(f"R²   : {r2:.6f}")


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        all_results.append({

            "dataset":
                "Dataset III - 10Fold + SMOTER",

            "fold":
                fold_number,

            "model":
                model_name,

            "train_rows_original":
                len(fold_train),

            "validation_rows":
                len(fold_val),

            "train_rows_after_smoter":
                len(X_resampled),

            "rare_threshold":
                rare_threshold,

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


        # Release current model
        del model

        gc.collect()


    # --------------------------------------------------------
    # Clean fold memory
    # --------------------------------------------------------

    del (
        fold_train,
        fold_val,
        X_train,
        y_train,
        X_val,
        y_val,
        X_train_scaled,
        X_val_scaled,
        X_resampled,
        y_resampled,
        models
    )

    gc.collect()


# ============================================================
# SAVE ALL FOLD RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


fold_results_path = os.path.join(
    OUTPUT_DIR,
    "dataset_III_cv_results.csv"
)


results_df.to_csv(
    fold_results_path,
    index=False
)


# ============================================================
# MODEL-LEVEL SUMMARY
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
    "dataset_III_cv_model_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("DATASET III - FINAL 10-FOLD RESULTS")
print("=" * 70)


print(
    summary_df.to_string(
        index=False
    )
)


print("\nFold results saved to:")
print(fold_results_path)


print("\nModel summary saved to:")
print(summary_path)


print("\nSaved fold models in:")
print(CV_MODEL_DIR)