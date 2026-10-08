import os
import time
import pandas as pd
import numpy as np

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
    "dataset_II_train.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml",
    "test.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "metrics"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Dataset II...")
train_df = pd.read_csv(TRAIN_PATH)

print("Loading untouched test set...")
test_df = pd.read_csv(TEST_PATH)

print(f"Dataset II shape: {train_df.shape}")
print(f"Test shape:      {test_df.shape}")


# ============================================================
# FEATURES / TARGET
# ============================================================

TARGET = "target_demand_kw"

# Dataset II already contains only:
# 25 predictive features + 1 target
feature_cols = [col for col in train_df.columns if col != TARGET]

X_train = train_df[feature_cols]
y_train = train_df[TARGET]

X_test = test_df[feature_cols]
y_test = test_df[TARGET]

print(f"\nNumber of features: {len(feature_cols)}")
print(f"Training rows:      {len(X_train):,}")
print(f"Testing rows:       {len(X_test):,}")


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    # Zero-safe MAPE
    non_zero = y_true != 0

    if non_zero.sum() > 0:
        mape = np.mean(
            np.abs(
                (y_true[non_zero] - y_pred[non_zero])
                / y_true[non_zero]
            )
        ) * 100
    else:
        mape = np.nan

    return mae, rmse, mape, r2


# ============================================================
# MODELS
# ============================================================

models = {

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
# TRAIN + TEST
# ============================================================

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print("=" * 60)

    start_time = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start_time

    print(f"{name} training completed in {train_time:.2f} seconds")

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    mae, rmse, mape, r2 = calculate_metrics(
        y_test,
        y_pred
    )

    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAPE : {mape:.6f}%")
    print(f"R²   : {r2:.6f}")

    results.append({
        "dataset": "Dataset II - No SMOTER",
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "training_time_seconds": train_time
    })

    # Save model
    model_path = os.path.join(
        MODEL_DIR,
        f"dataset_II_{name.lower()}_model.pkl"
    )

    import joblib
    joblib.dump(model, model_path)

    print(f"Model saved: {model_path}")


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

results_path = os.path.join(
    OUTPUT_DIR,
    "dataset_II_model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 60)
print("DATASET II RESULTS")
print("=" * 60)

print(
    results_df[
        ["model", "MAE", "RMSE", "MAPE", "R2"]
    ].to_string(index=False)
)

print(f"\nResults saved to:")
print(results_path)