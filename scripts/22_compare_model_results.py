import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METRICS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "metrics"
)


# ============================================================
# RESULT FILE PATHS
# ============================================================

dataset_I_path = os.path.join(
    METRICS_DIR,
    "dataset_I_model_results.csv"
)

dataset_II_path = os.path.join(
    METRICS_DIR,
    "dataset_II_model_results.csv"
)

dataset_III_path = os.path.join(
    METRICS_DIR,
    "dataset_III_cv_model_summary.csv"
)

dataset_IV_path = os.path.join(
    METRICS_DIR,
    "dataset_IV_cv_model_summary.csv"
)


# ============================================================
# LOAD RESULTS
# ============================================================

print("\nLoading model results...")

d1 = pd.read_csv(dataset_I_path)
d2 = pd.read_csv(dataset_II_path)
d3 = pd.read_csv(dataset_III_path)
d4 = pd.read_csv(dataset_IV_path)


# ============================================================
# DATASET I
# ============================================================

d1 = d1[
    ["model", "MAE", "RMSE", "MAPE", "R2"]
].copy()


# ============================================================
# DATASET II
# ============================================================

d2 = d2[
    ["model", "MAE", "RMSE", "MAPE", "R2"]
].copy()


# ============================================================
# DATASET III
# ============================================================

d3 = d3[
    [
        "model",
        "mean_MAE",
        "mean_RMSE",
        "mean_MAPE",
        "mean_R2"
    ]
].copy()

d3 = d3.rename(
    columns={
        "mean_MAE": "MAE",
        "mean_RMSE": "RMSE",
        "mean_MAPE": "MAPE",
        "mean_R2": "R2"
    }
)


# ============================================================
# DATASET IV
# ============================================================

d4 = d4[
    [
        "model",
        "mean_MAE",
        "mean_RMSE",
        "mean_MAPE",
        "mean_R2"
    ]
].copy()

d4 = d4.rename(
    columns={
        "mean_MAE": "MAE",
        "mean_RMSE": "RMSE",
        "mean_MAPE": "MAPE",
        "mean_R2": "R2"
    }
)


# ============================================================
# SHOW DATASET I RESULTS
# ============================================================

print("\n" + "=" * 90)
print("DATASET I - 80:20 + SMOTER")
print("=" * 90)

print(
    d1.sort_values(
        "R2",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# SHOW DATASET II RESULTS
# ============================================================

print("\n" + "=" * 90)
print("DATASET II - 80:20 + NO SMOTER")
print("=" * 90)

print(
    d2.sort_values(
        "R2",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# SHOW DATASET III RESULTS
# ============================================================

print("\n" + "=" * 90)
print("DATASET III - 10-FOLD + SMOTER")
print("=" * 90)

print(
    d3.sort_values(
        "R2",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# SHOW DATASET IV RESULTS
# ============================================================

print("\n" + "=" * 90)
print("DATASET IV - 10-FOLD + NO SMOTER")
print("=" * 90)

print(
    d4.sort_values(
        "R2",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# MODEL ORDER
# ============================================================

models = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "RandomForest",
    "ExtraTrees"
]


# ============================================================
# R2 COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": models,

    "Dataset_I_R2": [
        d1.loc[
            d1["model"] == m,
            "R2"
        ].iloc[0]
        for m in models
    ],

    "Dataset_II_R2": [
        d2.loc[
            d2["model"] == m,
            "R2"
        ].iloc[0]
        for m in models
    ],

    "Dataset_III_R2": [
        d3.loc[
            d3["model"] == m,
            "R2"
        ].iloc[0]
        for m in models
    ],

    "Dataset_IV_R2": [
        d4.loc[
            d4["model"] == m,
            "R2"
        ].iloc[0]
        for m in models
    ]
})


# ============================================================
# MAE COMPARISON
# ============================================================

comparison["Dataset_I_MAE"] = [
    d1.loc[
        d1["model"] == m,
        "MAE"
    ].iloc[0]
    for m in models
]

comparison["Dataset_II_MAE"] = [
    d2.loc[
        d2["model"] == m,
        "MAE"
    ].iloc[0]
    for m in models
]

comparison["Dataset_III_MAE"] = [
    d3.loc[
        d3["model"] == m,
        "MAE"
    ].iloc[0]
    for m in models
]

comparison["Dataset_IV_MAE"] = [
    d4.loc[
        d4["model"] == m,
        "MAE"
    ].iloc[0]
    for m in models
]


# ============================================================
# RMSE COMPARISON
# ============================================================

comparison["Dataset_I_RMSE"] = [
    d1.loc[
        d1["model"] == m,
        "RMSE"
    ].iloc[0]
    for m in models
]

comparison["Dataset_II_RMSE"] = [
    d2.loc[
        d2["model"] == m,
        "RMSE"
    ].iloc[0]
    for m in models
]

comparison["Dataset_III_RMSE"] = [
    d3.loc[
        d3["model"] == m,
        "RMSE"
    ].iloc[0]
    for m in models
]

comparison["Dataset_IV_RMSE"] = [
    d4.loc[
        d4["model"] == m,
        "RMSE"
    ].iloc[0]
    for m in models
]


# ============================================================
# PRINT R2 TABLE
# ============================================================

print("\n" + "=" * 90)
print("R² COMPARISON ACROSS ALL FOUR DATASETS")
print("=" * 90)

print(
    comparison[
        [
            "Model",
            "Dataset_I_R2",
            "Dataset_II_R2",
            "Dataset_III_R2",
            "Dataset_IV_R2"
        ]
    ].to_string(index=False)
)


# ============================================================
# PRINT MAE TABLE
# ============================================================

print("\n" + "=" * 90)
print("MAE COMPARISON ACROSS ALL FOUR DATASETS")
print("=" * 90)

print(
    comparison[
        [
            "Model",
            "Dataset_I_MAE",
            "Dataset_II_MAE",
            "Dataset_III_MAE",
            "Dataset_IV_MAE"
        ]
    ].to_string(index=False)
)


# ============================================================
# PRINT RMSE TABLE
# ============================================================

print("\n" + "=" * 90)
print("RMSE COMPARISON ACROSS ALL FOUR DATASETS")
print("=" * 90)

print(
    comparison[
        [
            "Model",
            "Dataset_I_RMSE",
            "Dataset_II_RMSE",
            "Dataset_III_RMSE",
            "Dataset_IV_RMSE"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE COMPARISON
# ============================================================

output_path = os.path.join(
    METRICS_DIR,
    "all_4_dataset_model_comparison.csv"
)

comparison.to_csv(
    output_path,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 90)
print("COMPARISON SAVED")
print("=" * 90)

print(output_path)

print(
    "\nNo automatic model selection was performed."
)

print(
    "Use the tables above to choose your top 3 models."
)