import os
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

METRICS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "metrics"
)


# ============================================================
# WEIGHTS
# ============================================================

R2_WEIGHT = 0.50
MAE_WEIGHT = 0.25
RMSE_WEIGHT = 0.15
CONSISTENCY_WEIGHT = 0.10


# ============================================================
# RESULT FILES
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
# EXTRACT REQUIRED METRICS
# ============================================================

d1 = d1[
    ["model", "MAE", "RMSE", "R2"]
].copy()

d2 = d2[
    ["model", "MAE", "RMSE", "R2"]
].copy()


# Dataset III uses mean values from 10-fold CV
d3 = d3[
    [
        "model",
        "mean_MAE",
        "mean_RMSE",
        "mean_R2"
    ]
].copy()

d3 = d3.rename(
    columns={
        "mean_MAE": "MAE",
        "mean_RMSE": "RMSE",
        "mean_R2": "R2"
    }
)


# Dataset IV uses mean values from 10-fold CV
d4 = d4[
    [
        "model",
        "mean_MAE",
        "mean_RMSE",
        "mean_R2"
    ]
].copy()

d4 = d4.rename(
    columns={
        "mean_MAE": "MAE",
        "mean_RMSE": "RMSE",
        "mean_R2": "R2"
    }
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
# COMBINE ALL FOUR DATASETS
# ============================================================

all_results = pd.concat(
    [
        d1.assign(dataset="Dataset I"),
        d2.assign(dataset="Dataset II"),
        d3.assign(dataset="Dataset III"),
        d4.assign(dataset="Dataset IV")
    ],
    ignore_index=True
)


# ============================================================
# AVERAGE METRICS ACROSS ALL FOUR CONDITIONS
# ============================================================

summary = (
    all_results
    .groupby("model")
    .agg(
        average_R2=("R2", "mean"),
        average_MAE=("MAE", "mean"),
        average_RMSE=("RMSE", "mean")
    )
    .reindex(models)
)


# ============================================================
# CONSISTENCY
# ============================================================
# Consistency is measured using the standard deviation of R²
# across Dataset I-IV.
#
# Lower standard deviation = more consistent.
# ============================================================

r2_consistency = (
    all_results
    .groupby("model")["R2"]
    .std(ddof=0)
    .reindex(models)
)

summary["R2_std"] = r2_consistency


# ============================================================
# MIN-MAX NORMALIZATION
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


# R²: higher is better
summary["R2_score"] = higher_is_better(
    summary["average_R2"]
)


# MAE: lower is better
summary["MAE_score"] = lower_is_better(
    summary["average_MAE"]
)


# RMSE: lower is better
summary["RMSE_score"] = lower_is_better(
    summary["average_RMSE"]
)


# Consistency: lower R² std is better
summary["Consistency_score"] = lower_is_better(
    summary["R2_std"]
)


# ============================================================
# COMPOSITE SCORE
# ============================================================

summary["Composite_score"] = (
    R2_WEIGHT * summary["R2_score"]
    + MAE_WEIGHT * summary["MAE_score"]
    + RMSE_WEIGHT * summary["RMSE_score"]
    + CONSISTENCY_WEIGHT * summary["Consistency_score"]
)


# Convert to 0-100
summary["Composite_score_100"] = (
    summary["Composite_score"] * 100
)


# ============================================================
# RANK MODELS
# ============================================================

summary = summary.sort_values(
    by="Composite_score",
    ascending=False
)

summary["Rank"] = range(
    1,
    len(summary) + 1
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 100)
print("MODEL SELECTION USING COMPOSITE SCORE")
print("=" * 100)

print(
    f"\nWeights:"
)

print(
    f"R²           = {R2_WEIGHT:.0%}"
)

print(
    f"MAE          = {MAE_WEIGHT:.0%}"
)

print(
    f"RMSE         = {RMSE_WEIGHT:.0%}"
)

print(
    f"Consistency   = {CONSISTENCY_WEIGHT:.0%}"
)


print("\n" + "-" * 100)

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
    summary[
        display_columns
    ].round(6).to_string()
)


# ============================================================
# PRINT FINAL RANKING
# ============================================================

print("\n" + "=" * 100)
print("FINAL MODEL RANKING")
print("=" * 100)

for model, row in summary.iterrows():

    print(
        f"{int(row['Rank'])}. "
        f"{model:15s} "
        f"Score = {row['Composite_score_100']:.2f}/100"
    )


# ============================================================
# PRINT TOP 3
# ============================================================

top_3 = summary.head(3)

print("\n" + "=" * 100)
print("TOP 3 MODELS")
print("=" * 100)

for model, row in top_3.iterrows():

    print(
        f"{int(row['Rank'])}. "
        f"{model} "
        f"({row['Composite_score_100']:.2f}/100)"
    )


# ============================================================
# SAVE RESULT
# ============================================================

output_path = os.path.join(
    METRICS_DIR,
    "model_selection_composite_scores.csv"
)

summary.reset_index().to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 100)
print("RESULT SAVED")
print("=" * 100)

print(output_path)