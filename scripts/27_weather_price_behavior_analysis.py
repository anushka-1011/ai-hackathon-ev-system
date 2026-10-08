"""Evaluate external factors and segment ACN users by charging behaviour."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
]
PRICE_COLUMNS = ["electricity_price", "price_imputed"]


def load_hybrid_module():
    path = BASE_DIR / "scripts" / "25_hybrid_models_station_optimization.py"
    specification = importlib.util.spec_from_file_location("hybrid_runner", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def calculate_metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.maximum(np.asarray(prediction), 0.0)
    actual = np.asarray(y_true)
    non_zero = actual != 0
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mape": float(np.mean(np.abs((actual[non_zero] - prediction[non_zero]) /
                                      actual[non_zero])) * 100)
        if non_zero.any() else np.nan,
        "r2": float(r2_score(actual, prediction)),
    }


def evaluate_external_factor_variants(frame: pd.DataFrame, folds: int) -> pd.DataFrame:
    hybrid = load_hybrid_module()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    timestamps = sorted(frame["timestamp"].unique())
    split_time = timestamps[int(len(timestamps) * 0.8)]
    train = frame[frame["timestamp"] < split_time].reset_index(drop=True)
    test = frame[frame["timestamp"] >= split_time].reset_index(drop=True)
    model_names = ["GNN_LSTM_TRANSFORMER", "CNN_LSTM", "TRANSFORMER_ATTENTION",
                   "GNN_BILSTM", "AUTOENCODER_LSTM"]
    variants = {
        "full_weather_price": [],
        "without_weather": WEATHER_COLUMNS,
        "without_electricity_price": PRICE_COLUMNS,
    }
    rows = []
    for variant, removed_columns in variants.items():
        variant_train = train.drop(columns=removed_columns, errors="ignore")
        variant_test = test.drop(columns=removed_columns, errors="ignore")
        for model_name in model_names:
            prediction = hybrid.fit_predict(
                hybrid.feature_block(variant_train, model_name),
                variant_train[hybrid.TARGET],
                hybrid.feature_block(variant_test, model_name),
                model_name,
            )
            rows.append({
                "evaluation": "80_20",
                "variant": variant,
                "model": model_name,
                **calculate_metrics(variant_test[hybrid.TARGET], prediction),
            })
        splitter = hybrid.KFold(n_splits=folds, shuffle=False)
        unique_train_times = np.array(sorted(train["timestamp"].unique()))
        for fold, (train_time_index, validation_time_index) in enumerate(
            splitter.split(unique_train_times), 1
        ):
            fold_train = train[train["timestamp"].isin(
                unique_train_times[train_time_index]
            )].drop(columns=removed_columns, errors="ignore")
            fold_validation = train[train["timestamp"].isin(
                unique_train_times[validation_time_index]
            )].drop(columns=removed_columns, errors="ignore")
            for model_name in model_names:
                prediction = hybrid.fit_predict(
                    hybrid.feature_block(fold_train, model_name),
                    fold_train[hybrid.TARGET],
                    hybrid.feature_block(fold_validation, model_name),
                    model_name,
                )
                rows.append({
                    "evaluation": "10_fold",
                    "fold": fold,
                    "variant": variant,
                    "model": model_name,
                    **calculate_metrics(fold_validation[hybrid.TARGET], prediction),
                })
    return pd.DataFrame(rows)


def external_feature_importance(frame: pd.DataFrame) -> pd.DataFrame:
    hybrid = load_hybrid_module()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    timestamps = sorted(frame["timestamp"].unique())
    split_time = timestamps[int(len(timestamps) * 0.8)]
    train = frame[frame["timestamp"] < split_time].reset_index(drop=True)
    test = frame[frame["timestamp"] >= split_time].reset_index(drop=True)
    train_features = hybrid.feature_block(train, "GNN_LSTM_TRANSFORMER")
    test_features = hybrid.feature_block(test, "GNN_LSTM_TRANSFORMER")
    scaler = hybrid.StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_test = scaler.transform(test_features)
    model = hybrid.make_model()
    model.fit(scaled_train, train[hybrid.TARGET])
    sample_size = min(5000, len(test_features))
    sample = np.linspace(0, len(test_features) - 1, sample_size, dtype=int)
    importance = permutation_importance(
        model,
        scaled_test[sample],
        test[hybrid.TARGET].iloc[sample],
        n_repeats=3,
        random_state=hybrid.RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
    )
    result = pd.DataFrame({
        "feature": test_features.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    result["factor_group"] = np.select(
        [result["feature"].isin(WEATHER_COLUMNS),
         result["feature"].isin(PRICE_COLUMNS)],
        ["weather", "electricity_price"],
        default="other",
    )
    return result


def session_features(sessions: pd.DataFrame) -> pd.DataFrame:
    sessions = sessions.copy()
    sessions["connectionTime"] = pd.to_datetime(sessions["connectionTime"], errors="coerce")
    sessions["charging_end_time"] = pd.to_datetime(
        sessions["charging_end_time"], errors="coerce"
    )
    sessions["kWhDelivered"] = pd.to_numeric(sessions["kWhDelivered"], errors="coerce")
    sessions["charging_duration_hours"] = pd.to_numeric(
        sessions["charging_duration_hours"], errors="coerce"
    )
    sessions = sessions.dropna(subset=["userID", "connectionTime"])
    sessions = sessions[sessions["kWhDelivered"].ge(0)]
    sessions["start_hour"] = sessions["connectionTime"].dt.hour
    sessions["is_weekend"] = (sessions["connectionTime"].dt.dayofweek >= 5).astype(int)
    sessions["is_peak_hour"] = sessions["start_hour"].between(7, 10) | \
        sessions["start_hour"].between(16, 19)
    grouped = sessions.groupby("userID")
    features = grouped.agg(
        session_count=("sessionID", "nunique"),
        total_energy_kwh=("kWhDelivered", "sum"),
        mean_energy_kwh=("kWhDelivered", "mean"),
        mean_duration_hours=("charging_duration_hours", "mean"),
        median_duration_hours=("charging_duration_hours", "median"),
        peak_hour_share=("is_peak_hour", "mean"),
        weekend_share=("is_weekend", "mean"),
        mean_start_hour=("start_hour", "mean"),
        sites_used=("site_name", "nunique"),
    ).reset_index()
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def segment_behaviour(sessions: pd.DataFrame, clusters: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = session_features(sessions)
    feature_columns = [column for column in features.columns if column != "userID"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[feature_columns])
    cluster_count = min(clusters, len(features))
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=20)
    features["cluster_id"] = model.fit_predict(scaled)
    summary = features.groupby("cluster_id", as_index=False)[feature_columns].mean()
    remaining = set(summary["cluster_id"])
    labels = {}
    labels[summary.loc[summary["session_count"].idxmax(), "cluster_id"]] = "frequent"
    remaining.discard(next(cluster for cluster, label in labels.items() if label == "frequent"))
    if remaining:
        long_cluster = summary[summary["cluster_id"].isin(remaining)].set_index(
            "cluster_id"
        )["mean_duration_hours"].idxmax()
        labels[long_cluster] = "long_duration"
        remaining.discard(long_cluster)
    if remaining:
        peak_cluster = summary[summary["cluster_id"].isin(remaining)].set_index(
            "cluster_id"
        )["peak_hour_share"].idxmax()
        labels[peak_cluster] = "peak_hour"
        remaining.discard(peak_cluster)
    for cluster in remaining:
        labels[cluster] = "occasional"
    features["behavior_label"] = features["cluster_id"].map(labels)
    summary["behavior_label"] = summary["cluster_id"].map(labels)
    summary["user_count"] = features.groupby("cluster_id").size().values
    return features, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    (OUTPUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    behavior_dir = OUTPUT_DIR / "behavior"
    behavior_dir.mkdir(parents=True, exist_ok=True)

    ml = pd.read_csv(DATA_DIR / "ml" / "ev_ml_dataset.csv")
    ml["timestamp"] = pd.to_datetime(ml["timestamp"])
    comparison = evaluate_external_factor_variants(ml, args.folds)
    comparison.to_csv(OUTPUT_DIR / "metrics" / "weather_price_ablation.csv", index=False)
    importance = external_feature_importance(ml)
    importance.to_csv(OUTPUT_DIR / "metrics" / "weather_price_feature_importance.csv", index=False)

    acn = pd.read_csv(DATA_DIR / "acn" / "acn_clean_sessions.csv")
    assignments, summary = segment_behaviour(acn, clusters=4)
    assignments.to_csv(behavior_dir / "user_behavior_clusters.csv", index=False)
    summary.to_csv(behavior_dir / "behavior_cluster_summary.csv", index=False)
    metadata = {
        "weather_features": WEATHER_COLUMNS,
        "price_features": PRICE_COLUMNS,
        "weather_price_rows": len(comparison),
        "segmented_users": len(assignments),
        "clusters": int(summary["cluster_id"].nunique()),
    }
    (behavior_dir / "behavior_run_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()