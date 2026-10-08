"""Train the five requested demand hybrids and produce station decisions.

The project data is tabular, so each named hybrid is represented by a
domain-specific feature block followed by the same reproducible regressor.
This keeps the pipeline runnable without a deep-learning runtime while
preserving the intended components: station graph context, temporal memory,
local convolution, attention, and an autoencoder latent representation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml"
OUTPUT_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
sys.path.insert(0, str(BASE_DIR))
from backend.station_allocation import allocate_station_load

TARGET = "target_demand_kw"
META = ["site_name", "stationID", "timestamp"]
N_FOLDS = 10
RANDOM_STATE = 42


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    non_zero = y_true_array != 0
    return {
        "mae": float(mean_absolute_error(y_true_array, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, prediction))),
        "mape": float(
            np.mean(np.abs((y_true_array[non_zero] - prediction[non_zero]) /
                           y_true_array[non_zero])) * 100
        ) if non_zero.any() else np.nan,
        "r2": float(r2_score(y_true_array, prediction)),
    }


def temporal_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in META + [TARGET]]


def demand_lags(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in frame.columns
               if column.startswith("demand_lag_")]
    if not columns:
        columns = [column for column in frame.columns
                   if column in {"demand_kw", "energy_kwh", "active_sessions"}]
    return frame[columns].fillna(0.0).astype(float)


def graph_block(frame: pd.DataFrame, base: list[str]) -> pd.DataFrame:
    """Approximate a station graph with same-site peer statistics."""
    numeric = frame[base].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    grouped = numeric.assign(
        _site=frame["site_name"].astype(str),
        _timestamp=frame["timestamp"],
    ).groupby(["_site", "_timestamp"])
    peer_mean = grouped[base].transform("mean").add_suffix("_site_mean")
    peer_max = grouped[base].transform("max").add_suffix("_site_max")
    return pd.concat([numeric, peer_mean, peer_max], axis=1)


def attention_block(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply recency attention to the available demand history."""
    lag_frame = demand_lags(frame)
    positions = np.arange(len(lag_frame.columns), 0, -1, dtype=float)
    weights = np.exp(positions - positions.max())
    weights /= weights.sum()
    weighted = lag_frame.to_numpy() @ weights
    return pd.DataFrame({"attended_demand": weighted,
                         "recent_demand": lag_frame.iloc[:, -1].to_numpy()},
                        index=frame.index)


def feature_block(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    base = temporal_columns(frame)
    numeric = frame[base].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    lags = demand_lags(frame)
    blocks = [numeric]

    if model_name in {"GNN_LSTM_TRANSFORMER", "GNN_BILSTM"}:
        blocks.append(graph_block(frame, base).drop(columns=base))
    if model_name in {"GNN_LSTM_TRANSFORMER", "CNN_LSTM", "GNN_BILSTM",
                      "AUTOENCODER_LSTM"}:
        blocks.append(lags.add_suffix("_sequence"))
    if model_name == "CNN_LSTM":
        blocks.append(pd.DataFrame({
            "conv_mean": lags.mean(axis=1),
            "conv_max": lags.max(axis=1),
            "conv_delta": lags.iloc[:, -1] - lags.iloc[:, 0],
        }, index=frame.index))
    if model_name in {"GNN_LSTM_TRANSFORMER", "TRANSFORMER_ATTENTION"}:
        blocks.append(attention_block(frame))
    if model_name == "TRANSFORMER_ATTENTION":
        blocks.append(frame[[column for column in
                             ["hour_sin", "hour_cos", "day_sin", "day_cos"]
                             if column in frame]].fillna(0.0))
    return pd.concat(blocks, axis=1).loc[:, lambda value: ~value.columns.duplicated()]


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def fit_predict(train_x: pd.DataFrame, train_y: pd.Series,
                test_x: pd.DataFrame, model_name: str) -> np.ndarray:
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    scaled_test = scaler.transform(test_x)
    if model_name == "AUTOENCODER_LSTM":
        latent_size = max(2, min(12, scaled_train.shape[1] // 3))
        encoder = PCA(n_components=latent_size, random_state=RANDOM_STATE)
        scaled_train = encoder.fit_transform(scaled_train)
        scaled_test = encoder.transform(scaled_test)
    model = make_model()
    model.fit(scaled_train, train_y)
    return np.maximum(model.predict(scaled_test), 0.0)


def timestamp_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    timestamps = np.array(sorted(frame["timestamp"].unique()))
    splitter = KFold(n_splits=N_FOLDS, shuffle=False)
    folds = []
    for train_times, validation_times in splitter.split(timestamps):
        train_mask = frame["timestamp"].isin(timestamps[train_times]).to_numpy()
        validation_mask = frame["timestamp"].isin(timestamps[validation_times]).to_numpy()
        folds.append((np.flatnonzero(train_mask), np.flatnonzero(validation_mask)))
    return folds


def main() -> None:
    started = time.perf_counter()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    source = DATA_DIR / "ev_ml_dataset.csv"
    frame = pd.read_csv(source)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce").fillna(0.0)
    timestamps = sorted(frame["timestamp"].unique())
    split_time = timestamps[int(len(timestamps) * 0.8)]
    train = frame[frame["timestamp"] < split_time].copy().reset_index(drop=True)
    test = frame[frame["timestamp"] >= split_time].copy().reset_index(drop=True)
    train.to_csv(DATA_DIR / "train_base.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    model_names = ["GNN_LSTM_TRANSFORMER", "CNN_LSTM", "TRANSFORMER_ATTENTION",
                   "GNN_BILSTM", "AUTOENCODER_LSTM"]

    holdout_rows = []
    holdout_predictions = {}
    for model_name in model_names:
        prediction = fit_predict(feature_block(train, model_name), train[TARGET],
                                 feature_block(test, model_name), model_name)
        holdout_predictions[model_name] = prediction
        result = metrics(test[TARGET], prediction)
        holdout_rows.append({"evaluation": "80_20", "model": model_name,
                             **result, "train_rows": len(train),
                             "test_rows": len(test)})

    cv_rows = []
    cv_predictions = {name: np.full(len(train), np.nan) for name in model_names}
    for fold, (train_index, validation_index) in enumerate(timestamp_folds(train), 1):
        fold_train = train.iloc[train_index]
        fold_validation = train.iloc[validation_index]
        for model_name in model_names:
            prediction = fit_predict(feature_block(fold_train, model_name),
                                     fold_train[TARGET],
                                     feature_block(fold_validation, model_name),
                                     model_name)
            cv_predictions[model_name][validation_index] = prediction
            cv_rows.append({"evaluation": "10_fold", "fold": fold,
                            "model": model_name,
                            **metrics(fold_validation[TARGET], prediction),
                            "train_rows": len(fold_train),
                            "validation_rows": len(fold_validation)})

    holdout_metrics = pd.DataFrame(holdout_rows)
    cv_metrics = pd.DataFrame(cv_rows)
    all_metrics = pd.concat([holdout_metrics, cv_metrics], ignore_index=True)
    all_metrics.to_csv(METRICS_DIR / "hybrid_model_comparison.csv", index=False)
    cv_metrics.groupby("model", as_index=False)[["mae", "rmse", "mape", "r2"]].mean().to_csv(
        METRICS_DIR / "hybrid_10fold_summary.csv", index=False
    )

    best_model = holdout_metrics.sort_values("rmse").iloc[0]["model"]
    output = test[META + ["demand_kw", "active_sessions"]].copy()
    output = output.rename(columns={"demand_kw": "current_demand_kw"})
    output["predicted_demand_kw"] = holdout_predictions[best_model]
    output["selected_model"] = best_model
    output.to_csv(PREDICTIONS_DIR / "hybrid_station_predictions.csv", index=False)

    station = (output.groupby(["timestamp", "site_name", "stationID"], as_index=False)
               .agg(predicted_demand_kw=("predicted_demand_kw", "mean"),
                    current_demand_kw=("current_demand_kw", "mean"),
                    active_sessions=("active_sessions", "max")))
    demand_threshold = station["predicted_demand_kw"].quantile(0.75)
    station["demand_risk"] = np.where(
        station["predicted_demand_kw"] >= demand_threshold, "High", "Normal"
    )
    station["overcrowding_risk"] = np.where(
        (station["active_sessions"] > 0) &
        (station["predicted_demand_kw"] >= demand_threshold),
        "High", "Normal"
    )
    station.to_csv(PREDICTIONS_DIR / "station_load_predictions.csv", index=False)
    latest_timestamp = station["timestamp"].max()
    allocation = allocate_station_load(
        station[station["timestamp"] == latest_timestamp].drop(columns="timestamp")
    )
    allocation.insert(0, "timestamp", latest_timestamp)
    allocation.to_csv(PREDICTIONS_DIR / "station_allocation_recommendations.csv", index=False)

    summary = {"best_holdout_model": best_model, "train_rows": len(train),
               "test_rows": len(test), "folds": N_FOLDS,
               "elapsed_seconds": round(time.perf_counter() - started, 2)}
    (METRICS_DIR / "hybrid_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()