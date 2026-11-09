"""Evaluate every hybrid on SMOTER and non-SMOTER training datasets."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
MODEL_NAMES = [
    "GNN_LSTM_TRANSFORMER",
    "CNN_LSTM",
    "TRANSFORMER_ATTENTION",
    "GNN_BILSTM",
    "AUTOENCODER_LSTM",
]


def load_hybrid_module():
    path = BASE_DIR / "scripts" / "25_hybrid_models_station_optimization.py"
    specification = importlib.util.spec_from_file_location("hybrid_runner", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def prepare_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore metadata required by the shared hybrid feature blocks."""
    frame = frame.copy()
    if "site_name" not in frame:
        frame["site_name"] = "synthetic_training_data"
    if "stationID" not in frame:
        frame["stationID"] = np.arange(len(frame)).astype(str)
    if "timestamp" not in frame:
        frame["timestamp"] = pd.date_range(
            "2019-01-01", periods=len(frame), freq="15min"
        )
    return frame


def evaluate_dataset(train_path: Path, test: pd.DataFrame, folds: int) -> list[dict]:
    hybrid = load_hybrid_module()
    train = prepare_training_frame(pd.read_csv(train_path))
    test = prepare_training_frame(test)
    rows = []
    dataset_name = train_path.stem
    is_smoter = dataset_name == "dataset_I_SMOTER_train"
    for model_name in MODEL_NAMES:
        prediction = hybrid.fit_predict(
            hybrid.feature_block(train, model_name),
            train[hybrid.TARGET],
            hybrid.feature_block(test, model_name),
            model_name,
        )
        rows.append({
            "dataset": dataset_name,
            "smoter": is_smoter,
            "evaluation": "80_20",
            "model": model_name,
            **hybrid.metrics(test[hybrid.TARGET], prediction),
            "train_rows": len(train),
            "test_rows": len(test),
        })

    splitter = KFold(n_splits=folds, shuffle=False)
    for fold, (train_index, validation_index) in enumerate(splitter.split(train), 1):
        fold_train = train.iloc[train_index]
        validation = train.iloc[validation_index]
        for model_name in MODEL_NAMES:
            prediction = hybrid.fit_predict(
                hybrid.feature_block(fold_train, model_name),
                fold_train[hybrid.TARGET],
                hybrid.feature_block(validation, model_name),
                model_name,
            )
            rows.append({
                "dataset": dataset_name,
                "smoter": is_smoter,
                "evaluation": "10_fold",
                "fold": fold,
                "model": model_name,
                **hybrid.metrics(validation[hybrid.TARGET], prediction),
                "train_rows": len(fold_train),
                "validation_rows": len(validation),
            })
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    test = pd.read_csv(DATA_DIR / "test.csv")
    rows = []
    for filename in ["dataset_I_SMOTER_train.csv", "dataset_II_train.csv"]:
        rows.extend(evaluate_dataset(DATA_DIR / filename, test, args.folds))
    result = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(METRICS_DIR / "hybrid_smoter_comparison.csv", index=False)
    summary = (result.groupby(["dataset", "smoter", "evaluation", "model"],
                              as_index=False)[["mae", "rmse", "mape", "r2"]]
               .mean())
    summary.to_csv(METRICS_DIR / "hybrid_smoter_summary.csv", index=False)
    best = (result[result["evaluation"] == "80_20"]
            .sort_values("rmse").iloc[0][["dataset", "model", "rmse"]]
            .to_dict())
    metadata = {"folds": args.folds, "rows": len(result), "best_80_20": best}
    (METRICS_DIR / "hybrid_smoter_run_summary.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()