"""Build live recommendation predictions from both SMOTER training regimes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml"
METRICS_FILE = BASE_DIR / "outputs" / "metrics" / "hybrid_smoter_summary.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "predictions" / "smoter_blended_station_predictions.csv"


def load_hybrid_module():
    path = BASE_DIR / "scripts" / "25_hybrid_models_station_optimization.py"
    specification = importlib.util.spec_from_file_location("hybrid_runner", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
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


def main() -> None:
    hybrid = load_hybrid_module()
    summary = pd.read_csv(METRICS_FILE)
    holdout = summary[summary["evaluation"] == "80_20"].copy()
    selected = (holdout.sort_values("rmse")
                .groupby("dataset", as_index=False).first())
    test = prepare_frame(pd.read_csv(DATA_DIR / "test.csv"))
    predictions = []
    metadata = []

    for row in selected.to_dict(orient="records"):
        dataset = row["dataset"]
        model_name = row["model"]
        train = prepare_frame(pd.read_csv(DATA_DIR / f"{dataset}.csv"))
        prediction = hybrid.fit_predict(
            hybrid.feature_block(train, model_name),
            train[hybrid.TARGET],
            hybrid.feature_block(test, model_name),
            model_name,
        )
        predictions.append(prediction)
        metadata.append({
            "dataset": dataset,
            "model": model_name,
            "rmse": float(row["rmse"]),
        })

    rmse = np.array([item["rmse"] for item in metadata], dtype=float)
    weights = (1.0 / np.maximum(rmse, 1e-12))
    weights /= weights.sum()
    blended = np.average(np.vstack(predictions), axis=0, weights=weights)
    output = test[hybrid.META + ["demand_kw", "active_sessions"]].copy()
    output = output.rename(columns={"demand_kw": "current_demand_kw"})
    output["predicted_demand_kw"] = np.maximum(blended, 0.0)
    output["selected_model"] = "SMOTER_NON_SMOTER_WEIGHTED_BLEND"
    output["smoter_model"] = next(
        item["model"] for item in metadata if item["dataset"] == "dataset_I_SMOTER_train"
    )
    output["non_smoter_model"] = next(
        item["model"] for item in metadata if item["dataset"] == "dataset_II_train"
    )
    output.to_csv(OUTPUT_FILE, index=False)
    run_summary = {
        "selected_training_models": metadata,
        "weights": weights.tolist(),
        "output_rows": len(output),
        "output_file": str(OUTPUT_FILE),
    }
    summary_path = OUTPUT_FILE.with_name("smoter_blended_run_summary.json")
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()