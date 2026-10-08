# EV_CHARGING_DEMAND_PREDICTION_AND_STATION_OPTIMIZATION

## Hybrid prediction and station optimization

Run the complete requested evaluation from the repository root with:

```text
python scripts/25_hybrid_models_station_optimization.py
```

The runner reads `data/processed/ml/ev_ml_dataset.csv`, recreates the clean
chronological 80/20 `train_base.csv` and `test.csv` files, and evaluates these
five named hybrids:

- GNN + LSTM + Transformer
- CNN + LSTM
- Transformer + Attention
- GNN + BiLSTM
- Autoencoder + LSTM

The source data is tabular, so the pipeline encodes each component as a
feature block: same-site station peer statistics for GNN, demand-history
sequences for LSTM/BiLSTM, local lag convolutions for CNN, recency and cyclic
time attention for Transformer/Attention, and a fold-local PCA latent space
for Autoencoder. A shared gradient-boosting head makes the five experiments
comparable and keeps the project runnable with the existing Python stack.

Outputs are written to `outputs/metrics/` and `outputs/predictions/`:

- `hybrid_model_comparison.csv` contains 80/20 and per-fold metrics.
- `hybrid_10fold_summary.csv` contains mean 10-fold metrics.
- `hybrid_run_summary.json` records the selected holdout model and run size.
- `station_load_predictions.csv` contains predicted/current load per station.
- `station_allocation_recommendations.csv` ranks available stations at the
	latest prediction timestamp and reports allocated and unmet power.

The reusable station allocator is in `backend/station_allocation.py`.

## Coordinate-based recommendation

No frontend is required. After running the hybrid pipeline, provide GPS
coordinates and request details directly on the command line:

```text
python scripts/26_coordinate_station_recommendation.py --date 2019-05-19 --time 23:45 --latitude 34.134765 --longitude -118.127183 --energy-kwh 20 --max-wait-minutes 60
```

The command returns the prediction timestamp used, distance to each site,
predicted demand, available EVSEs, congestion, estimated waiting time, the
recommended site, and ranked available station recommendations. Add
`--output recommendation.json` to save the response as JSON.

## Weather, price, and charging behavior analysis

Run the additional research objectives with:

```text
python scripts/27_weather_price_behavior_analysis.py
```

This evaluates all five hybrids with three feature variants: full weather and
price inputs, weather removed, and electricity price removed. It reports both
80/20 and 10-fold MAE, RMSE, MAPE, and R2 metrics, and computes permutation
importance for the external features. It also aggregates historical ACN
sessions by `userID` and assigns four behavior labels: `frequent`,
`occasional`, `peak_hour`, and `long_duration`.

Results are saved to:

- `outputs/metrics/weather_price_ablation.csv`
- `outputs/metrics/weather_price_feature_importance.csv`
- `outputs/behavior/user_behavior_clusters.csv`
- `outputs/behavior/behavior_cluster_summary.csv`
- `outputs/behavior/behavior_run_summary.json`

Create a visual graph of the behavior clusters with:

```text
python scripts/28_visualize_behavior_clusters.py
```

The PCA scatter plot is saved as `outputs/behavior/behavior_clusters_pca.png`.

## SMOTER and non-SMOTER hybrid comparison

Evaluate all five hybrids on both prepared training variants with:

```text
python scripts/29_hybrid_smoter_comparison.py
```

This evaluates `dataset_I_SMOTER_train.csv` and `dataset_II_train.csv` using
the untouched `test.csv` holdout and 10-fold validation. Dataset I is the
SMOTER-balanced training set; Dataset II is the non-SMOTER training set.
Because SMOTER rows are synthetic and do not retain real timestamps, this
comparison uses row-based folds and labels that protocol explicitly.

Results are saved to:

- `outputs/metrics/hybrid_smoter_comparison.csv`
- `outputs/metrics/hybrid_smoter_summary.csv`
- `outputs/metrics/hybrid_smoter_run_summary.json`

## Blended deployment predictions

To make both training regimes influence coordinate recommendations, build the
deployment prediction file with:

```text
python scripts/30_build_blended_deployment_predictions.py
```

This selects the best 80/20 hybrid separately for Dataset I and Dataset II,
fits each selected hybrid on its complete training dataset, and blends their
test predictions using inverse-RMSE weights. The coordinate recommender uses
this blended file by default:

- `outputs/predictions/smoter_blended_station_predictions.csv`
- `outputs/predictions/smoter_blended_run_summary.json`