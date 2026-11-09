"""Create a 2D PCA graph of the charging-behavior clusters."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "outputs" / "behavior" / "user_behavior_clusters.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "behavior" / "behavior_clusters_pca.png"


def main() -> None:
    data = pd.read_csv(INPUT_FILE)
    feature_columns = [
        "session_count",
        "total_energy_kwh",
        "mean_energy_kwh",
        "mean_duration_hours",
        "median_duration_hours",
        "peak_hour_share",
        "weekend_share",
        "mean_start_hour",
        "sites_used",
    ]
    scaled = StandardScaler().fit_transform(data[feature_columns])
    coordinates = PCA(n_components=2, random_state=42).fit_transform(scaled)
    data["pca_1"] = coordinates[:, 0]
    data["pca_2"] = coordinates[:, 1]

    colors = {
        "frequent": "#d1495b",
        "occasional": "#00798c",
        "peak_hour": "#edae49",
        "long_duration": "#30638e",
    }
    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for label, group in data.groupby("behavior_label"):
        axis.scatter(
            group["pca_1"],
            group["pca_2"],
            s=42,
            alpha=0.78,
            color=colors.get(label, "#555555"),
            label=f"{label} ({len(group)})",
            edgecolors="white",
            linewidths=0.35,
        )
        axis.scatter(
            group["pca_1"].mean(),
            group["pca_2"].mean(),
            marker="X",
            s=180,
            color=colors.get(label, "#555555"),
            edgecolors="black",
            linewidths=0.8,
        )
        axis.annotate(
            label,
            (group["pca_1"].mean(), group["pca_2"].mean()),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
        )

    axis.set_title("ACN Charging-Behavior Segments", fontsize=16, weight="bold")
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.grid(alpha=0.2)
    axis.legend(title="Behavior cluster", frameon=True)
    fig.savefig(OUTPUT_FILE, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()