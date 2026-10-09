import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "behavior",
    "user_behavior_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "behavior"
)

CLUSTER_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "user_behavior_clusters.csv"
)

SUMMARY_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "behavior_cluster_summary.csv"
)


# ============================================================
# LOAD USER BEHAVIOR DATA
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS CHARGING-BEHAVIOR CLUSTERING")
print("=" * 70)

print("\nLoading user behavior dataset...")

df = pd.read_csv(INPUT_PATH)

print(
    f"Users loaded: {len(df):,}"
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# SELECT BEHAVIORAL FEATURES
# ============================================================

# These features directly describe charging behavior.
#
# session_count:
#   How frequently the user charges.
#
# avg_session_duration_min:
#   Typical charging duration.
#
# avg_energy_kwh:
#   Typical energy consumed per session.
#
# avg_arrival_hour:
#   Typical time of day when charging starts.
#
# peak_hour_percentage:
#   Percentage of sessions during peak charging periods.
#
# weekend_percentage:
#   Percentage of sessions occurring on weekends.

feature_columns = [
    "session_count",
    "avg_session_duration_min",
    "avg_energy_kwh",
    "avg_arrival_hour",
    "peak_hour_percentage",
    "weekend_percentage"
]

X = df[feature_columns].copy()


print("\nFeatures used for clustering:")

for feature in feature_columns:
    print(f" - {feature}")


# ============================================================
# CHECK FOR MISSING VALUES
# ============================================================

if X.isnull().any().any():

    missing_columns = X.columns[
        X.isnull().any()
    ].tolist()

    raise ValueError(
        "Missing values found in clustering features: "
        f"{missing_columns}"
    )


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

print("\nStandardizing behavioral features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(
    "Features standardized successfully."
)


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

N_CLUSTERS = 4

print(
    f"\nRunning K-Means with "
    f"{N_CLUSTERS} clusters..."
)

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=20
)

df["cluster"] = kmeans.fit_predict(
    X_scaled
)

# ============================================================
# ASSIGN BEHAVIOR LABELS
# ============================================================

behavior_labels = {
    0: "Frequent Peak-Hour Users",
    1: "High-Energy Long-Duration Users",
    2: "Occasional Short-Duration Users",
    3: "Weekend Users"
}

df["behavior_label"] = df["cluster"].map(
    behavior_labels
)

# ============================================================
# SILHOUETTE SCORE
# ============================================================

silhouette = silhouette_score(
    X_scaled,
    df["cluster"]
)

print(
    f"\nSilhouette Score: "
    f"{silhouette:.4f}"
)


# ============================================================
# CLUSTER COUNTS
# ============================================================

print("\n" + "=" * 70)
print("CLUSTER COUNTS")
print("=" * 70)

cluster_counts = (
    df["cluster"]
    .value_counts()
    .sort_index()
)

for cluster, count in cluster_counts.items():

    percentage = (
        count / len(df)
    ) * 100

    print(
        f"Cluster {cluster}: "
        f"{count} users "
        f"({percentage:.2f}%)"
    )


# ============================================================
# CLUSTER BEHAVIOR SUMMARY
# ============================================================

cluster_summary = (
    df.groupby("cluster")[feature_columns]
    .mean()
    .round(3)
)

# Add number of users in each cluster
cluster_summary.insert(
    0,
    "user_count",
    df.groupby("cluster").size()
)

# Add percentage of total users
cluster_summary["user_percentage"] = (
    cluster_summary["user_count"]
    / len(df)
    * 100
).round(2)


# ============================================================
# DISPLAY CLUSTER SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLUSTER BEHAVIOR SUMMARY")
print("=" * 70)

print(
    cluster_summary.to_string()
)


# ============================================================
# SAVE USER-LEVEL CLUSTER DATA
# ============================================================

df.to_csv(
    CLUSTER_OUTPUT_PATH,
    index=False
)


# ============================================================
# SAVE CLUSTER SUMMARY
# ============================================================

cluster_summary.to_csv(
    SUMMARY_OUTPUT_PATH
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"\nUser clusters:"
    f"\n{CLUSTER_OUTPUT_PATH}"
)

print(
    f"\nCluster summary:"
    f"\n{SUMMARY_OUTPUT_PATH}"
)

print("\nClustering complete.")