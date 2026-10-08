from pathlib import Path
import pandas as pd

# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml"
    / "train_base.csv"
)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)

target = train["target_demand_kw"]

print("=" * 60)
print("TRAINING TARGET DISTRIBUTION")
print("=" * 60)

print("Training rows:", len(train))
print("Training columns:", len(train.columns))

# ============================================================
# ZERO VS NON-ZERO
# ============================================================

zero_count = (target == 0).sum()
positive_count = (target > 0).sum()

print("\nZero-demand samples:", zero_count)
print("Positive-demand samples:", positive_count)

print(
    "Zero-demand percentage:",
    round(zero_count / len(target) * 100, 2),
    "%"
)

print(
    "Positive-demand percentage:",
    round(positive_count / len(target) * 100, 2),
    "%"
)

# ============================================================
# ALL TARGET STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("ALL TARGET STATISTICS")
print("=" * 60)

print(target.describe())

# ============================================================
# POSITIVE-DEMAND STATISTICS
# ============================================================

positive = target[target > 0]

print("\n" + "=" * 60)
print("POSITIVE-DEMAND STATISTICS")
print("=" * 60)

print(positive.describe())

# ============================================================
# POSITIVE TARGET QUANTILES
# ============================================================

print("\n" + "=" * 60)
print("POSITIVE-DEMAND QUANTILES")
print("=" * 60)

for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
    value = positive.quantile(q)

    print(
        f"{int(q * 100)}th percentile:",
        round(value, 4),
        "kW"
    )

# ============================================================
# COUNTS ABOVE POSSIBLE HIGH-DEMAND THRESHOLDS
# ============================================================

print("\n" + "=" * 60)
print("HIGH-DEMAND COUNTS")
print("=" * 60)

for threshold in [5, 10, 15, 20, 25, 30, 40, 50]:

    count = (target >= threshold).sum()

    print(
        f">= {threshold:>2} kW:",
        count,
        "samples",
        f"({count / len(target) * 100:.3f}%)"
    )

# ============================================================
# MAXIMUM
# ============================================================

print("\nMaximum target demand:", target.max(), "kW")