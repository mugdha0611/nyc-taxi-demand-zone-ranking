from pathlib import Path
import pandas as pd
import numpy as np
import re

# ============================================================
# PATHS
# ============================================================

# validate_ranking.py is inside src/
ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "outputs"

SCRIPT = ROOT / "src" / "ranking_evaluation_final.py"

# SCRIPT = ROOT / "ranking_evaluation_final.py"
LEARNED_METRICS = OUT / "ranking_learned_metrics.csv"
LEARNED_PREDS = OUT / "ranking_learned_predictions.csv"
BASELINE_METRICS = OUT / "ranking_baseline_metrics.csv"


# ============================================================
# HELPER
# ============================================================

passed_all = True

def check(name, condition, detail=""):
    global passed_all

    if condition:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")
        passed_all = False

    if detail:
        print(f"       {detail}")


print("=" * 65)
print("LEARNED RANKER — QUICK VALIDATION")
print("=" * 65)


# ============================================================
# 1. FILE CHECK
# ============================================================

print("\n1. FILES")

for f in [
    SCRIPT,
    LEARNED_METRICS,
    LEARNED_PREDS,
    BASELINE_METRICS,
]:
    check(
        f.name,
        f.exists(),
        str(f)
    )


# Stop if files are missing
if not passed_all:
    raise SystemExit("\nMissing required files. Fix this first.")


# ============================================================
# 2. LOAD RESULTS
# ============================================================

learned = pd.read_csv(LEARNED_METRICS)
preds = pd.read_csv(LEARNED_PREDS)
baseline = pd.read_csv(BASELINE_METRICS)

print("\n2. SAVED RESULTS")

learned = learned[
    learned["baseline"] == "learned_ranker"
].copy()

print(f"Learned metric rows: {len(learned):,}")
print(f"Learned prediction rows: {len(preds):,}")


# ============================================================
# 3. FOUR FOLDS
# ============================================================

print("\n3. FOLD CHECK")

folds = sorted(learned["fold"].unique())

print("Folds:", folds)

check(
    "Exactly four learned-ranker folds",
    folds == [1, 2, 3, 4],
    f"Found {folds}"
)


# ============================================================
# 4. REPORTED METRICS
# ============================================================

print("\n4. LEARNED-RANKER RESULTS")

ndcg10 = learned["ndcg_at_10"].mean()
precision10 = learned["precision_at_10"].mean()

print(f"NDCG@10:       {ndcg10:.4f}")
print(f"Precision@10:  {precision10:.4f}")

print("\nFold NDCG@10:")

fold_scores = (
    learned
    .groupby("fold")["ndcg_at_10"]
    .mean()
)

for fold, score in fold_scores.items():
    print(f"  Fold {int(fold)}: {score:.4f}")


# ============================================================
# 5. TEMPORAL LEAKAGE — SOURCE CHECK
# ============================================================

print("\n5. TEMPORAL LEAKAGE")

source = SCRIPT.read_text(encoding="utf-8")
# Normalize whitespace so formatting/indentation cannot
# cause a false failure.
normalized_source = re.sub(r"\s+", " ", source)

temporal_check = (
    "history_before = train[" in normalized_source
    and 'train["pickup_date"] < historical_date' in normalized_source
    and "history=history_before" in normalized_source
)

check(
    "Historical features use dates before target date",
    temporal_check,
    (
        "Detected history restricted to pickup_date < "
        "historical_date before feature construction."
        if temporal_check
        else "Temporal restriction not detected."
    )
)

# ============================================================
# 6. TARGET NOT IN MODEL FEATURES
# ============================================================

print("\n6. TARGET LEAKAGE")

feature_match = re.search(
    r'MODEL_FEATURES\s*=\s*\[(.*?)\]',
    source,
    flags=re.DOTALL
)

if feature_match:

    feature_text = feature_match.group(1)

    forbidden = [
        "target_gross_spend",
        "gross_passenger_spend",
        "actual_gross_passenger_spend",
    ]

    leaked = [
        x for x in forbidden
        if x in feature_text
    ]

    check(
        "Target is not directly included in MODEL_FEATURES",
        len(leaked) == 0,
        f"Found: {leaked}" if leaked else "No target variable found."
    )

else:
    print("[WARN] MODEL_FEATURES block not found automatically.")


# ============================================================
# 7. TRAIN / TEST SPLIT CHECK
# ============================================================

print("\n7. TRAIN / TEST SPLIT")

train_split = bool(
    re.search(
        r'train\s*=\s*df\['
        r'\s*df\["pickup_date"\]\s*<\s*fold_start',
        source,
        flags=re.DOTALL
    )
)

test_split = bool(
    re.search(
        r'test\s*=\s*df\[',
        source
    )
)

check(
    "Training data ends before fold start",
    train_split
)

check(
    "Explicit test dataset exists",
    test_split
)


# ============================================================
# 8. PREDICTION SANITY
# ============================================================

print("\n8. PREDICTION SANITY")

required = [
    "fold",
    "test_date",
    "pickup_hour",
    "PULocationID",
    "predicted_gross_passenger_spend",
    "actual_gross_passenger_spend",
]

missing = [
    c for c in required
    if c not in preds.columns
]

check(
    "Required prediction columns exist",
    len(missing) == 0,
    f"Missing: {missing}" if missing else ""
)

if "predicted_gross_passenger_spend" in preds:

    p = preds["predicted_gross_passenger_spend"]

    check(
        "No NaN predictions",
        p.isna().sum() == 0,
        f"NaN count: {p.isna().sum()}"
    )

    check(
        "All predictions are finite",
        np.isfinite(p).all()
    )


# ============================================================
# 9. LEARNED VS STRONGEST BASELINE
# ============================================================

print("\n9. LEARNED VS BASELINE")

base = (
    baseline
    .groupby("baseline")["ndcg_at_10"]
    .mean()
    .sort_values(ascending=False)
)

best_baseline = base.index[0]
best_score = base.iloc[0]

print(f"Strongest baseline: {best_baseline}")
print(f"Baseline NDCG@10:   {best_score:.4f}")
print(f"Learned NDCG@10:    {ndcg10:.4f}")
print(f"Absolute gain:      {ndcg10 - best_score:+.4f}")

check(
    "Learned ranker beats strongest baseline",
    ndcg10 > best_score
)


# ============================================================
# 10. FINAL
# ============================================================

print("\n" + "=" * 65)

if passed_all:
    print("OVERALL: PASS")
    print("=" * 65)
    print(
        "\nNo obvious leakage or evaluation-structure problem "
        "was detected."
    )
else:
    print("OVERALL: REVIEW")
    print("=" * 65)
    print(
        "\nAt least one check failed. Do not use the learned "
        "ranker result on your resume yet."
    )