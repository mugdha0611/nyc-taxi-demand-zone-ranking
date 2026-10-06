from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "zone_hour_timeseries.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

TEST_WINDOW_DAYS = 7
TOP_K_VALUES = [5, 10, 20]

# Same evaluation windows used conceptually in the demand
# forecasting experiment.
FOLD_STARTS = [
    "2025-02-03",
    "2025-02-17",
    "2025-03-03",
    "2025-03-17",
]


# ============================================================
# Metrics
# ============================================================

def dcg(relevances):
    """
    Discounted Cumulative Gain.

    Relevance is actual future gross passenger spend.
    """
    relevances = np.asarray(relevances, dtype=float)

    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(relevances) + 2))

    return np.sum(relevances / discounts)


def ndcg_at_k(predicted_zones, actual_relevance, k):
    """
    NDCG@K.

    predicted_zones:
        Zones ranked by the baseline.

    actual_relevance:
        Dictionary:
            zone -> actual gross passenger spend
        for the target hour.
    """

    predicted_top_k = predicted_zones[:k]

    predicted_relevances = [
        actual_relevance.get(zone, 0.0)
        for zone in predicted_top_k
    ]

    ideal_relevances = sorted(
        actual_relevance.values(),
        reverse=True
    )[:k]

    ideal_dcg = dcg(ideal_relevances)

    if ideal_dcg == 0:
        return np.nan

    return dcg(predicted_relevances) / ideal_dcg


def hit_rate_at_k(predicted_zones, actual_relevance, k):
    """
    Hit Rate@K.

    A hit occurs if at least one predicted top-K zone
    is also in the actual top-K zones by gross spend.
    """

    predicted_top_k = set(predicted_zones[:k])

    actual_top_k = set(
        sorted(
            actual_relevance,
            key=actual_relevance.get,
            reverse=True
        )[:k]
    )

    if not actual_top_k:
        return np.nan

    return float(bool(predicted_top_k & actual_top_k))

def precision_at_k(predicted_zones, actual_relevance, k):
    """
    Precision@K.

    Relevant zones are defined as the actual top-K zones
    by gross passenger spend during the target hour.
    """

    predicted_top_k = set(predicted_zones[:k])

    actual_top_k = set(
        sorted(
            actual_relevance,
            key=actual_relevance.get,
            reverse=True
        )[:k]
    )

    if not predicted_top_k:
        return np.nan

    return len(predicted_top_k & actual_top_k) / k
# ============================================================
# Load data
# ============================================================

print(f"Loading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

df["pickup_date"] = pd.to_datetime(df["pickup_date"])
df["pickup_hour"] = df["pickup_hour"].astype(int)
df["PULocationID"] = df["PULocationID"].astype(int)

df["dow"] = df["pickup_date"].dt.dayofweek
df["is_weekend"] = df["dow"] >= 5

df = df.sort_values(
    ["pickup_date", "pickup_hour", "PULocationID"]
).reset_index(drop=True)

print(f"Loaded shape: {df.shape}")
print(
    f"Date range: "
    f"{df['pickup_date'].min().date()} "
    f"to "
    f"{df['pickup_date'].max().date()}"
)

print(f"Unique zones: {df['PULocationID'].nunique()}")


# ============================================================
# Evaluation
# ============================================================

all_metrics = []
all_predictions = []


for fold_number, fold_start_str in enumerate(FOLD_STARTS, start=1):

    fold_start = pd.Timestamp(fold_start_str)
    fold_end = fold_start + pd.Timedelta(days=TEST_WINDOW_DAYS)

    train = df[df["pickup_date"] < fold_start].copy()

    test = df[
        (df["pickup_date"] >= fold_start)
        & (df["pickup_date"] < fold_end)
    ].copy()

    print("\n" + "=" * 70)
    print(f"FOLD {fold_number}")
    print(f"Train: before {fold_start.date()}")
    print(
        f"Test:  {fold_start.date()} "
        f"to {(fold_end - pd.Timedelta(days=1)).date()}"
    )
    print(f"Training rows: {len(train):,}")
    print(f"Testing rows:  {len(test):,}")

    if train.empty or test.empty:
        print("Skipping empty fold.")
        continue


    # --------------------------------------------------------
    # Baseline 1: Global popularity
    # --------------------------------------------------------

    global_popularity = (
        train.groupby("PULocationID")["trips"]
        .sum()
        .sort_values(ascending=False)
    )

    global_ranking = global_popularity.index.tolist()


    # --------------------------------------------------------
    # Baseline 2:
    # Historical zone × hour × day-of-week popularity
    # --------------------------------------------------------

    historical_profile = (
        train.groupby(
            ["dow", "pickup_hour", "PULocationID"]
        )["trips"]
        .mean()
        .reset_index()
    )

    # Used as a fallback when a zone has no observations for
    # a particular day-of-week/hour combination.
    zone_fallback = (
        train.groupby("PULocationID")["trips"]
        .mean()
    )


    # --------------------------------------------------------
    # Evaluate each test hour
    # --------------------------------------------------------

    for (test_date, test_hour), test_hour_df in test.groupby(
        ["pickup_date", "pickup_hour"]
    ):

        dow = test_date.dayofweek

        # ----------------------------------------------------
        # Actual future relevance
        #
        # Gross passenger spend is the ground-truth relevance.
        # Zones absent from this hour effectively have zero
        # observed spend.
        # ----------------------------------------------------

        actual_relevance = (
            test_hour_df
            .groupby("PULocationID")["gross_passenger_spend"]
            .sum()
            .to_dict()
        )


        # ----------------------------------------------------
        # Candidate zones
        #
        # We only recommend zones that existed in the training
        # period. This prevents future information leakage.
        # ----------------------------------------------------

        candidate_zones = set(train["PULocationID"].unique())


        # ----------------------------------------------------
        # BASELINE 1 — GLOBAL POPULARITY
        # ----------------------------------------------------

        ranking_global = [
            zone
            for zone in global_ranking
            if zone in candidate_zones
        ]


        # ----------------------------------------------------
        # BASELINE 2 — ZONE × HOUR × DAY-OF-WEEK
        # ----------------------------------------------------

        profile_slice = historical_profile[
            (historical_profile["dow"] == dow)
            & (historical_profile["pickup_hour"] == test_hour)
        ].copy()

        profile_scores = dict(
            zip(
                profile_slice["PULocationID"],
                profile_slice["trips"]
            )
        )

        # Use zone-wide historical average as fallback.
        profile_scores = {
            zone: profile_scores.get(
                zone,
                zone_fallback.get(zone, 0.0)
            )
            for zone in candidate_zones
        }

        ranking_profile = sorted(
            candidate_zones,
            key=lambda zone: profile_scores.get(zone, 0.0),
            reverse=True
        )


        # ----------------------------------------------------
        # BASELINE 3 — SAME HOUR LAST WEEK
        # ----------------------------------------------------

        previous_week_date = (
            test_date - pd.Timedelta(days=7)
        )

        previous_week = train[
            (train["pickup_date"] == previous_week_date)
            & (train["pickup_hour"] == test_hour)
        ]

        previous_week_scores = dict(
            zip(
                previous_week["PULocationID"],
                previous_week["trips"]
            )
        )
    
        ranking_last_week = sorted(
            candidate_zones,
            key=lambda zone: previous_week_scores.get(zone, 0.0),
            reverse=True
        )

        # --------------------------------------------------------
        # BASELINE 4 — TRAILING 4-WEEK MATCHING-HOUR AVERAGE
        # --------------------------------------------------------

        matching_scores = {}

        for weeks_back in range(1, 5):
        
            historical_date = (
                test_date - pd.Timedelta(days=7 * weeks_back)
            )

            historical_hour = train[
                (train["pickup_date"] == historical_date)
                & (train["pickup_hour"] == test_hour)
            ]

            for _, row in historical_hour.iterrows():
            
                zone = row["PULocationID"]

                if zone not in matching_scores:
                    matching_scores[zone] = []

                matching_scores[zone].append(row["trips"])


        trailing_4_week_scores = {
            zone: np.mean(
                matching_scores.get(zone, [0.0])
            )
            for zone in candidate_zones
        }

        ranking_trailing_4_week = sorted(
            candidate_zones,
            key=lambda zone: trailing_4_week_scores.get(zone, 0.0),
            reverse=True
        )

        rankings = {
            "global_popularity": ranking_global,
            "zone_hour_profile": ranking_profile,
            "same_hour_last_week": ranking_last_week,
            "trailing_4_week_profile": ranking_trailing_4_week,
            }
        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        for baseline_name, ranking in rankings.items():

            metric_row = {
                "fold": fold_number,
                "test_date": test_date.date(),
                "pickup_hour": test_hour,
                "day_of_week": test_date.day_name(),
                "baseline": baseline_name,
            }

            for k in TOP_K_VALUES:

                metric_row[f"ndcg_at_{k}"] = ndcg_at_k(
                    ranking,
                    actual_relevance,
                    k
                )

                metric_row[f"hit_rate_at_{k}"] = hit_rate_at_k(
                    ranking,
                    actual_relevance,
                    k
                )

                metric_row[f"precision_at_{k}"] = precision_at_k(
                    ranking,
                    actual_relevance,
                    k
                )

            all_metrics.append(metric_row)


        # ----------------------------------------------------
        # Save top-20 predictions for inspection
        # ----------------------------------------------------

        actual_top_20 = sorted(
            actual_relevance,
            key=actual_relevance.get,
            reverse=True
        )[:20]

        actual_rank = {
            zone: rank + 1
            for rank, zone in enumerate(actual_top_20)
        }

        zone_names = (
            test_hour_df[
                ["PULocationID", "pickup_zone", "pickup_borough"]
            ]
            .drop_duplicates("PULocationID")
            .set_index("PULocationID")
        )

        for baseline_name, ranking in rankings.items():

            for rank, zone in enumerate(ranking[:20], start=1):

                all_predictions.append({
                    "fold": fold_number,
                    "test_date": test_date.date(),
                    "pickup_hour": test_hour,
                    "day_of_week": test_date.day_name(),
                    "baseline": baseline_name,
                    "rank": rank,
                    "PULocationID": zone,
                    "pickup_zone": (
                        zone_names.loc[zone, "pickup_zone"]
                        if zone in zone_names.index
                        else "Unknown"
                    ),
                    "pickup_borough": (
                        zone_names.loc[zone, "pickup_borough"]
                        if zone in zone_names.index
                        else "Unknown"
                    ),
                    "actual_gross_passenger_spend": (
                        actual_relevance.get(zone, 0.0)
                    ),
                    "actual_rank_top20": actual_rank.get(zone, np.nan),
                })


# ============================================================
# Save outputs
# ============================================================

metrics_df = pd.DataFrame(all_metrics)
predictions_df = pd.DataFrame(all_predictions)

metrics_path = OUTPUT_DIR / "ranking_baseline_metrics.csv"
predictions_path = OUTPUT_DIR / "ranking_baseline_predictions.csv"

metrics_df.to_csv(metrics_path, index=False)
predictions_df.to_csv(predictions_path, index=False)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

summary = (
    metrics_df
    .groupby("baseline")
    [
        [
            "ndcg_at_5",
            "ndcg_at_10",
            "ndcg_at_20",
            "precision_at_10",
            "hit_rate_at_10",
        ]
    ]
    .mean()
    .sort_values("ndcg_at_10", ascending=False)
)

print(summary.to_string())

print("\nFold-level NDCG@10:")
print(
    metrics_df
    .groupby(["baseline", "fold"])["ndcg_at_10"]
    .mean()
    .unstack()
    .to_string()
)

print(f"\nSaved metrics: {metrics_path}")
print(f"Saved predictions: {predictions_path}")