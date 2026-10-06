from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

def get_time_block(hour):
    if 0 <= hour <= 5:
        return "Overnight"
    elif 6 <= hour <= 9:
        return "Morning"
    elif 10 <= hour <= 15:
        return "Midday"
    elif 16 <= hour <= 20:
        return "Evening"
    else:
        return "Late evening"

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
SUPPORT_THRESHOLDS = [500, 1000]

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
            actual_relevance.keys(),
            key=lambda zone: (
                -actual_relevance.get(zone, 0.0),
                zone
            )
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
            actual_relevance.keys(),
            key=lambda zone: (
                -actual_relevance.get(zone, 0.0),
                zone
            )
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

df["time_block"] = df["pickup_hour"].apply(get_time_block)
df["day_type"] = np.where(
    df["is_weekend"],
    "Weekend",
    "Weekday"
)
df["is_airport"] = df["pickup_zone"].str.contains(
    "airport",
    case=False,
    na=False
)

print("\nAirport zones:")
print(
    df.loc[df["is_airport"], "pickup_zone"]
    .drop_duplicates()
    .sort_values()
    .to_string(index=False)
)

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
print("\nTime blocks:")
print(df[["pickup_hour", "time_block"]].drop_duplicates().sort_values("pickup_hour").to_string(index=False))
def build_zone_hour_features(
    history,
    target_date,
    target_hour,
):
    """
    Build leakage-safe features for every zone for a target
    date/hour using only observations strictly before target_date.
    """

    zones = (
        history[
            [
                "PULocationID",
                "pickup_zone",
                "pickup_borough",
                "is_airport",
            ]
        ]
        .drop_duplicates("PULocationID")
        .set_index("PULocationID")
    )

    candidate_zones = zones.index.tolist()

    target_timestamp = pd.Timestamp(target_date)

    dow = target_timestamp.dayofweek
    is_weekend = dow >= 5

    rows = []

    # --------------------------------------------------------
    # Zone-wide historical statistics
    # --------------------------------------------------------

    zone_stats = (
    history
    .groupby("PULocationID")
    .agg(
        historical_trips=("trips", "sum"),
        historical_gross_spend=(
            "gross_passenger_spend",
            "sum",
        ),
        weighted_duration_sum=(
            "avg_trip_duration_min",
            lambda x: 0.0,
        ),
        weighted_distance_sum=(
            "avg_trip_distance_miles",
            lambda x: 0.0,
        ),
    )
)

    # Reconstruct weighted duration and distance
    history_with_weights = history.copy()

    history_with_weights["duration_weighted"] = (
        history_with_weights["avg_trip_duration_min"]
        * history_with_weights["trips"]
    )

    history_with_weights["distance_weighted"] = (
        history_with_weights["avg_trip_distance_miles"]
        * history_with_weights["trips"]
    )

    weighted_stats = (
        history_with_weights
        .groupby("PULocationID")
        .agg(
            total_duration_weighted=(
                "duration_weighted",
                "sum",
            ),
            total_distance_weighted=(
                "distance_weighted",
                "sum",
            ),
        )
    )

    zone_stats = zone_stats.drop(
        columns=[
            "weighted_duration_sum",
            "weighted_distance_sum",
        ]
    ).join(weighted_stats)

    zone_stats["historical_avg_spend_per_trip"] = (
        zone_stats["historical_gross_spend"]
        / zone_stats["historical_trips"]
    )

    zone_stats["historical_avg_duration"] = (
        zone_stats["total_duration_weighted"]
        / zone_stats["historical_trips"]
    )

    zone_stats["historical_distance"] = (
        zone_stats["total_distance_weighted"]
        / zone_stats["historical_trips"]
    )

    zone_stats = zone_stats[
        [
            "historical_trips",
            "historical_gross_spend",
            "historical_avg_spend_per_trip",
            "historical_avg_duration",
            "historical_distance",
        ]
    ]

    # --------------------------------------------------------
    # Historical zone × DOW × hour demand
    # --------------------------------------------------------

    matching_profile = (
        history[
            (history["dow"] == dow)
            & (history["pickup_hour"] == target_hour)
        ]
        .groupby("PULocationID")
        .agg(
            matching_dow_hour_trips=("trips", "sum"),
            matching_dow_hour_spend=(
                "gross_passenger_spend",
                "sum",
            ),
        )
    )

    # --------------------------------------------------------
    # Same hour last week
    # --------------------------------------------------------

    last_week_date = (
        target_timestamp - pd.Timedelta(days=7)
    )

    last_week = history[
        (history["pickup_date"] == last_week_date)
        & (history["pickup_hour"] == target_hour)
    ]

    last_week_features = (
        last_week
        .set_index("PULocationID")
        .reindex(candidate_zones)
    )

    # --------------------------------------------------------
    # Trailing four matching hours
    # --------------------------------------------------------

    trailing_rows = []

    for weeks_back in range(1, 5):

        historical_date = (
            target_timestamp
            - pd.Timedelta(days=7 * weeks_back)
        )

        matching = history[
            (history["pickup_date"] == historical_date)
            & (history["pickup_hour"] == target_hour)
        ][
            [
                "PULocationID",
                "trips",
                "gross_passenger_spend",
            ]
        ].copy()

        matching["weeks_back"] = weeks_back

        trailing_rows.append(matching)

    if trailing_rows:

        trailing = pd.concat(
            trailing_rows,
            ignore_index=True,
        )

        trailing_features = (
            trailing
            .groupby("PULocationID")
            .agg(
                trailing_4_week_trips=(
                    "trips",
                    "mean",
                ),
                trailing_4_week_spend=(
                    "gross_passenger_spend",
                    "mean",
                ),
            )
        )

    else:

        trailing_features = pd.DataFrame(
            index=candidate_zones
        )

    # --------------------------------------------------------
    # Assemble
    # --------------------------------------------------------

    for zone in candidate_zones:

        row = {
            "PULocationID": zone,
            "hour": target_hour,
            "day_of_week": dow,
            "is_weekend": int(is_weekend),
            "is_airport": int(
                zones.loc[zone, "is_airport"]
            ),
        }

        if zone in zone_stats.index:

            row.update(
                zone_stats.loc[zone].to_dict()
            )

        if zone in matching_profile.index:

            row.update(
                matching_profile.loc[zone].to_dict()
            )

        if zone in last_week_features.index:

            row["same_hour_last_week_trips"] = (
                last_week_features.loc[
                    zone, "trips"
                ]
                if pd.notna(
                    last_week_features.loc[
                        zone, "trips"
                    ]
                )
                else 0.0
            )

        else:

            row["same_hour_last_week_trips"] = 0.0

        if zone in trailing_features.index:

            row.update(
                trailing_features.loc[zone].to_dict()
            )

        rows.append(row)

    features = pd.DataFrame(rows)

    numeric_cols = [
        col
        for col in features.columns
        if col != "PULocationID"
    ]

    features[numeric_cols] = (
        features[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return features

# ---------------------------------------------------------
# Learned ranker feature sanity check
# ---------------------------------------------------------

feature_test_date = pd.Timestamp("2025-02-03")
feature_test_hour = 12

feature_test = build_zone_hour_features(
    history=df[df["pickup_date"] < feature_test_date],
    target_date=feature_test_date,
    target_hour=feature_test_hour,
)

print("\nLEARNED RANKER FEATURE CHECK")
print("Shape:", feature_test.shape)
print("\nColumns:")
print(feature_test.columns.tolist())

print("\nSample:")
print(feature_test.head())

print("\nMissing values:")
print(feature_test.isna().sum())

print("\nNumeric summary:")
print(feature_test.describe().T)

# ============================================================
# Evaluation
# ============================================================

all_metrics = []
all_predictions = []
airport_metrics = []
support_metrics = []

learned_metrics = []
learned_predictions = []


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
    )

    global_ranking = sorted(
        global_popularity.index.tolist(),
        key=lambda zone: (
            -global_popularity.get(zone, 0.0),
            zone,
        )
    )

    # --------------------------------------------------------
    # Baseline 2:
    # Historical zone × hour × day-of-week popularity
    # --------------------------------------------------------

    historical_slot_counts = (
        train[
            ["pickup_date", "dow", "pickup_hour"]
        ]
        .drop_duplicates()
        .groupby(
            ["dow", "pickup_hour"]
        )
        .size()
        .rename("n_historical_slots")
        .reset_index()
    )

    historical_profile = (
        train.groupby(
            ["dow", "pickup_hour", "PULocationID"],
            as_index=False,
        )["trips"]
        .sum()
        .merge(
            historical_slot_counts,
            on=["dow", "pickup_hour"],
            how="left",
        )
    )

    historical_profile["avg_trips"] = (
        historical_profile["trips"]
        / historical_profile["n_historical_slots"]
    )

    assert historical_profile["avg_trips"].notna().all()
    assert (historical_profile["avg_trips"] >= 0).all()

    # --------------------------------------------------------
    # Zone-wide fallback
    # --------------------------------------------------------

    n_train_date_hours = (
        train[
            ["pickup_date", "pickup_hour"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    zone_fallback = (
        train.groupby("PULocationID")["trips"]
        .sum()
        / n_train_date_hours
    )

    assert zone_fallback.notna().all()
    assert (zone_fallback >= 0).all()

    # --------------------------------------------------------
    # Evaluate each test hour
    # --------------------------------------------------------

    for (test_date, test_hour), test_hour_df in test.groupby(
        ["pickup_date", "pickup_hour"]
    ):

        dow = test_date.dayofweek

        # ----------------------------------------------------
        # Actual future relevance
        # ----------------------------------------------------

        actual_relevance = (
            test_hour_df
            .groupby("PULocationID")[
                "gross_passenger_spend"
            ]
            .sum()
            .to_dict()
        )

        # ----------------------------------------------------
        # Candidate zones
        # ----------------------------------------------------

        candidate_zones = set(
            train["PULocationID"].unique()
        )

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
            & (
                historical_profile["pickup_hour"]
                == test_hour
            )
        ].copy()

        profile_scores = dict(
            zip(
                profile_slice["PULocationID"],
                profile_slice["avg_trips"],
            )
        )

        profile_scores = {
            zone: profile_scores.get(
                zone,
                zone_fallback.get(zone, 0.0),
            )
            for zone in candidate_zones
        }

        ranking_profile = sorted(
            candidate_zones,
            key=lambda zone: (
                -profile_scores.get(zone, 0.0),
                zone,
            ),
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
                previous_week["trips"],
            )
        )

        ranking_last_week = sorted(
            candidate_zones,
            key=lambda zone: (
                -previous_week_scores.get(zone, 0.0),
                zone,
            ),
        )

        # --------------------------------------------------------
        # BASELINE 4 — TRAILING 4-WEEK MATCHING-HOUR AVERAGE
        # --------------------------------------------------------

        matching_scores = {}

        for weeks_back in range(1, 5):

            historical_date = (
                test_date
                - pd.Timedelta(days=7 * weeks_back)
            )

            historical_hour = train[
                (train["pickup_date"] == historical_date)
                & (train["pickup_hour"] == test_hour)
            ]

            for _, row in historical_hour.iterrows():

                zone = row["PULocationID"]

                if zone not in matching_scores:
                    matching_scores[zone] = []

                matching_scores[zone].append(
                    row["trips"]
                )

        trailing_4_week_scores = {
            zone: np.mean(
                matching_scores.get(zone, [0.0])
            )
            for zone in candidate_zones
        }

        ranking_trailing_4_week = sorted(
            candidate_zones,
            key=lambda zone: (
                -trailing_4_week_scores.get(
                    zone,
                    0.0,
                ),
                zone,
            ),
        )

        rankings = {
            "global_popularity": ranking_global,
            "zone_hour_profile": ranking_profile,
            "same_hour_last_week": ranking_last_week,
            "trailing_4_week_profile": ranking_trailing_4_week,
        }

        # ====================================================
        # LEARNED RANKER
        # ====================================================

        learned_features = build_zone_hour_features(
            history=train,
            target_date=test_date,
            target_hour=test_hour,
        )

        # ----------------------------------------------------
        # Build leakage-safe training examples
        #
        # Each historical date/hour becomes one training
        # example for every zone that existed before that
        # date/hour.
        # ----------------------------------------------------

        training_examples = []

        historical_hours = (
            train[
                ["pickup_date", "pickup_hour"]
            ]
            .drop_duplicates()
            .sort_values(
                ["pickup_date", "pickup_hour"]
            )
        )

        for _, historical_row in historical_hours.iterrows():

            historical_date = historical_row[
                "pickup_date"
            ]

            historical_hour = int(
                historical_row["pickup_hour"]
            )

            # Need enough history for the feature builder.
            history_before = train[
                train["pickup_date"]
                < historical_date
            ]

            if history_before.empty:
                continue

            historical_features = (
                build_zone_hour_features(
                    history=history_before,
                    target_date=historical_date,
                    target_hour=historical_hour,
                )
            )

            historical_target = (
                train[
                    (train["pickup_date"] == historical_date)
                    & (
                        train["pickup_hour"]
                        == historical_hour
                    )
                ]
                .groupby("PULocationID")[
                    "gross_passenger_spend"
                ]
                .sum()
            )

            historical_features[
                "target_gross_spend"
            ] = (
                historical_features["PULocationID"]
                .map(historical_target)
                .fillna(0.0)
            )

            training_examples.append(
                historical_features
            )

        if training_examples:

            learned_training = pd.concat(
                training_examples,
                ignore_index=True,
            )

            feature_columns = [
                col
                for col in learned_features.columns
                if col != "PULocationID"
            ]

            X_train = learned_training[
                feature_columns
            ]

            y_train = learned_training[
                "target_gross_spend"
            ]

            X_test = learned_features[
                feature_columns
            ]

            learned_model = (
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=42,
                )
            )

            learned_model.fit(
                X_train,
                y_train,
            )

            learned_features[
                "predicted_gross_spend"
            ] = learned_model.predict(
                X_test
            )

            ranking_learned = (
                learned_features
                .sort_values(
                    [
                        "predicted_gross_spend",
                        "PULocationID",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
                ["PULocationID"]
                .tolist()
            )

            # ------------------------------------------------
            # Learned-ranker metrics
            # ------------------------------------------------

            learned_metric_row = {
                "fold": fold_number,
                "test_date": test_date.date(),
                "pickup_hour": test_hour,
                "day_of_week": test_date.day_name(),
                "baseline": "learned_ranker",
                "time_block": get_time_block(
                    test_hour
                ),
                "day_type": (
                    "Weekend"
                    if test_date.dayofweek >= 5
                    else "Weekday"
                ),
            }

            for k in TOP_K_VALUES:

                learned_metric_row[
                    f"ndcg_at_{k}"
                ] = ndcg_at_k(
                    ranking_learned,
                    actual_relevance,
                    k,
                )

                learned_metric_row[
                    f"hit_rate_at_{k}"
                ] = hit_rate_at_k(
                    ranking_learned,
                    actual_relevance,
                    k,
                )

                learned_metric_row[
                    f"precision_at_{k}"
                ] = precision_at_k(
                    ranking_learned,
                    actual_relevance,
                    k,
                )

            learned_metrics.append(
                learned_metric_row
            )

            # ------------------------------------------------
            # Save top-20 learned predictions
            # ------------------------------------------------

            zone_names = (
                test_hour_df[
                    [
                        "PULocationID",
                        "pickup_zone",
                        "pickup_borough",
                    ]
                ]
                .drop_duplicates(
                    "PULocationID"
                )
                .set_index("PULocationID")
            )

            actual_top_20 = sorted(
                actual_relevance.keys(),
                key=lambda zone: (
                    -actual_relevance.get(
                        zone,
                        0.0,
                    ),
                    zone,
                ),
            )[:20]

            actual_rank = {
                zone: rank + 1
                for rank, zone in enumerate(
                    actual_top_20
                )
            }

            for rank, zone in enumerate(
                ranking_learned[:20],
                start=1,
            ):

                learned_predictions.append({
                    "fold": fold_number,
                    "test_date": test_date.date(),
                    "pickup_hour": test_hour,
                    "day_of_week": test_date.day_name(),
                    "baseline": "learned_ranker",
                    "rank": rank,
                    "PULocationID": zone,
                    "pickup_zone": (
                        zone_names.loc[
                            zone,
                            "pickup_zone",
                        ]
                        if zone in zone_names.index
                        else "Unknown"
                    ),
                    "pickup_borough": (
                        zone_names.loc[
                            zone,
                            "pickup_borough",
                        ]
                        if zone in zone_names.index
                        else "Unknown"
                    ),
                    "predicted_gross_passenger_spend": (
                        learned_features.loc[
                            learned_features[
                                "PULocationID"
                            ] == zone,
                            "predicted_gross_spend",
                        ].iloc[0]
                    ),
                    "actual_gross_passenger_spend": (
                        actual_relevance.get(
                            zone,
                            0.0,
                        )
                    ),
                    "actual_rank_top20": (
                        actual_rank.get(
                            zone,
                            np.nan,
                        )
                    ),
                })

        # ====================================================
        # SUPPORT THRESHOLD EXPERIMENT
        # ====================================================

        zone_training_support = (
            train
            .groupby("PULocationID")["trips"]
            .sum()
            .to_dict()
        )

        for support_threshold in SUPPORT_THRESHOLDS:

            supported_zones = {
                zone
                for zone, trip_count
                in zone_training_support.items()
                if trip_count >= support_threshold
            }

            support_coverage = (
                len(supported_zones)
                / len(candidate_zones)
                if candidate_zones
                else np.nan
            )

            ranking_global_support = [
                zone
                for zone in ranking_global
                if zone in supported_zones
            ]

            ranking_profile_support = [
                zone
                for zone in ranking_profile
                if zone in supported_zones
            ]

            ranking_last_week_support = [
                zone
                for zone in ranking_last_week
                if zone in supported_zones
            ]

            ranking_trailing_4_week_support = [
                zone
                for zone in ranking_trailing_4_week
                if zone in supported_zones
            ]

            support_rankings = {
                "global_popularity":
                    ranking_global_support,
                "zone_hour_profile":
                    ranking_profile_support,
                "same_hour_last_week":
                    ranking_last_week_support,
                "trailing_4_week_profile":
                    ranking_trailing_4_week_support,
            }

            for baseline_name, ranking in (
                support_rankings.items()
            ):

                support_metric_row = {
                    "fold": fold_number,
                    "test_date": test_date.date(),
                    "pickup_hour": test_hour,
                    "day_of_week": test_date.day_name(),
                    "baseline": baseline_name,
                    "support_threshold":
                        support_threshold,
                    "supported_zone_count":
                        len(supported_zones),
                    "candidate_zone_count":
                        len(candidate_zones),
                    "support_coverage":
                        support_coverage,
                    "time_block":
                        get_time_block(test_hour),
                    "day_type": (
                        "Weekend"
                        if test_date.dayofweek >= 5
                        else "Weekday"
                    ),
                }

                for k in TOP_K_VALUES:

                    support_metric_row[
                        f"ndcg_at_{k}"
                    ] = ndcg_at_k(
                        ranking,
                        actual_relevance,
                        k,
                    )

                    support_metric_row[
                        f"precision_at_{k}"
                    ] = precision_at_k(
                        ranking,
                        actual_relevance,
                        k,
                    )

                support_metrics.append(
                    support_metric_row
                )

        # ====================================================
        # Calculate baseline metrics
        # ====================================================

        for baseline_name, ranking in rankings.items():

            metric_row = {
                "fold": fold_number,
                "test_date": test_date.date(),
                "pickup_hour": test_hour,
                "day_of_week": test_date.day_name(),
                "baseline": baseline_name,
                "time_block":
                    get_time_block(test_hour),
                "day_type": (
                    "Weekend"
                    if test_date.dayofweek >= 5
                    else "Weekday"
                ),
            }

            for k in TOP_K_VALUES:

                metric_row[
                    f"ndcg_at_{k}"
                ] = ndcg_at_k(
                    ranking,
                    actual_relevance,
                    k,
                )

                metric_row[
                    f"hit_rate_at_{k}"
                ] = hit_rate_at_k(
                    ranking,
                    actual_relevance,
                    k,
                )

                metric_row[
                    f"precision_at_{k}"
                ] = precision_at_k(
                    ranking,
                    actual_relevance,
                    k,
                )

            all_metrics.append(
                metric_row
            )

        # ----------------------------------------------------
        # Airport vs non-airport evaluation
        # ----------------------------------------------------

        airport_zone_ids = set(
            df.loc[
                df["is_airport"],
                "PULocationID",
            ].unique()
        )

        for baseline_name, ranking in rankings.items():

            for airport_group, group_zone_ids in {
                "Airport": airport_zone_ids,
                "Non-airport":
                    candidate_zones - airport_zone_ids,
            }.items():

                group_ranking = [
                    zone
                    for zone in ranking
                    if zone in group_zone_ids
                ]

                group_relevance = {
                    zone: relevance
                    for zone, relevance
                    in actual_relevance.items()
                    if zone in group_zone_ids
                }

                airport_metric_row = {
                    "fold": fold_number,
                    "test_date": test_date.date(),
                    "pickup_hour": test_hour,
                    "day_of_week": test_date.day_name(),
                    "baseline": baseline_name,
                    "airport_group": airport_group,
                    "time_block":
                        get_time_block(test_hour),
                    "day_type": (
                        "Weekend"
                        if test_date.dayofweek >= 5
                        else "Weekday"
                    ),
                }

                for k in TOP_K_VALUES:

                    airport_metric_row[
                        f"ndcg_at_{k}"
                    ] = ndcg_at_k(
                        group_ranking,
                        group_relevance,
                        k,
                    )

                    airport_metric_row[
                        f"precision_at_{k}"
                    ] = precision_at_k(
                        group_ranking,
                        group_relevance,
                        k,
                    )

                airport_metrics.append(
                    airport_metric_row
                )

        # ----------------------------------------------------
        # Save top-20 predictions for inspection
        # ----------------------------------------------------

        actual_top_20 = sorted(
            actual_relevance.keys(),
            key=lambda zone: (
                -actual_relevance.get(
                    zone,
                    0.0,
                ),
                zone,
            ),
        )[:20]

        actual_rank = {
            zone: rank + 1
            for rank, zone in enumerate(
                actual_top_20
            )
        }

        zone_names = (
            test_hour_df[
                [
                    "PULocationID",
                    "pickup_zone",
                    "pickup_borough",
                ]
            ]
            .drop_duplicates(
                "PULocationID"
            )
            .set_index("PULocationID")
        )

        for baseline_name, ranking in (
            rankings.items()
        ):

            for rank, zone in enumerate(
                ranking[:20],
                start=1,
            ):

                all_predictions.append({
                    "fold": fold_number,
                    "test_date": test_date.date(),
                    "pickup_hour": test_hour,
                    "day_of_week": test_date.day_name(),
                    "baseline": baseline_name,
                    "rank": rank,
                    "PULocationID": zone,
                    "pickup_zone": (
                        zone_names.loc[
                            zone,
                            "pickup_zone",
                        ]
                        if zone in zone_names.index
                        else "Unknown"
                    ),
                    "pickup_borough": (
                        zone_names.loc[
                            zone,
                            "pickup_borough",
                        ]
                        if zone in zone_names.index
                        else "Unknown"
                    ),
                    "actual_gross_passenger_spend": (
                        actual_relevance.get(
                            zone,
                            0.0,
                        )
                    ),
                    "actual_rank_top20": (
                        actual_rank.get(
                            zone,
                            np.nan,
                        )
                    ),
                })

# ============================================================
# Save outputs
# ============================================================

metrics_df = pd.DataFrame(all_metrics)
predictions_df = pd.DataFrame(all_predictions)
airport_results_df = pd.DataFrame(airport_metrics)
support_metrics_df = pd.DataFrame(support_metrics)

print("\n" + "=" * 80)
print("AIRPORT VS NON-AIRPORT")
print("=" * 80)

print(
    airport_results_df
    .groupby(["baseline", "airport_group"])[
        ["ndcg_at_5", "ndcg_at_10", "ndcg_at_20",
         "precision_at_5", "precision_at_10", "precision_at_20"]
    ]
    .mean()
    .round(4)
    .to_string()
)

metrics_path = OUTPUT_DIR / "ranking_baseline_metrics.csv"
predictions_path = OUTPUT_DIR / "ranking_baseline_predictions.csv"
airport_metrics_path = OUTPUT_DIR / "ranking_airport_metrics.csv"

metrics_df.to_csv(metrics_path, index=False)
predictions_df.to_csv(predictions_path, index=False)
airport_results_df.to_csv(airport_metrics_path, index=False)
support_metrics_path = (
    OUTPUT_DIR / "ranking_support_threshold_metrics.csv"
)

support_metrics_df.to_csv(
    support_metrics_path,
    index=False
)

print("\n" + "=" * 80)
print("SUPPORT THRESHOLD EXPERIMENT")
print("=" * 80)

support_summary = (
    support_metrics_df
    .groupby(["support_threshold", "baseline"])[
        [
            "ndcg_at_5",
            "ndcg_at_10",
            "ndcg_at_20",
            "precision_at_10",
            "support_coverage",
        ]
    ]
    .mean()
    .round(4)
)

print(support_summary.to_string())

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("TIME-BLOCK BREAKDOWN")
print("=" * 70)

time_block_summary = (
    metrics_df
    .groupby(["baseline", "time_block"])[
        ["ndcg_at_10", "precision_at_10"]
    ]
    .mean()
    .round(4)
)

print(time_block_summary)
print("\n" + "=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

print("\n" + "=" * 70)
print("WEEKDAY VS WEEKEND BREAKDOWN")
print("=" * 70)

day_type_summary = (
    metrics_df
    .groupby(["baseline", "day_type"])[
        ["ndcg_at_10", "precision_at_10"]
    ]
    .mean()
    .round(4)
)

print(day_type_summary)

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