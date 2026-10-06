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

# ============================================================
# Fast learned-ranker training dataset
# ============================================================

def build_learned_training_dataset_fast(train, min_history_days=28):
    """
    Build a dense, leakage-safe supervised dataset for the learned
    zone-hour gross-spend ranker.

    This is deliberately separate from build_zone_hour_features().
    The existing feature helper is retained for interactive/test-time
    inference. For model training, rebuilding that feature table for
    every historical hour is unnecessarily expensive.

    Training rows use only information strictly before the target
    date/hour. Missing zone-hours are represented explicitly with zero
    trips/spend so the model sees both active and inactive zones.
    """

    work = train.copy()
    work["pickup_date"] = pd.to_datetime(work["pickup_date"])
    work["pickup_hour"] = work["pickup_hour"].astype(int)

    candidate_zones = sorted(work["PULocationID"].unique().tolist())

    # Static zone metadata known from the training period.
    zone_meta = (
        work[
            [
                "PULocationID",
                "is_airport",
            ]
        ]
        .drop_duplicates("PULocationID")
        .set_index("PULocationID")
        .reindex(candidate_zones)
        .fillna(0)
    )

    # Use only date/hour slots that actually exist in the source data.
    date_hours = (
        work[["pickup_date", "pickup_hour"]]
        .drop_duplicates()
        .sort_values(["pickup_date", "pickup_hour"])
    )

    # Dense zone x date-hour panel. This is small enough for the
    # quarter-sized dataset and makes missing zone-hours explicit.
    zones_df = pd.DataFrame({"PULocationID": candidate_zones})
    zones_df["_key"] = 1
    date_hours = date_hours.copy()
    date_hours["_key"] = 1

    panel = zones_df.merge(date_hours, on="_key", how="inner").drop(
        columns="_key"
    )

    observed = (
        work.groupby(
            ["pickup_date", "pickup_hour", "PULocationID"],
            as_index=False,
        )
        .agg(
            trips=("trips", "sum"),
            gross_passenger_spend=("gross_passenger_spend", "sum"),
            duration_weighted=(
                "avg_trip_duration_min",
                lambda x: np.nan,
            ),
            distance_weighted=(
                "avg_trip_distance_miles",
                lambda x: np.nan,
            ),
        )
    )

    # Recompute weighted duration/distance from the original trip counts.
    work_weighted = work.copy()
    work_weighted["duration_weighted"] = (
        work_weighted["avg_trip_duration_min"]
        * work_weighted["trips"]
    )
    work_weighted["distance_weighted"] = (
        work_weighted["avg_trip_distance_miles"]
        * work_weighted["trips"]
    )

    weighted = (
        work_weighted.groupby(
            ["pickup_date", "pickup_hour", "PULocationID"],
            as_index=False,
        )[
            [
                "duration_weighted",
                "distance_weighted",
            ]
        ]
        .sum()
    )

    observed = observed.drop(
        columns=["duration_weighted", "distance_weighted"]
    ).merge(
        weighted,
        on=["pickup_date", "pickup_hour", "PULocationID"],
        how="left",
    )

    panel = panel.merge(
        observed,
        on=["pickup_date", "pickup_hour", "PULocationID"],
        how="left",
    )

    for col in [
        "trips",
        "gross_passenger_spend",
        "duration_weighted",
        "distance_weighted",
    ]:
        panel[col] = panel[col].fillna(0.0)

    panel["day_of_week"] = panel["pickup_date"].dt.dayofweek
    panel["hour"] = panel["pickup_hour"]
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype(int)
    panel["is_airport"] = (
        panel["PULocationID"].map(zone_meta["is_airport"]).fillna(0).astype(int)
    )
    panel["pickup_datetime"] = (
        panel["pickup_date"]
        + pd.to_timedelta(panel["pickup_hour"], unit="h")
    )

    panel = panel.sort_values(
        ["PULocationID", "pickup_date", "pickup_hour"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Historical zone-wide statistics.
    # These are computed from dates strictly before the target date.
    # --------------------------------------------------------

    daily = (
        panel.groupby(
            ["PULocationID", "pickup_date"],
            as_index=False,
        )[
            [
                "trips",
                "gross_passenger_spend",
                "duration_weighted",
                "distance_weighted",
            ]
        ]
        .sum()
        .sort_values(["PULocationID", "pickup_date"])
    )

    daily_group = daily.groupby("PULocationID")
    daily["prior_trips"] = (
        daily_group["trips"].cumsum() - daily["trips"]
    )
    daily["prior_spend"] = (
        daily_group["gross_passenger_spend"].cumsum()
        - daily["gross_passenger_spend"]
    )
    daily["prior_duration_weighted"] = (
        daily_group["duration_weighted"].cumsum()
        - daily["duration_weighted"]
    )
    daily["prior_distance_weighted"] = (
        daily_group["distance_weighted"].cumsum()
        - daily["distance_weighted"]
    )

    panel = panel.merge(
        daily[
            [
                "PULocationID",
                "pickup_date",
                "prior_trips",
                "prior_spend",
                "prior_duration_weighted",
                "prior_distance_weighted",
            ]
        ],
        on=["PULocationID", "pickup_date"],
        how="left",
    )

    panel["historical_trips"] = panel["prior_trips"]
    panel["historical_gross_spend"] = panel["prior_spend"]
    panel["historical_avg_spend_per_trip"] = (
        panel["prior_spend"]
        / panel["prior_trips"].replace(0, np.nan)
    )
    panel["historical_avg_duration"] = (
        panel["prior_duration_weighted"]
        / panel["prior_trips"].replace(0, np.nan)
    )
    panel["historical_distance"] = (
        panel["prior_distance_weighted"]
        / panel["prior_trips"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Historical zone x DOW x hour demand.
    # Match the existing baseline/helper semantics: cumulative
    # sum over prior matching DOW/hour slots.
    # --------------------------------------------------------

    slot = panel.sort_values(
        ["PULocationID", "day_of_week", "pickup_hour", "pickup_date"]
    ).copy()
    slot_group = slot.groupby(
        ["PULocationID", "day_of_week", "pickup_hour"]
    )

    slot["matching_dow_hour_trips"] = (
        slot_group["trips"].cumsum() - slot["trips"]
    )
    slot["matching_dow_hour_spend"] = (
        slot_group["gross_passenger_spend"].cumsum()
        - slot["gross_passenger_spend"]
    )

    panel = panel.drop(
        columns=[
            "matching_dow_hour_trips",
            "matching_dow_hour_spend",
        ],
        errors="ignore",
    ).merge(
        slot[
            [
                "PULocationID",
                "pickup_date",
                "pickup_hour",
                "matching_dow_hour_trips",
                "matching_dow_hour_spend",
            ]
        ],
        on=["PULocationID", "pickup_date", "pickup_hour"],
        how="left",
    )

    # --------------------------------------------------------
    # Same-hour historical values.
    # Use explicit date merges rather than positional shifts so
    # daylight-saving-time gaps cannot change the meaning of a lag.
    # --------------------------------------------------------

    lookup = panel[
        [
            "PULocationID",
            "pickup_datetime",
            "trips",
            "gross_passenger_spend",
        ]
    ].copy()

    for weeks_back in range(1, 5):
        shifted = lookup.copy()
        shifted["pickup_datetime"] = (
            shifted["pickup_datetime"]
            + pd.Timedelta(days=7 * weeks_back)
        )
        shifted = shifted.rename(
            columns={
                "trips": f"lag_{weeks_back}_week_trips",
                "gross_passenger_spend": (
                    f"lag_{weeks_back}_week_spend"
                ),
            }
        )

        panel = panel.merge(
            shifted,
            on=["PULocationID", "pickup_datetime"],
            how="left",
        )

    panel["same_hour_last_week_trips"] = (
        panel["lag_1_week_trips"]
    )

    panel["trailing_4_week_trips"] = panel[
        [
            "lag_1_week_trips",
            "lag_2_week_trips",
            "lag_3_week_trips",
            "lag_4_week_trips",
        ]
    ].mean(axis=1)

    panel["trailing_4_week_spend"] = panel[
        [
            "lag_1_week_spend",
            "lag_2_week_spend",
            "lag_3_week_spend",
            "lag_4_week_spend",
        ]
    ].mean(axis=1)

    # Keep target rows with enough historical context.
    min_target_date = panel["pickup_date"].min() + pd.Timedelta(
        days=min_history_days
    )
    panel = panel[panel["pickup_date"] >= min_target_date].copy()

    feature_columns = [
        "PULocationID",
        "pickup_date",
        "pickup_hour",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_airport",
        "historical_trips",
        "historical_gross_spend",
        "historical_avg_spend_per_trip",
        "historical_avg_duration",
        "historical_distance",
        "matching_dow_hour_trips",
        "matching_dow_hour_spend",
        "same_hour_last_week_trips",
        "trailing_4_week_trips",
        "trailing_4_week_spend",
        "gross_passenger_spend",
    ]

    panel = panel[feature_columns].copy()

    numeric_columns = [
        col
        for col in feature_columns
        if col not in [
            "PULocationID",
            "pickup_date",
        ]
    ]

    panel[numeric_columns] = (
        panel[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    panel = panel.rename(
        columns={
            "gross_passenger_spend": "target_gross_spend"
        }
    )

    return panel


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



# ====================================================
# LEARNED RANKER — FAST, LEAKAGE-SAFE EVALUATION
# ====================================================

print("\nBuilding learned-ranker models using the fast training dataset...")

MODEL_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_airport",
    "historical_trips",
    "historical_gross_spend",
    "historical_avg_spend_per_trip",
    "historical_avg_duration",
    "historical_distance",
    "matching_dow_hour_trips",
    "matching_dow_hour_spend",
    "same_hour_last_week_trips",
    "trailing_4_week_trips",
    "trailing_4_week_spend",
]

# Train/evaluate one model per rolling-origin fold.
# The existing baseline evaluation above is intentionally unchanged.
learned_fold_models = {}

learned_fold_datasets = {}

# --------------------------------------------------------
# Train one model per fold.
# --------------------------------------------------------

for learned_fold_number, fold_start_str in enumerate(FOLD_STARTS, start=1):

    fold_start = pd.Timestamp(fold_start_str)
    fold_end = fold_start + pd.Timedelta(days=TEST_WINDOW_DAYS)

    train_fold = df[df["pickup_date"] < fold_start].copy()

    if train_fold.empty:
        continue

    print("\n" + "-" * 70)
    print(f"LEARNED RANKER — FOLD {learned_fold_number}")
    print(f"Training period: before {fold_start.date()}")

    learned_training = build_learned_training_dataset_fast(
        train_fold,
        min_history_days=28,
    )

    print(
        f"Training examples: {len(learned_training):,}"
    )

    if learned_training.empty:
        print("No learned-ranker training examples; skipping fold.")
        continue

    X_train = learned_training[MODEL_FEATURES]
    y_train = learned_training["target_gross_spend"]

    learned_model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    learned_model.fit(X_train, y_train)

    learned_fold_models[learned_fold_number] = learned_model
    learned_fold_datasets[learned_fold_number] = learned_training

# --------------------------------------------------------
# Evaluate the learned model on exactly the same test folds
# used by the established baselines.
# --------------------------------------------------------

for learned_fold_number, fold_start_str in enumerate(FOLD_STARTS, start=1):

    if learned_fold_number not in learned_fold_models:
        continue

    fold_start = pd.Timestamp(fold_start_str)
    fold_end = fold_start + pd.Timedelta(days=TEST_WINDOW_DAYS)

    train_fold = df[df["pickup_date"] < fold_start].copy()

    test_fold = df[
        (df["pickup_date"] >= fold_start)
        & (df["pickup_date"] < fold_end)
    ].copy()

    learned_model = learned_fold_models[learned_fold_number]

    for (test_date, test_hour), test_hour_df in test_fold.groupby(
        ["pickup_date", "pickup_hour"]
    ):

        actual_relevance = (
            test_hour_df
            .groupby("PULocationID")["gross_passenger_spend"]
            .sum()
            .to_dict()
        )

        # Reuse the existing, already-validated feature builder for
        # test-time inference. This keeps interactive/test-time feature
        # semantics aligned with the feature sanity check.
        learned_features = build_zone_hour_features(
            history=train_fold,
            target_date=test_date,
            target_hour=test_hour,
        )

        X_test = learned_features[MODEL_FEATURES]

        learned_features["predicted_gross_spend"] = (
            learned_model.predict(X_test)
        )

        ranking_learned = (
            learned_features
            .sort_values(
                [
                    "predicted_gross_spend",
                    "PULocationID",
                ],
                ascending=[False, True],
            )["PULocationID"]
            .tolist()
        )

        learned_metric_row = {
            "fold": learned_fold_number,
            "test_date": test_date.date(),
            "pickup_hour": test_hour,
            "day_of_week": test_date.day_name(),
            "baseline": "learned_ranker",
            "time_block": get_time_block(test_hour),
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

        learned_metrics.append(learned_metric_row)

        # Save top-20 learned predictions for inspection.
        zone_names = (
            test_hour_df[
                [
                    "PULocationID",
                    "pickup_zone",
                    "pickup_borough",
                ]
            ]
            .drop_duplicates("PULocationID")
            .set_index("PULocationID")
        )

        actual_top_20 = sorted(
            actual_relevance.keys(),
            key=lambda zone: (
                -actual_relevance.get(zone, 0.0),
                zone,
            ),
        )[:20]

        actual_rank = {
            zone: rank + 1
            for rank, zone in enumerate(actual_top_20)
        }

        predicted_lookup = learned_features.set_index(
            "PULocationID"
        )["predicted_gross_spend"]

        for rank, zone in enumerate(
            ranking_learned[:20],
            start=1,
        ):

            learned_predictions.append({
                "fold": learned_fold_number,
                "test_date": test_date.date(),
                "pickup_hour": test_hour,
                "day_of_week": test_date.day_name(),
                "baseline": "learned_ranker",
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
                "predicted_gross_passenger_spend": (
                    predicted_lookup.get(zone, 0.0)
                ),
                "actual_gross_passenger_spend": (
                    actual_relevance.get(zone, 0.0)
                ),
                "actual_rank_top20": (
                    actual_rank.get(zone, np.nan)
                ),
            })

# --------------------------------------------------------
# Learned-ranker summary.
# --------------------------------------------------------

learned_metrics_df = pd.DataFrame(learned_metrics)
learned_predictions_df = pd.DataFrame(learned_predictions)

print("\n" + "=" * 80)
print("LEARNED RANKER RESULTS")
print("=" * 80)

if not learned_metrics_df.empty:

    learned_summary = (
        learned_metrics_df
        .groupby("baseline")[[
            "ndcg_at_5",
            "ndcg_at_10",
            "ndcg_at_20",
            "precision_at_10",
            "hit_rate_at_10",
        ]]
        .mean()
        .round(4)
    )

    print(learned_summary.to_string())

    print("\nFold-level NDCG@10:")
    print(
        learned_metrics_df
        .groupby("fold")["ndcg_at_10"]
        .mean()
        .round(4)
        .to_string()
    )

    learned_metrics_path = (
        OUTPUT_DIR / "ranking_learned_metrics.csv"
    )
    learned_predictions_path = (
        OUTPUT_DIR / "ranking_learned_predictions.csv"
    )

    learned_metrics_df.to_csv(
        learned_metrics_path,
        index=False,
    )

    learned_predictions_df.to_csv(
        learned_predictions_path,
        index=False,
    )

    print(
        f"\nSaved learned metrics: {learned_metrics_path}"
    )
    print(
        f"Saved learned predictions: {learned_predictions_path}"
    )

else:
    print("No learned-ranker results were generated.")


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
