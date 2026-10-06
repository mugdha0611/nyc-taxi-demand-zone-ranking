from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COLUMNS
# ============================================================

KEEP_COLS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "tolls_amount",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
    "Airport_fee",
    "cbd_congestion_fee",
]


# ============================================================
# CLEAN ONE MONTH
# ============================================================

def clean_one_file(parquet_file):

    print(f"\nProcessing: {parquet_file.name}")

    # --------------------------------------------------------
    # Derive study month from filename
    #
    # Example:
    # yellow_tripdata_2025-01.parquet -> 2025-01
    # yellow_tripdata_2024-12.parquet -> 2024-12
    # --------------------------------------------------------

    month_str = parquet_file.stem.split("_")[-1]

    study_month = pd.Period(
        month_str,
        freq="M"
    )

    start_date = study_month.start_time
    end_date = (study_month + 1).start_time

    print(
        f"Study month: "
        f"{start_date.date()} to "
        f"{(end_date - pd.Timedelta(days=1)).date()}"
    )

    # --------------------------------------------------------
    # Read schema WITHOUT loading the whole file
    # --------------------------------------------------------

    available_cols = pq.ParquetFile(parquet_file).schema.names

    use_cols = [
        col
        for col in KEEP_COLS
        if col in available_cols
    ]

    raw = pd.read_parquet(
        parquet_file,
        columns=use_cols,
        engine="pyarrow"
    )

    raw_count = len(raw)

    print(f"Raw rows: {raw_count:,}")

    # ========================================================
    # DATETIME CONVERSION
    # ========================================================

    raw["tpep_pickup_datetime"] = pd.to_datetime(
        raw["tpep_pickup_datetime"],
        errors="coerce"
    )

    raw["tpep_dropoff_datetime"] = pd.to_datetime(
        raw["tpep_dropoff_datetime"],
        errors="coerce"
    )

    # ========================================================
    # STANDARDIZE TLC COLUMN NAMES
    # ========================================================

    raw.rename(
        columns={
            "tpep_pickup_datetime": "pickup_datetime",
            "tpep_dropoff_datetime": "dropoff_datetime",
        },
        inplace=True,
    )

    # ========================================================
    # NORMALIZE AIRPORT FEE COLUMN
    # ========================================================

    if "Airport_fee" in raw.columns:

        if "airport_fee" in raw.columns:

            raw["airport_fee"] = (
                raw["airport_fee"]
                .fillna(raw["Airport_fee"])
            )

        else:

            raw.rename(
                columns={
                    "Airport_fee": "airport_fee"
                },
                inplace=True,
            )

        if "Airport_fee" in raw.columns:

            raw.drop(
                columns=["Airport_fee"],
                inplace=True,
            )

    # ========================================================
    # STUDY MONTH FILTER
    # ========================================================

    in_study_period = (
        raw["pickup_datetime"].notna()
        & (raw["pickup_datetime"] >= start_date)
        & (raw["pickup_datetime"] < end_date)
    )

    out_of_period_count = int(
        (~in_study_period).sum()
    )

    print(
        f"Outside file's study month: "
        f"{out_of_period_count:,}"
    )

    # Keep only records belonging to this file's month
    raw = raw.loc[in_study_period]

    study_period_count = len(raw)

    # ========================================================
    # DERIVED FEATURES
    # ========================================================

    raw["trip_duration_min"] = (
        raw["dropoff_datetime"]
        - raw["pickup_datetime"]
    ).dt.total_seconds() / 60

    raw["date"] = (
        raw["pickup_datetime"]
        .dt.floor("D")
    )

    raw["hour"] = (
        raw["pickup_datetime"]
        .dt.hour
    )

    raw["day_of_week"] = (
        raw["pickup_datetime"]
        .dt.day_name()
    )

    raw["day_of_week_num"] = (
        raw["pickup_datetime"]
        .dt.dayofweek
    )

    raw["is_weekend"] = (
        raw["day_of_week_num"] >= 5
    )

    raw["month"] = (
        raw["pickup_datetime"]
        .dt.month
    )

    # ========================================================
    # DATA QUALITY RULES
    # ========================================================

    # These rules are tracked separately for diagnostics.
    #
    # IMPORTANT:
    # These categories can overlap. A single trip may violate
    # more than one rule.
    #
    # The first-failing-rule classification below provides
    # mutually exclusive buckets.

    quality_rules = {

        # ----------------------------------------------------
        # DROP-OFF
        # ----------------------------------------------------

        "missing_dropoff_datetime": (
            raw["dropoff_datetime"].isna()
        ),

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        "missing_location": (
            raw["PULocationID"].isna()
            | raw["DOLocationID"].isna()
        ),

        # ----------------------------------------------------
        # DURATION
        # ----------------------------------------------------

        "zero_or_negative_duration": (
            raw["trip_duration_min"] <= 0
        ),

        "short_duration_under_1_min": (
            (raw["trip_duration_min"] > 0)
            & (raw["trip_duration_min"] < 1)
        ),

        "long_duration_over_180_min": (
            raw["trip_duration_min"] > 180
        ),

        # ----------------------------------------------------
        # DISTANCE
        # ----------------------------------------------------

        "zero_distance": (
            raw["trip_distance"] <= 0
        ),

        "distance_over_100_miles": (
            raw["trip_distance"] > 100
        ),

        # ----------------------------------------------------
        # FARE
        # ----------------------------------------------------

        "negative_fare": (
            raw["fare_amount"] < 0
        ),

        "fare_over_500": (
            raw["fare_amount"] > 500
        ),

        # ----------------------------------------------------
        # TOTAL AMOUNT
        # ----------------------------------------------------

        "negative_total_amount": (
            raw["total_amount"] < 0
        ),

        "total_amount_over_1000": (
            raw["total_amount"] > 1000
        ),
    }

    # ========================================================
    # INDIVIDUAL QUALITY-RULE COUNTS
    # ========================================================

    quality_breakdown = []

    for rule, mask in quality_rules.items():

        excluded_records = int(
            mask.sum()
        )

        quality_breakdown.append({
            "file": parquet_file.name,
            "rule": rule,
            "excluded_records": excluded_records,
            "pct_of_study_period_records": round(
                excluded_records
                / study_period_count
                * 100,
                2
            )
            if study_period_count > 0
            else 0,
        })

    quality_breakdown_df = pd.DataFrame(
        quality_breakdown
    )

    # ========================================================
    # FIRST FAILING RULE
    # ========================================================

    # Every record starts as valid.
    #
    # Once a record is assigned a failure category, later
    # rules cannot overwrite it.
    #
    # Therefore these buckets are mutually exclusive.

    first_failure = pd.Series(
        "valid",
        index=raw.index,
        dtype="object",
    )

    rule_priority = [
        "missing_dropoff_datetime",
        "missing_location",
        "zero_or_negative_duration",
        "short_duration_under_1_min",
        "long_duration_over_180_min",
        "zero_distance",
        "distance_over_100_miles",
        "negative_fare",
        "fare_over_500",
        "negative_total_amount",
        "total_amount_over_1000",
    ]

    for rule in rule_priority:

        mask = (
            quality_rules[rule]
            & (first_failure == "valid")
        )

        first_failure.loc[mask] = rule

    # ========================================================
    # FIRST FAILURE SUMMARY
    # ========================================================

    first_failure_counts = (
        first_failure
        .value_counts()
        .rename_axis("first_failing_rule")
        .reset_index(
            name="records"
        )
    )

    first_failure_counts["pct_of_study_period"] = (
        first_failure_counts["records"]
        / study_period_count
        * 100
    ).round(2)

    first_failure_counts.insert(
        0,
        "file",
        parquet_file.name,
    )

    # ========================================================
    # FINAL VALIDITY MASK
    # ========================================================

    invalid_mask = (
        first_failure != "valid"
    )

    valid = ~invalid_mask

    invalid_in_period_count = int(
        invalid_mask.sum()
    )

    # ========================================================
    # FILTER INVALID RECORDS
    # ========================================================

    raw = raw.loc[valid]

    clean_count = len(raw)

    total_excluded = (
        out_of_period_count
        + invalid_in_period_count
    )

    # ========================================================
    # DATA-QUALITY ACCOUNTING CHECK
    # ========================================================

    assert (
        raw_count
        == (
            out_of_period_count
            + invalid_in_period_count
            + clean_count
        )
    ), (
        "Data-quality accounting error: "
        "raw != out_of_period + invalid + clean"
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"Study-period rows: "
        f"{study_period_count:,}"
    )

    print(
        f"Clean rows: "
        f"{clean_count:,}"
    )

    print(
        f"Invalid rows within study period: "
        f"{invalid_in_period_count:,}"
    )

    print(
        f"Outside file's study month: "
        f"{out_of_period_count:,}"
    )

    print(
        f"Total excluded: "
        f"{total_excluded:,}"
    )

    if raw_count > 0:

        print(
            f"Total exclusion percentage: "
            f"{total_excluded / raw_count * 100:.2f}%"
        )

    # ========================================================
    # PRINT FIRST-FAILURE BREAKDOWN
    # ========================================================

    print("\nFirst failing rule:")

    for _, row in first_failure_counts.iterrows():

        print(
            f"  {row['first_failing_rule']}: "
            f"{int(row['records']):,} "
            f"({row['pct_of_study_period']:.2f}%)"
        )

    # ========================================================
    # ADDITIONAL ROW-LEVEL METRICS
    # ========================================================

    # These are retained as trip-level fields only.
    # Project-level KPIs should be calculated as ratios of
    # aggregate sums in SQL.

    raw["revenue_per_mile"] = (
        raw["total_amount"]
        / raw["trip_distance"].replace(
            0,
            np.nan
        )
    )

    raw["tip_rate"] = (
        raw["tip_amount"]
        / raw["fare_amount"].replace(
            0,
            np.nan
        )
    )

    # ========================================================
    # SAVE CLEANED DATA
    # ========================================================

    output_file = (
        PROCESSED_DIR
        / f"{parquet_file.stem}_clean.parquet"
    )

    raw.to_parquet(
        output_file,
        engine="pyarrow",
        index=False,
    )

    print(
        f"Saved: {output_file}"
    )

    # ========================================================
    # SAVE PER-MONTH QUALITY BREAKDOWN
    # ========================================================

    quality_breakdown_file = (
        PROCESSED_DIR
        / f"{parquet_file.stem}_quality_breakdown.csv"
    )

    quality_breakdown_df.to_csv(
        quality_breakdown_file,
        index=False,
    )

    first_failure_file = (
        PROCESSED_DIR
        / f"{parquet_file.stem}_first_failure.csv"
    )

    first_failure_counts.to_csv(
        first_failure_file,
        index=False,
    )

    print(
        f"Saved: {quality_breakdown_file}"
    )

    print(
        f"Saved: {first_failure_file}"
    )

    # ========================================================
    # RETURN SUMMARY
    # ========================================================

    return {
        "file": parquet_file.name,
        "raw_trip_count": raw_count,
        "out_of_period_count": out_of_period_count,
        "invalid_in_period_count": invalid_in_period_count,
        "clean_trip_count": clean_count,
        "total_excluded_trip_count": total_excluded,
        "excluded_pct": round(
            total_excluded
            / raw_count
            * 100,
            2,
        ),
    }


# ============================================================
# MAIN
# ============================================================

files = sorted(
    RAW_DIR.glob(
        "yellow_tripdata_*.parquet"
    )
)

if not files:

    raise FileNotFoundError(
        f"No TLC parquet files found in {RAW_DIR}"
    )


# ============================================================
# REMOVE STALE QUALITY OUTPUTS
# ============================================================

for old_file in PROCESSED_DIR.glob(
    "*_quality_breakdown.csv"
):

    old_file.unlink()


for old_file in PROCESSED_DIR.glob(
    "*_first_failure.csv"
):

    old_file.unlink()


# ============================================================
# PROCESS FILES
# ============================================================

results = []

for file in files:

    result = clean_one_file(file)

    results.append(result)


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

summary = pd.DataFrame(
    results
)

summary.to_csv(
    PROCESSED_DIR
    / "data_quality_summary.csv",
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n======================================")
print("DATA CLEANING COMPLETE")
print("======================================")

print(
    summary.to_string(
        index=False
    )
)