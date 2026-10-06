SELECT
    COUNT(*) AS trips,
    COUNT(DISTINCT CAST(pickup_datetime AS DATE)) AS active_days,

    ROUND(AVG(trip_distance), 2) AS avg_trip_distance_miles,
    ROUND(MEDIAN(trip_distance), 2) AS median_trip_distance_miles,

    ROUND(AVG(trip_duration_min), 2) AS avg_trip_duration_min,
    ROUND(MEDIAN(trip_duration_min), 2) AS median_trip_duration_min,

    ROUND(AVG(total_amount), 2) AS avg_gross_spend_per_trip,
    ROUND(MEDIAN(total_amount), 2) AS median_gross_spend_per_trip,

    ROUND(SUM(total_amount), 2) AS gross_passenger_spend,

    ROUND(
        SUM(total_amount) / NULLIF(SUM(trip_distance), 0),
        2
    ) AS gross_spend_per_mile,

    ROUND(
        SUM(CASE WHEN payment_type = 1 THEN tip_amount ELSE 0 END)
        /
        NULLIF(
            SUM(CASE WHEN payment_type = 1 THEN fare_amount ELSE 0 END),
            0
        ),
        4
    ) AS card_tip_rate

FROM read_parquet(
    'data/processed/yellow_tripdata_*_clean.parquet',
    union_by_name = true
);