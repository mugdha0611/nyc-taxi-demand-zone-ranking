SELECT
    CAST(pickup_datetime AS DATE) AS pickup_date,
    EXTRACT(HOUR FROM pickup_datetime) AS pickup_hour,
    COUNT(*) AS trips

FROM read_parquet(
    'data/processed/yellow_tripdata_*_clean.parquet',
    union_by_name = true
)

GROUP BY
    pickup_date,
    pickup_hour

ORDER BY
    pickup_date,
    pickup_hour;