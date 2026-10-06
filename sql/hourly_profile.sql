SELECT
    EXTRACT(HOUR FROM pickup_datetime) AS pickup_hour,

    CASE
        WHEN EXTRACT(DOW FROM pickup_datetime) IN (0, 6)
            THEN 'Weekend'
        ELSE 'Weekday'
    END AS day_type,

    ROUND(
        COUNT(*) * 1.0
        / COUNT(DISTINCT CAST(pickup_datetime AS DATE)),
        2
    ) AS avg_daily_trips

FROM read_parquet(
    'data/processed/yellow_tripdata_*_clean.parquet',
    union_by_name = true
)

GROUP BY
    pickup_hour,
    day_type

ORDER BY
    day_type,
    pickup_hour;