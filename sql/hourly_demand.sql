SELECT
    EXTRACT(HOUR FROM t.pickup_datetime) AS pickup_hour,

    CASE
        WHEN EXTRACT(DOW FROM t.pickup_datetime) IN (0, 6)
            THEN 'Weekend'
        ELSE 'Weekday'
    END AS day_type,

    z.Borough AS pickup_borough,
    z.Zone AS pickup_zone,

    ROUND(
        COUNT(*) * 1.0
        / COUNT(DISTINCT CAST(t.pickup_datetime AS DATE)),
        2
    ) AS avg_daily_trips

FROM read_parquet(
    'data/processed/yellow_tripdata_*_clean.parquet',
    union_by_name = true
) t

LEFT JOIN read_csv_auto(
    'data/raw/taxi_zone_lookup.csv'
) z
    ON t.PULocationID = z.LocationID

GROUP BY
    pickup_hour,
    day_type,
    z.Borough,
    z.Zone

ORDER BY
    day_type,
    pickup_hour,
    avg_daily_trips DESC;