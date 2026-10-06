SELECT
    CAST(t.pickup_datetime AS DATE) AS pickup_date,
    EXTRACT(HOUR FROM t.pickup_datetime) AS pickup_hour,

    t.PULocationID,

    CASE
        WHEN t.PULocationID = 264 THEN 'Unknown'
        WHEN t.PULocationID = 265 THEN 'Outside NYC'
        ELSE COALESCE(z.Zone, 'Unknown')
    END AS pickup_zone,

    CASE
        WHEN t.PULocationID IN (264, 265) THEN 'Unknown'
        ELSE COALESCE(z.Borough, 'Unknown')
    END AS pickup_borough,

    COUNT(*) AS trips,

    ROUND(
        SUM(t.total_amount),
        2
    ) AS gross_passenger_spend,

    ROUND(
        AVG(t.total_amount),
        2
    ) AS avg_gross_spend_per_trip,

    ROUND(
        SUM(t.total_amount)
        / NULLIF(SUM(t.trip_duration_min), 0),
        4
    ) AS gross_spend_per_minute,

    ROUND(
        AVG(t.trip_duration_min),
        2
    ) AS avg_trip_duration_min,

    ROUND(
        AVG(t.trip_distance),
        2
    ) AS avg_trip_distance_miles

FROM read_parquet(
    'data/processed/yellow_tripdata_*_clean.parquet',
    union_by_name = true
) t

LEFT JOIN read_csv_auto(
    'data/raw/taxi_zone_lookup.csv'
) z
    ON t.PULocationID = z.LocationID

GROUP BY
    CAST(t.pickup_datetime AS DATE),
    EXTRACT(HOUR FROM t.pickup_datetime),
    t.PULocationID,
    pickup_zone,
    pickup_borough

ORDER BY
    pickup_date,
    pickup_hour,
    trips DESC;