WITH days AS (
    SELECT
        is_weekend,
        COUNT(DISTINCT CAST(pickup_datetime AS DATE)) AS n_days
    FROM read_parquet(
        'data/processed/yellow_tripdata_*_clean.parquet',
        union_by_name = true
    )
    GROUP BY is_weekend
),

zone_hour_counts AS (
    SELECT
        t.is_weekend,
        EXTRACT(HOUR FROM t.pickup_datetime) AS hour,
        t.PULocationID,
        COUNT(*) AS trips
    FROM read_parquet(
        'data/processed/yellow_tripdata_*_clean.parquet',
        union_by_name = true
    ) t
    GROUP BY
        t.is_weekend,
        EXTRACT(HOUR FROM t.pickup_datetime),
        t.PULocationID
)

SELECT
    c.hour,

    CASE
        WHEN c.is_weekend THEN 'Weekend'
        ELSE 'Weekday'
    END AS day_type,

    c.PULocationID,

    CASE
        WHEN c.PULocationID = 264 THEN 'Unknown'
        WHEN c.PULocationID = 265 THEN 'Outside NYC'
        ELSE COALESCE(z.Zone, 'Unknown')
    END AS pickup_zone,

    CASE
        WHEN c.PULocationID IN (264, 265) THEN 'Unknown'
        ELSE COALESCE(z.Borough, 'Unknown')
    END AS pickup_borough,

    ROUND(
        c.trips * 1.0 / d.n_days,
        2
    ) AS avg_daily_trips

FROM zone_hour_counts c

JOIN days d
    ON c.is_weekend = d.is_weekend

LEFT JOIN read_csv_auto(
    'data/raw/taxi_zone_lookup.csv'
) z
    ON c.PULocationID = z.LocationID

ORDER BY
    day_type,
    c.hour,
    avg_daily_trips DESC;