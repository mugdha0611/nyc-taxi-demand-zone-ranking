from __future__ import annotations

from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
SQL = ROOT / "sql"

con = duckdb.connect()

queries = {
    "kpis": "kpis.sql",
    "kpis_monthly": "kpis_monthly.sql",
    "hourly_profile": "hourly_profile.sql",
    "zone_hour_profile": "zone_hour_profile.sql",
    "hourly_timeseries": "hourly_timeseries.sql",
    "zone_hour_timeseries": "zone_hour_timeseries.sql",
    "zone_analysis": "zone_analysis.sql",
    "payment_analysis": "payment_analysis.sql",
}

for name, filename in queries.items():
    query = (SQL / filename).read_text()
    df = con.execute(query).df()
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"Wrote {name}.csv: {len(df):,} rows")

# Heatmap-ready table: zone x hour.
# ============================================================
# ZONE × HOUR HEATMAP
# ============================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

heatmap = con.execute("""
    WITH zone_lookup AS (
        SELECT
            CAST(LocationID AS INTEGER) AS LocationID,
            Borough,
            Zone
        FROM read_csv_auto(
            'data/raw/taxi_zone_lookup.csv',
            HEADER = TRUE
        )
    )

    SELECT
        z.Borough AS pickup_borough,
        z.Zone AS pickup_zone,
        EXTRACT(
            HOUR FROM t.pickup_datetime
        ) AS hour,
        COUNT(*) AS trips

    FROM read_parquet(
        'data/processed/yellow_tripdata_*_clean.parquet'
    ) AS t

    LEFT JOIN zone_lookup AS z
        ON CAST(t.PULocationID AS INTEGER)
        = z.LocationID

    GROUP BY
        z.Borough,
        z.Zone,
        EXTRACT(HOUR FROM t.pickup_datetime)

    ORDER BY
        z.Borough,
        z.Zone,
        hour
""").df()

heatmap.to_csv(
    PROCESSED_DIR / "zone_hour_heatmap.csv",
    index=False
)

print(
    f"Wrote zone_hour_heatmap.csv: "
    f"{len(heatmap):,} rows"
)