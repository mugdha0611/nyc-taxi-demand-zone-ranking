# NYC Taxi Demand Forecasting & Pickup-Zone Ranking

An end-to-end data science project using 10.35M NYC Yellow Taxi trips from
January–March 2025 to analyze demand patterns, forecast hourly demand, and
rank pickup zones by historical gross passenger-spend opportunity.

## 🚀 Live Demo

**[Open the Interactive Streamlit Dashboard](https://nyc-taxi-demand-zone-ranking-fcstgpdk57j7qjhzbpdsp2.streamlit.app/)**

Explore:
- 📊 NYC taxi demand and zone-level analytics
- 🔮 Hourly demand forecasting
- 🏆 Pickup-zone ranking and offline model evaluation
- 🚕 Personalized Driver Opportunity Explorer
- 📈 Historical demand patterns by hour, day type, and zone

> **Note:** The dashboard analyzes Yellow Taxi data only. Opportunity scores
> represent gross passenger-spend opportunity, not driver earnings or profit.
> The dataset does not observe driver supply, competition, wait times,
> trip acceptance, deadhead distance, or operating costs.

## 📌 Key Results

| Metric | Result |
|---|---:|
| Raw trips analyzed | 11.2M |
| Clean trips retained | 10.35M |
| Study period | Jan–Mar 2025 |
| Demand forecast WAPE | 6.75% (next hour) |
| 24-hour forecast WAPE | 8.90% |
| Best baseline NDCG@10 | 0.825 |
| Learned ranker NDCG@10 | **0.967** |
| Learned ranker Precision@10 | **82.6%** |

## Objective

Build an end-to-end mobility analytics workflow that:

1. Ingests monthly TLC Parquet files and the official taxi-zone lookup.
2. Cleans invalid/implausible trip records with explicit data-quality rules.
3. Uses SQL for KPI and segment analysis.
4. Builds hourly and zone-level demand tables.
5. Produces dashboard-ready CSVs for Power BI.
6. Forecasts hourly pickup demand as an optional modeling layer.

## Data source

Official NYC TLC Trip Record Data:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The project defaults to January–March 2025 Yellow Taxi data, but `src/download_data.py` can download multiple months. TLC publishes the trip records monthly in Parquet format and provides a separate taxi-zone lookup table.

## Project structure

```
nyc-tlc-trip-analytics/
├── data/
│   ├── raw/                 # downloaded Parquet + zone lookup (gitignored)
│   └── processed/           # dashboard/model tables (gitignored)
├── sql/
│   ├── kpis.sql
│   ├── hourly_demand.sql
│   └── zone_analysis.sql
├── src/
│   ├── download_data.py
│   ├── clean_transform.py
│   ├── run_sql_analysis.py
│   └── forecast_demand.py
├── dashboard/
│   └── POWER_BI_GUIDE.md
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## 1. Download data

```bash
python src/download_data.py --months 2026-01
```

For a more robust analysis, use three months:

```bash
python src/download_data.py --months 2025-10 2025-11 2025-12
```

The script uses the official TLC CloudFront links.

## 2. Clean and transform

```bash
python src/clean_transform.py
```

The cleaning layer removes records with missing timestamps/zones and implausible values such as non-positive trip duration, negative distance, or negative monetary fields. The exact exclusion counts are written to `data/processed/data_quality_summary.csv` so the resume can report real numbers rather than estimated ones.

## 3. Run SQL analysis

```bash
python src/run_sql_analysis.py
```

This creates:

- `kpis.csv`
- `hourly_demand.csv`
- `zone_analysis.csv`
- `zone_hour_heatmap.csv`
- `payment_analysis.csv`

DuckDB queries the Parquet data directly, which avoids loading the full raw dataset into pandas.

## 4. Demand forecasting

```bash
python src/forecast_demand.py
```

The model predicts hourly pickup demand. It uses time-derived features and lag/rolling features and evaluates using a chronological train/test split.

The model is intentionally kept interpretable for a resume project. A useful extension is comparing a seasonal baseline against Gradient Boosting and reporting MAE/RMSE.

## 5. Power BI

Import the CSVs in `data/processed/` into Power BI and follow `dashboard/POWER_BI_GUIDE.md`.

## Resume evidence to collect after running

Do not fill these in before running the pipeline:

- Raw trip count: `raw_trip_count`
- Cleaned trip count: `clean_trip_count`
- Number/percentage removed by data-quality rule
- Top pickup zones by trip volume
- Peak demand hour/day
- Highest-revenue zones
- Forecast MAE/RMSE
- 2–3 defensible dashboard findings

## Data-quality note

TLC states that the trip records are submitted by technology providers and that TLC does not represent that the raw records are fully accurate. The project therefore treats cleaning and data-quality auditing as a first-class part of the analysis.
