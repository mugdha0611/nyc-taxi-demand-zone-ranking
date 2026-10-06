# Power BI Dashboard

## Recommended pages

### 1. Executive Overview
Cards:
- Total trips
- Total revenue
- Average fare/trip
- Average trip duration
- Average distance

Charts:
- Trips by date
- Trips by hour
- Revenue by day

### 2. Demand & Hotspots
- Filled map or shape map by pickup zone
- Hour x day-of-week heatmap
- Top pickup zones by trips
- Top pickup zones by revenue

### 3. Trip Economics
- Revenue per trip by hour
- Revenue per mile by zone
- Trip duration vs distance
- Payment-type mix
- Card tip rate

### 4. Data Quality
- Raw records
- Clean records
- Excluded records
- Exclusion percentage

Use `data_quality_summary.csv` to make the cleaning process transparent.

## KPI definitions

**Trips:** count of cleaned trip records.

**Revenue:** sum of `total_amount`.

**Revenue per trip:** total revenue / trips.

**Revenue per mile:** total amount / trip distance, excluding zero-distance records.

**Card tip rate:** tip amount / fare amount for credit-card trips only. Cash tips are not captured in the TLC `tip_amount` field.

## Key finding workflow

Do not write a finding before checking it in the data. The final resume bullet should state a concrete result such as:

"Identified [X]% higher pickup demand during [time window] in [zones], informing [dashboard/operational insight]."

Only use the exact percentage and zones produced by the analysis.
