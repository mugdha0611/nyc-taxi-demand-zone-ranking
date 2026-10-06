from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================
# Helper functions
# ============================================================

def wape(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Weighted Absolute Percentage Error.
    """
    denominator = np.abs(actual).sum()

    if denominator == 0:
        return np.nan

    return np.abs(actual - predicted).sum() / denominator


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Calculate forecasting metrics.
    """
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "wape": wape(actual, predicted),
    }


def make_model():
    """
    Create a fresh HistGradientBoosting model.
    """
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        random_state=42,
    )


# ============================================================
# Load hourly demand
# ============================================================

hourly = pd.read_csv(
    OUT / "hourly_timeseries.csv",
    parse_dates=["pickup_date"],
)

print("Loaded hourly demand:")
print(hourly.shape)
print(hourly.head())


# ============================================================
# Create timestamp
# ============================================================

hourly["timestamp"] = (
    hourly["pickup_date"]
    + pd.to_timedelta(hourly["pickup_hour"], unit="h")
)

hourly = (
    hourly[["timestamp", "trips"]]
    .groupby("timestamp", as_index=False)["trips"]
    .sum()
    .sort_values("timestamp")
)


# ============================================================
# Preserve the DST gap
# ============================================================

# Create the complete hourly index.
# IMPORTANT:
# We reindex WITHOUT fill_value=0.
# Therefore the nonexistent March 9, 2025 02:00 hour remains NaN.

full_index = pd.date_range(
    start=hourly["timestamp"].min(),
    end=hourly["timestamp"].max(),
    freq="h",
)

ts = (
    hourly
    .set_index("timestamp")
    .reindex(full_index)
    .rename_axis("timestamp")
    .reset_index()
)

ts = ts.rename(columns={"trips": "trips"})


# ============================================================
# Verify the missing hour
# ============================================================

missing_hours = ts[ts["trips"].isna()]

print()
print("Missing hourly observations:")
print(missing_hours)

print()
print(f"Expected hourly observations: {len(full_index):,}")
print(f"Observed hourly observations: {hourly.shape[0]:,}")
print(f"Missing observations: {len(missing_hours):,}")


# ============================================================
# Calendar features
# ============================================================

ts["hour"] = ts["timestamp"].dt.hour
ts["dow"] = ts["timestamp"].dt.dayofweek
ts["is_weekend"] = (ts["dow"] >= 5).astype(int)

ts["date"] = ts["timestamp"].dt.date

# Holiday indicators are used only for error analysis.
holidays = {
    pd.Timestamp("2025-01-01").date(): "New Year's Day",
    pd.Timestamp("2025-01-20").date(): "MLK Day",
    pd.Timestamp("2025-02-17").date(): "Presidents' Day",
}

ts["holiday"] = ts["date"].map(holidays).fillna("None")


# ============================================================
# Lag features
# ============================================================

ts["lag_1"] = ts["trips"].shift(1)
ts["lag_24"] = ts["trips"].shift(24)
ts["lag_48"] = ts["trips"].shift(48)
ts["lag_168"] = ts["trips"].shift(168)


# ============================================================
# Rolling demand feature
# ============================================================

ts["rolling_24"] = (
    ts["trips"]
    .shift(1)
    .rolling(24, min_periods=18)
    .mean()
)


# ============================================================
# Model feature sets
# ============================================================

# Model that can use the immediately previous hour.
features_lag1 = [
    "hour",
    "dow",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_48",
    "lag_168",
    "rolling_24",
]


# Model intended for >=24-hour-ahead planning.
# Nothing newer than 24 hours is used.
features_24h = [
    "hour",
    "dow",
    "is_weekend",
    "lag_24",
    "lag_48",
    "lag_168",
]


# ============================================================
# Rolling-origin validation
# ============================================================

# We use one-week test windows.
#
# Each fold:
#   train = everything available before the test window
#   test  = next 7 days
#
# This mimics repeatedly forecasting future periods.

forecast_start_dates = [
    pd.Timestamp("2025-02-03"),
    pd.Timestamp("2025-02-17"),
    pd.Timestamp("2025-03-03"),
    pd.Timestamp("2025-03-17"),
]

test_window_hours = 7 * 24

prediction_rows = []
metric_rows = []


# ============================================================
# Run rolling-origin folds
# ============================================================

for fold_number, fold_start in enumerate(
    forecast_start_dates,
    start=1,
):

    fold_end = fold_start + pd.Timedelta(hours=test_window_hours)

    print()
    print("=" * 70)
    print(f"Fold {fold_number}")
    print(f"Test period: {fold_start} → {fold_end}")
    print("=" * 70)

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    train = ts[
        ts["timestamp"] < fold_start
    ].copy()

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    test = ts[
        (ts["timestamp"] >= fold_start)
        & (ts["timestamp"] < fold_end)
    ].copy()

    # --------------------------------------------------------
    # Remove rows affected by the DST gap / missing values.
    #
    # We require:
    #   target is observed
    #   relevant lag features are observed
    # --------------------------------------------------------

    test_lag1 = test.dropna(
        subset=["trips"] + features_lag1
    ).copy()

    test_24h = test.dropna(
        subset=["trips"] + features_24h
    ).copy()

    train_lag1 = train.dropna(
        subset=["trips"] + features_lag1
    ).copy()

    train_24h = train.dropna(
        subset=["trips"] + features_24h
    ).copy()

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    model_lag1 = make_model()
    model_24h = make_model()

    # --------------------------------------------------------
    # Fit lag-1 model
    # --------------------------------------------------------

    model_lag1.fit(
        train_lag1[features_lag1],
        train_lag1["trips"],
    )

    pred_lag1 = model_lag1.predict(
        test_lag1[features_lag1]
    )

    # --------------------------------------------------------
    # Fit 24-hour-ahead model
    # --------------------------------------------------------

    model_24h.fit(
        train_24h[features_24h],
        train_24h["trips"],
    )

    pred_24h = model_24h.predict(
        test_24h[features_24h]
    )

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    # Previous hour
    baseline_lag1 = test_lag1["lag_1"]

    # Same hour yesterday
    baseline_lag24 = test_24h["lag_24"]

    # Same hour last week
    baseline_lag168 = test_24h["lag_168"]

    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    for df, predictions, model_name in [
        (
            test_lag1,
            pred_lag1,
            "hist_gradient_boosting_lag1",
        ),
        (
            test_24h,
            pred_24h,
            "hist_gradient_boosting_24h",
        ),
    ]:

        for timestamp, actual, prediction, hour, day_type, holiday in zip(
            df["timestamp"],
            df["trips"],
            predictions,
            df["hour"],
            np.where(
                df["is_weekend"] == 1,
                "Weekend",
                "Weekday",
            ),
            df["holiday"],
        ):

            prediction_rows.append(
                {
                    "timestamp": timestamp,
                    "date": timestamp.date(),
                    "hour": hour,
                    "day_type": day_type,
                    "holiday": holiday,
                    "fold": fold_number,
                    "model": model_name,
                    "actual_trips": actual,
                    "predicted_trips": prediction,
                }
            )

    # --------------------------------------------------------
    # Store baseline predictions
    # --------------------------------------------------------

    for df, predictions, model_name in [
        (
            test_lag1,
            baseline_lag1,
            "baseline_lag1",
        ),
        (
            test_24h,
            baseline_lag24,
            "baseline_lag24",
        ),
        (
            test_24h,
            baseline_lag168,
            "baseline_lag168",
        ),
    ]:

        for timestamp, actual, prediction, hour, day_type, holiday in zip(
            df["timestamp"],
            df["trips"],
            predictions,
            df["hour"],
            np.where(
                df["is_weekend"] == 1,
                "Weekend",
                "Weekday",
            ),
            df["holiday"],
        ):

            prediction_rows.append(
                {
                    "timestamp": timestamp,
                    "date": timestamp.date(),
                    "hour": hour,
                    "day_type": day_type,
                    "holiday": holiday,
                    "fold": fold_number,
                    "model": model_name,
                    "actual_trips": actual,
                    "predicted_trips": prediction,
                }
            )


# ============================================================
# Predictions dataframe
# ============================================================

predictions = pd.DataFrame(prediction_rows)

predictions = predictions.sort_values(
    ["fold", "timestamp", "model"]
).reset_index(drop=True)


# ============================================================
# Overall metrics
# ============================================================

for model_name, group in predictions.groupby("model"):

    metrics = evaluate(
        group["actual_trips"],
        group["predicted_trips"],
    )

    metric_rows.append(
        {
            "evaluation_set": "overall",
            "group": "all",
            "fold": "all",
            "model": model_name,
            "n": len(group),
            **metrics,
        }
    )


# ============================================================
# Per-fold metrics
# ============================================================

for (fold, model_name), group in predictions.groupby(
    ["fold", "model"]
):

    metrics = evaluate(
        group["actual_trips"],
        group["predicted_trips"],
    )

    metric_rows.append(
        {
            "evaluation_set": "by_fold",
            "group": "all",
            "fold": fold,
            "model": model_name,
            "n": len(group),
            **metrics,
        }
    )


# ============================================================
# Error by hour
# ============================================================

for (hour, model_name), group in predictions.groupby(
    ["hour", "model"]
):

    metrics = evaluate(
        group["actual_trips"],
        group["predicted_trips"],
    )

    metric_rows.append(
        {
            "evaluation_set": "by_hour",
            "group": hour,
            "fold": "all",
            "model": model_name,
            "n": len(group),
            **metrics,
        }
    )


# ============================================================
# Error by weekday / weekend
# ============================================================

for (day_type, model_name), group in predictions.groupby(
    ["day_type", "model"]
):

    metrics = evaluate(
        group["actual_trips"],
        group["predicted_trips"],
    )

    metric_rows.append(
        {
            "evaluation_set": "by_day_type",
            "group": day_type,
            "fold": "all",
            "model": model_name,
            "n": len(group),
            **metrics,
        }
    )


# ============================================================
# Holiday error analysis
# ============================================================

holiday_predictions = predictions[
    predictions["holiday"] != "None"
].copy()

for (holiday, model_name), group in holiday_predictions.groupby(
    ["holiday", "model"]
):

    metrics = evaluate(
        group["actual_trips"],
        group["predicted_trips"],
    )

    metric_rows.append(
        {
            "evaluation_set": "holiday",
            "group": holiday,
            "fold": "all",
            "model": model_name,
            "n": len(group),
            **metrics,
        }
    )


# ============================================================
# Save metrics
# ============================================================

metrics_df = pd.DataFrame(metric_rows)

metrics_df.to_csv(
    OUT / "forecast_metrics.csv",
    index=False,
)


# ============================================================
# Save predictions
# ============================================================

predictions.to_csv(
    OUT / "forecast_predictions.csv",
    index=False,
)


# ============================================================
# Train final models on all available valid data
# ============================================================

final_train_lag1 = ts.dropna(
    subset=["trips"] + features_lag1
).copy()

final_train_24h = ts.dropna(
    subset=["trips"] + features_24h
).copy()

final_model_lag1 = make_model()

final_model_lag1.fit(
    final_train_lag1[features_lag1],
    final_train_lag1["trips"],
)

final_model_24h = make_model()

final_model_24h.fit(
    final_train_24h[features_24h],
    final_train_24h["trips"],
)


# Save the main model.
joblib.dump(
    final_model_lag1,
    MODEL_DIR / "demand_model.joblib"
)


# ============================================================
# Print overall results
# ============================================================

overall = metrics_df[
    metrics_df["evaluation_set"] == "overall"
].copy()

overall = overall.sort_values("wape")


print()
print()
print("=" * 70)
print("FORECASTING RESULTS")
print("=" * 70)

print(
    overall[
        [
            "model",
            "n",
            "mae",
            "rmse",
            "wape",
        ]
    ].to_string(index=False)
)

print()
print("DST check:")
print(
    "Missing hourly observations:",
    len(missing_hours),
)

print()
print("Saved:")
print(" - forecast_metrics.csv")
print(" - forecast_predictions.csv")
print(" - demand_model.joblib")