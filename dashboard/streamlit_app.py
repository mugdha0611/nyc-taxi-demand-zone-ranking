from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NYC Taxi Demand & Zone Ranking", page_icon="🚕", layout="wide")

# All numbers come from small result files committed in outputs/.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


@st.cache_data
def load(name, keep_na=True):
    p = OUT / name
    if not p.exists():
        return None
    return pd.read_csv(p, keep_default_na=keep_na)


def need(*names, keep_na=True):
    dfs = [load(n, keep_na) for n in names]
    missing = [n for n, d in zip(names, dfs) if d is None]
    if missing:
        st.error(f"Missing from outputs/: {', '.join(missing)}. Copy them from the pipeline results and reload.")
        st.stop()
    return dfs[0] if len(dfs) == 1 else dfs


def usd_m(x):
    return f"${x / 1e6:,.1f}M"


MODELS = {
    "hist_gradient_boosting_lag1": "HGB next-hour",
    "hist_gradient_boosting_24h": "HGB 24h-ahead",
    "baseline_lag168": "Same hour last week",
    "baseline_lag24": "Same hour yesterday",
    "baseline_lag1": "Previous hour",
}
BASE = {
    "zone_hour_profile": "Zone-hour profile",
    "trailing_4_week_profile": "Trailing 4-week profile",
    "same_hour_last_week": "Same hour last week",
    "global_popularity": "Global popularity",
}
RANKER_LABEL = "Learned ranker"
BLOCKS = ["Overnight", "Morning", "Midday", "Evening", "Late evening"]

st.sidebar.title("NYC Taxi Analytics")
st.sidebar.caption("Yellow Taxi, Jan to Mar 2025")
page = st.sidebar.radio(
    "Explore",
    ["Overview", "Demand & zones", "Forecasting", "Zone ranking","🚕 Driver Opportunity", "Data quality", "Limits & roadmap"],
)
st.title("🚕 NYC Taxi Demand & Zone Ranking")

# ---------------------------------------------------------------- shared data
kpi = need("kpis.csv").iloc[0]
monthly = need("kpis_monthly.csv")
zones = need("zone_analysis.csv", keep_na=False)
rk = need("ranking_baseline_metrics.csv")
rk_learned = need("ranking_learned_metrics.csv")
rk_learned_pred = need("ranking_learned_predictions.csv")
fp = need("forecast_predictions.csv")



def forecast_scores(d, by):
    d = d.assign(err=d.actual_trips - d.predicted_trips)
    d = d.assign(ae=d.err.abs(), se=d.err**2)
    g = d.groupby(by).agg(n=("ae", "size"), mae=("ae", "mean"), mse=("se", "mean"),
                          ae_sum=("ae", "sum"), y=("actual_trips", "sum"))
    g["rmse"] = np.sqrt(g.mse)
    g["wape"] = g.ae_sum / g.y
    return g[["n", "mae", "rmse", "wape"]]


# Score every model on the same hours so comparisons are like-for-like.
n_models = fp.model.nunique()
counts = fp.groupby("timestamp").model.nunique()
matched = fp[fp.timestamp.isin(counts[counts == n_models].index)]
overall = forecast_scores(fp, "model")
mt = forecast_scores(matched, "model")

rk_overall = rk.groupby("baseline")[["ndcg_at_5", "ndcg_at_10", "ndcg_at_20", "precision_at_10"]].mean()
rk_overall = rk_overall.loc[[b for b in BASE if b in rk_overall.index]]

# ------------------------------------------------------------------- overview
if page == "Overview":
    c = st.columns(4)
    c[0].metric("Clean trips", f"{kpi.trips / 1e6:.2f}M")
    c[1].metric("Days covered", int(kpi.active_days))
    c[2].metric("Gross passenger spend", usd_m(kpi.gross_passenger_spend))
    c[3].metric("Median spend per trip", f"${monthly.median_gross_spend.median():.2f}",
                help="Median of the three monthly medians.")

    z = zones.sort_values("trips", ascending=False)
    top10 = z.trips.head(10).sum() / z.trips.sum()
    n80 = int((z.trips.cumsum() / z.trips.sum() < 0.8).sum() + 1)
    b = zones.groupby("pickup_borough")[["trips", "gross_passenger_spend"]].sum()
    b = b / b.sum()
    tpd = monthly.sort_values("month").trips_per_day
    w168, h1, h24 = mt.loc["baseline_lag168"], mt.loc["hist_gradient_boosting_lag1"], mt.loc["hist_gradient_boosting_24h"]
    best = rk_overall.ndcg_at_10.idxmax()

    st.subheader("What the analysis found")
    st.markdown(
        f"- **Demand per day rose {tpd.iloc[-1] / tpd.iloc[0] - 1:.0%}** from January to March "
        f"({tpd.iloc[0]:,.0f} to {tpd.iloc[-1]:,.0f} trips per day). This is descriptive; the data can't separate season, weather, and policy.\n"
        f"- **Pickups are concentrated.** The top 10 zones carry {top10:.0%} of trips and {n80} zones carry 80%. "
        f"Manhattan has {b.trips.get('Manhattan', 0):.0%} of trips and {b.gross_passenger_spend.get('Manhattan', 0):.0%} of spend.\n"
        f"- **Time-aware ranking beats hour-agnostic popularity:** NDCG@10 of {rk_overall.ndcg_at_10[best]:.3f} "
        f"({BASE[best]}) vs {rk_overall.ndcg_at_10.get('global_popularity', float('nan')):.3f}.\n"
        f"- **Next-hour forecast:** {1 - h1.mae / w168.mae:.0%} lower MAE than same-hour-last-week on {int(h1.n)} matched hours "
        f"(WAPE {h1.wape:.1%} vs {w168.wape:.1%}).\n"
        f"- **24-hour-ahead forecast:** {1 - h24.mae / w168.mae:.0%} lower MAE (WAPE {h24.wape:.1%}). A modest gain; not tested for significance."
    )
    st.info("Gross passenger spend reflects passenger spending, not driver earnings. "
    "The dataset does not include vehicle supply, wait times, cancellations, "
    "or driver costs.")
    with st.expander("About the ranking & spend interpretation"):
        st.markdown(
        """
        Gross passenger spend (`total_amount`) represents passenger spending,
        including fares, tips, tolls, taxes, and applicable surcharges.

        Because the dataset does not contain vehicle supply, wait times,
        cancellations, or driver costs, the zone-ranking results should be
        interpreted as **historical passenger-demand and gross-spend opportunity**,
        not driver earnings, profitability, or utilization.
        """
    )

# ------------------------------------------------------------- demand & zones
elif page == "Demand & zones":
    st.header("Demand and zone economics")
    hp = need("hourly_profile.csv")
    st.subheader("Average trips per hour of day")
    st.line_chart(hp.pivot(index="pickup_hour", columns="day_type", values="avg_daily_trips"), height=300)
    st.caption("Weekdays peak in the evening; weekends run later into the night.")

    st.subheader("Monthly summary")
    st.dataframe(monthly.round(2))

    st.subheader("Zones")
    f1, f2 = st.columns(2)
    boroughs = sorted(zones.pickup_borough.unique())
    pick = f1.multiselect("Borough", boroughs, default=boroughs)
    min_trips = f2.slider("Minimum trips", 0, 50_000, 1_000, step=500)
    sort_by = st.selectbox("Sort by", ["trips", "gross_passenger_spend", "avg_gross_spend_per_trip", "gross_spend_per_mile"])
    zf = zones[zones.pickup_borough.isin(pick) & (zones.trips >= min_trips)].sort_values(sort_by, ascending=False)
    st.dataframe(zf.head(25).round(2))
    st.caption(f"{len(zf)} zones match. Small zones are hidden by default because their averages are unstable.")

    st.subheader("Borough share of trips and spend")
    bs = zones.groupby("pickup_borough")[["trips", "gross_passenger_spend"]].sum()
    st.bar_chart(bs / bs.sum())

# ---------------------------------------------------------------- forecasting
elif page == "Forecasting":
    st.header("Hourly demand forecasting")
    st.markdown("Four one-week test folds in time order (Feb 3, Feb 17, Mar 3, Mar 17), each trained only on earlier data. "
                "The missing 2 AM hour on March 9 (daylight saving) is left missing, not filled with zero.")

    st.subheader(f"Matched comparison ({int(mt.n.iloc[0])} hours scored by every model)")
    t = mt.rename(index=MODELS).sort_values("mae")
    t["MAE vs last week"] = t.mae / mt.loc["baseline_lag168", "mae"] - 1
    st.dataframe(t.drop(columns="n").style.format({"mae": "{:.1f}", "rmse": "{:.1f}", "wape": "{:.2%}", "MAE vs last week": "{:+.1%}"}))
    st.caption("Same-hour-last-week is the fair baseline for both models. Previous-hour persistence is a weak benchmark.")

    st.subheader("Error by hour of day (WAPE)")
    byh = forecast_scores(matched, ["hour", "model"]).wape.unstack()
    st.line_chart(byh[["hist_gradient_boosting_lag1", "hist_gradient_boosting_24h", "baseline_lag168"]].rename(columns=MODELS), height=280)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By fold (MAE)")
        st.dataframe(forecast_scores(matched, ["fold", "model"]).mae.unstack().rename(columns=MODELS).round(1))
    with c2:
        st.subheader("Weekday vs weekend (WAPE)")
        st.dataframe(forecast_scores(matched, ["day_type", "model"]).wape.unstack().rename(columns=MODELS).style.format("{:.2%}"))

    st.subheader("Actual vs predicted")
    fold = st.selectbox("Test week", sorted(fp.fold.unique()))
    w = fp[fp.fold == fold].assign(timestamp=lambda d: pd.to_datetime(d.timestamp))
    pv = w.pivot_table(index="timestamp", columns="model", values="predicted_trips")
    pv["Actual"] = w.groupby("timestamp").actual_trips.first()
    st.line_chart(pv[["Actual", "hist_gradient_boosting_lag1", "hist_gradient_boosting_24h"]].rename(columns=MODELS), height=300)

    st.warning("Offline evaluation on one quarter. Four folds are too few for a significance claim, and Presidents' Day is the only holiday in the test weeks.")

# --------------------------------------------------------------- zone ranking
elif page == "Zone ranking":
    st.header("Pickup-zone opportunity ranking")
    st.markdown("Explore historical pickup-zone opportunity and the offline performance "
    "of a learned ranking model. Scores represent gross passenger-spend "
    "opportunity, not driver earnings.")

    st.subheader("Try it")
    prof = need("zone_hour_profile.csv", keep_na=False)
    c = st.columns(4)
    day_type = c[0].radio("Day type", ["Weekday", "Weekend"], horizontal=True)
    hour = c[1].slider("Pickup hour", 0, 23, 19)
    min_t = c[2].number_input("Minimum zone trips", 0, 100_000, 1_000, step=500)
    skip_air = c[3].checkbox("Exclude airports")
    d = prof[(prof.day_type == day_type) & (prof.hour == hour)].merge(
        zones[["PULocationID", "trips", "avg_gross_spend_per_trip"]], on="PULocationID")
    d = d[d.trips >= min_t]
    if skip_air:
        d = d[~d.pickup_zone.str.contains("Airport", case=False)]
    d["opportunity_score"] = d.avg_daily_trips * d.avg_gross_spend_per_trip
    top = d.nlargest(10, "opportunity_score").reset_index(drop=True)
    top.index += 1
    st.dataframe(top[["pickup_zone", "pickup_borough", "avg_daily_trips", "avg_gross_spend_per_trip", "opportunity_score"]].round(1))
    st.caption("Score = expected trips in that hour x average gross spend per trip. It measures total spend in the zone, "
               "not what one driver would earn: airport queues and competition are not observed.")

        # -----------------------------------------------------------
    # Learned ranker — offline evaluation
    # -----------------------------------------------------------

    st.divider()
    st.subheader("Learned ranker — offline evaluation")

    learned_overall = (
        rk_learned
        .groupby("baseline")[
            ["ndcg_at_5", "ndcg_at_10", "ndcg_at_20", "precision_at_10"]
        ]
        .mean()
    )

    lr = learned_overall.loc["learned_ranker"]

    baseline_best = rk_overall.ndcg_at_10.idxmax()
    baseline_ndcg = rk_overall.loc[baseline_best, "ndcg_at_10"]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "NDCG@10",
        f"{lr.ndcg_at_10:.3f}",
        f"+{lr.ndcg_at_10 - baseline_ndcg:.3f} vs best baseline"
    )

    c2.metric(
        "Precision@10",
        f"{lr.precision_at_10:.1%}"
    )

    c3.metric(
        "NDCG@5",
        f"{lr.ndcg_at_5:.3f}"
    )

    st.caption(
        f"Four rolling-origin test folds. Strongest baseline: "
        f"{BASE.get(baseline_best, baseline_best)} "
        f"(NDCG@10 = {baseline_ndcg:.3f})."
    )
    st.subheader("Fold consistency")

    fold_df = (
        rk_learned[
            rk_learned["baseline"] == "learned_ranker"
        ]
        .groupby("fold")[
            ["ndcg_at_5", "ndcg_at_10", "ndcg_at_20", "precision_at_10"]
        ]
        .mean()
    )

    st.dataframe(
        fold_df.style.format({
            "ndcg_at_5": "{:.3f}",
            "ndcg_at_10": "{:.3f}",
            "ndcg_at_20": "{:.3f}",
            "precision_at_10": "{:.1%}",
        }),
        use_container_width=True,
    )
        # -----------------------------------------------------------
    # Held-out prediction explorer
    # -----------------------------------------------------------

    st.divider()
    st.subheader("Explore a held-out prediction")

    st.caption(
        "These are predictions generated on historical test folds. "
        "Use this view to compare the model's predicted pickup-zone "
        "gross-spend ranking with what actually occurred."
    )

    learned_preds = rk_learned_pred[
        rk_learned_pred["baseline"] == "learned_ranker"
    ].copy()

    # Make sure dates are comparable
    learned_preds["test_date"] = pd.to_datetime(
        learned_preds["test_date"]
    )

    # -------------------------
    # Select fold
    # -------------------------

    fold_options = sorted(
        learned_preds["fold"].dropna().unique()
    )

    selected_fold = st.selectbox(
        "Evaluation fold",
        fold_options,
        format_func=lambda x: f"Fold {int(x)}"
    )

    fold_data = learned_preds[
        learned_preds["fold"] == selected_fold
    ].copy()

    # -------------------------
    # Select date
    # -------------------------

    date_options = sorted(
        fold_data["test_date"].dt.date.unique()
    )

    selected_date = st.selectbox(
        "Test date",
        date_options
    )

    date_data = fold_data[
        fold_data["test_date"].dt.date == selected_date
    ].copy()

    # -------------------------
    # Select hour
    # -------------------------

    hour_options = sorted(
        date_data["pickup_hour"].unique()
    )

    selected_hour = st.selectbox(
        "Pickup hour",
        hour_options,
        format_func=lambda x: f"{int(x):02d}:00"
    )

    example = date_data[
        date_data["pickup_hour"] == selected_hour
    ].copy()

    # -------------------------
    # Display predictions
    # -------------------------

    if example.empty:

        st.info(
            "No saved learned-ranker predictions are available "
            "for this date and hour."
        )

    else:

        # Predictions are already saved as ranked observations.
        # Sort explicitly so the UI always shows highest predicted
        # opportunity first.
        example = example.sort_values(
            "predicted_gross_passenger_spend",
            ascending=False
        ).copy()

        example["model_rank"] = (
            np.arange(len(example)) + 1
        )

        # Actual rank based on realized gross spend.
        example["actual_rank"] = (
            example[
                "actual_gross_passenger_spend"
            ]
            .rank(
                method="min",
                ascending=False
            )
            .astype(int)
        )

        display = example.head(10)[
            [
                "model_rank",
                "pickup_zone",
                "pickup_borough",
                "predicted_gross_passenger_spend",
                "actual_gross_passenger_spend",
                "actual_rank",
            ]
        ].copy()

        display.columns = [
            "Model rank",
            "Pickup zone",
            "Borough",
            "Predicted gross spend",
            "Actual gross spend",
            "Actual rank",
        ]

        st.dataframe(
            display.style.format({
                "Predicted gross spend": "${:,.0f}",
                "Actual gross spend": "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------
        # Summary metrics
        # -------------------------

        selected_metrics = rk_learned[
            (rk_learned["baseline"] == "learned_ranker")
            & (rk_learned["fold"] == selected_fold)
            & (pd.to_datetime(rk_learned["test_date"]).dt.date == selected_date)
            & (rk_learned["pickup_hour"] == selected_hour)
        ]

        if not selected_metrics.empty:

            metric_row = selected_metrics.iloc[0]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "NDCG@10",
                f"{metric_row.ndcg_at_10:.3f}"
            )

            c2.metric(
                "Precision@10",
                f"{metric_row.precision_at_10:.1%}"
            )

            actual_top_zone = example.loc[
            example["actual_gross_passenger_spend"].idxmax(),
            "pickup_zone"
        ]

        c3.metric(
            "Actual top-ranked zone",
            actual_top_zone
        )

        st.caption(
            "Gross spend represents passenger spending in the "
            "pickup zone during the selected hour; it is not "
            "driver earnings or profit."
        )

    st.subheader("Baseline comparison")
    st.dataframe(rk_overall.rename(index=BASE).round(3))
    st.caption("Hit rate@10 is omitted: it is 1.0 for almost every method, so it does not separate them.")

    st.subheader("NDCG@10 by fold")
    st.dataframe(rk.pivot_table(index="baseline", columns="fold", values="ndcg_at_10").rename(index=BASE).round(3))

    st.subheader("NDCG@10 by time of day")
    tb = rk.pivot_table(index="time_block", columns="baseline", values="ndcg_at_10").reindex(BLOCKS)
    st.bar_chart(tb.rename(columns=BASE), height=300)
    if {"global_popularity", "zone_hour_profile"} <= set(tb.columns):
        gain = (tb.zone_hour_profile - tb.global_popularity)
        st.caption(f"Time-awareness helps most in {gain.idxmax().lower()} hours (+{gain.max():.2f}) and least in {gain.idxmin().lower()} hours (+{gain.min():.2f}).")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Weekday vs weekend")
        st.dataframe(rk.pivot_table(index="baseline", columns="day_type", values="ndcg_at_10").rename(index=BASE).round(3))
    with c2:
        ap = load("ranking_airport_metrics.csv")
        if ap is not None:
            st.subheader("Airport vs other zones")
            st.dataframe(ap.pivot_table(index="baseline", columns="airport_group", values="ndcg_at_10").rename(index=BASE).round(3))
            st.caption("Only a few airport zones exist, so their NDCG is near 1 for every method.")

    st.info("The profile baselines are close; a learned ranker is worthwhile only if it clearly beats them.")

# ---------------------------------------------------------------
# driver opportunity
elif page == "🚕 Driver Opportunity":

    st.header("🚕 Driver Opportunity Explorer")

    st.markdown(
        "Explore pickup zones with the strongest historical opportunity "
        "for a planned driving period. Personalize the ranking based on "
        "whether you prioritize trip volume, higher-value trips, or a balance."
    )

    st.info(
        "This tool estimates pickup and gross passenger-spend opportunity "
        "from historical Yellow Taxi data. It does not estimate driver "
        "earnings or profit."
    )

    # -----------------------------------------------------------
    # Driver preferences
    # -----------------------------------------------------------

    st.subheader("Plan your drive")

    c1, c2, c3 = st.columns(3)

    with c1:
        day_type = st.selectbox(
            "Day type",
            ["Weekday", "Weekend"],
        )

    with c2:
        hour = st.slider(
            "Pickup hour",
            min_value=0,
            max_value=23,
            value=18,
            format="%d:00",
        )

    with c3:
        objective = st.selectbox(
            "What's your priority?",
            [
                "Balanced",
                "More trips",
                "Higher-value trips",
            ],
        )

    # -----------------------------------------------------------
    # Optional airport preference
    # -----------------------------------------------------------

    c4, c5 = st.columns(2)

    with c4:
        exclude_airports = st.checkbox(
            "Exclude airport pickup zones"
        )

    with c5:
        top_n = st.selectbox(
            "Show top zones",
            [5, 10, 15],
            index=1,
        )

    # -----------------------------------------------------------
    # Load historical zone-hour profile
    # -----------------------------------------------------------

    prof = need(
        "zone_hour_profile.csv",
        keep_na=False,
    )

    d = prof[
        (prof["day_type"] == day_type)
        & (prof["hour"] == hour)
    ].copy()

    # -----------------------------------------------------------
    # Join overall zone statistics
    # -----------------------------------------------------------

    d = d.merge(
        zones[
            [
                "PULocationID",
                "trips",
                "avg_gross_spend_per_trip",
            ]
        ],
        on="PULocationID",
        how="left",
        suffixes=("", "_overall"),
    )

    # -----------------------------------------------------------
    # Airport filter
    # -----------------------------------------------------------

    if exclude_airports:
        d = d[
            ~d["pickup_zone"].str.contains(
                "Airport",
                case=False,
                na=False,
            )
        ]

    # -----------------------------------------------------------
    # Build interpretable opportunity measures
    # -----------------------------------------------------------

    # Historical expected pickup volume for this
    # day type / hour.
    d["trip_opportunity"] = d["avg_daily_trips"]

    # Historical gross passenger-spend opportunity.
    d["gross_spend_opportunity"] = (
        d["avg_daily_trips"]
        * d["avg_gross_spend_per_trip"]
    )

    # -----------------------------------------------------------
    # Normalize components for personalization
    # -----------------------------------------------------------

    def normalize(series):

        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:
            return pd.Series(
                1.0,
                index=series.index,
            )

        return (
            (series - minimum)
            / (maximum - minimum)
        )

    d["trip_score"] = normalize(
        d["trip_opportunity"]
    )

    d["spend_score"] = normalize(
        d["gross_spend_opportunity"]
    )

    # -----------------------------------------------------------
    # Personalized objective
    # -----------------------------------------------------------

    if objective == "More trips":

        d["opportunity_score"] = (
            0.75 * d["trip_score"]
            + 0.25 * d["spend_score"]
        )

        objective_description = (
            "prioritizes historical pickup volume"
        )

    elif objective == "Higher-value trips":

        d["opportunity_score"] = (
            0.25 * d["trip_score"]
            + 0.75 * d["spend_score"]
        )

        objective_description = (
            "prioritizes historical gross passenger-spend opportunity"
        )

    else:

        d["opportunity_score"] = (
            0.50 * d["trip_score"]
            + 0.50 * d["spend_score"]
        )

        objective_description = (
            "balances historical pickup volume and "
            "gross passenger-spend opportunity"
        )

    # -----------------------------------------------------------
    # Rank zones
    # -----------------------------------------------------------

    recommendations = (
        d
        .sort_values(
            "opportunity_score",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    recommendations.insert(
        0,
        "Rank",
        range(
            1,
            len(recommendations) + 1,
        ),
    )

    # -----------------------------------------------------------
    # Headline recommendation
    # -----------------------------------------------------------

    if not recommendations.empty:

        best = recommendations.iloc[0]

        st.subheader("🏆 Top pickup opportunity")

        st.markdown(
            f"### {best['pickup_zone']}"
        )

        st.caption(
            f"{day_type} at {int(hour):02d}:00 · "
            f"{objective_description}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Expected pickups",
            f"{best['avg_daily_trips']:,.0f}",
        )

        c2.metric(
            "Avg. spend / trip",
            f"${best['avg_gross_spend_per_trip']:,.2f}",
        )

        c3.metric(
            "Gross spend opportunity",
            f"${best['gross_spend_opportunity']:,.0f}",
        )

        c4.metric(
            "Borough",
            best["pickup_borough"],
        )

    # -----------------------------------------------------------
    # Recommendation table
    # -----------------------------------------------------------

    st.subheader("Recommended pickup zones")

    display = recommendations[
        [
            "Rank",
            "pickup_zone",
            "pickup_borough",
            "avg_daily_trips",
            "avg_gross_spend_per_trip",
            "gross_spend_opportunity",
        ]
    ].copy()

    display.columns = [
        "Rank",
        "Pickup zone",
        "Borough",
        "Expected pickups",
        "Avg. spend / trip",
        "Gross spend opportunity",
    ]

    st.dataframe(
        display.style.format({
            "Expected pickups": "{:,.0f}",
            "Avg. spend / trip": "${:,.2f}",
            "Gross spend opportunity": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------------
    # Why this recommendation?
    # -----------------------------------------------------------

    st.subheader("Why these zones?")

    st.markdown(
        f"""
        The ranking for **{day_type.lower()}s at {int(hour):02d}:00**
        {objective_description}.

        The opportunity score combines:

        - **Historical pickup volume** — how many pickups the zone
          typically receives during the selected period.
        - **Gross passenger-spend opportunity** — expected pickup
          volume multiplied by average gross spend per trip.
        - **Your selected priority** — changes how much each component
          contributes to the ranking.
        """
    )

    # -----------------------------------------------------------
    # Important limitation
    # -----------------------------------------------------------

    st.warning(
        "Driver earnings are not directly estimated. The TLC dataset "
        "does not observe the number of competing drivers, passenger "
        "wait times, trip acceptance, deadhead distance, fuel costs, "
        "or other operating costs. Therefore, a high gross-spend "
        "opportunity does not necessarily mean higher driver profit."
    )

# --------------------------------------------------------------- data quality
elif page == "Data quality":
    st.header("Data quality")
    dq = need("data_quality_summary.csv")
    st.dataframe(dq)
    tot_raw, tot_clean = dq.raw_trip_count.sum(), dq.clean_trip_count.sum()
    st.metric("Records kept", f"{tot_clean / tot_raw:.1%}", help=f"{tot_clean:,} of {tot_raw:,}")

    files = sorted(OUT.glob("*_first_failure.csv"))
    if files:
        ff = pd.concat([pd.read_csv(f) for f in files])
        ff["month"] = ff.file.str.extract(r"(\d{4}-\d{2})")
        bad = ff[ff.first_failing_rule != "valid"]
        piv = bad.pivot_table(index="first_failing_rule", columns="month", values="records", aggfunc="sum", fill_value=0)
        share = piv.sum(axis=1).sort_values(ascending=False)
        st.subheader("Why records were excluded")
        st.bar_chart(piv.loc[share.index])
        st.markdown(
            f"- **{share.index[0].replace('_', ' ')}** is the largest cause ({share.iloc[0] / share.sum():.0%} of exclusions), "
            f"followed by **{share.index[1].replace('_', ' ')}** ({share.iloc[1] / share.sum():.0%}).\n"
            "- Each record is counted once, under the first rule it fails. Overlapping failures are tracked separately in the breakdown files."
        )
        if "zero_or_negative_duration" in piv.index:
            d = piv.loc["zero_or_negative_duration"]
            st.markdown(f"- Non-positive durations rose from {d.iloc[0]:,} to {d.iloc[-1]:,} records across the quarter and are worth investigating by vendor and date.")
    else:
        st.info("Add the *_first_failure.csv files to outputs/ to see the exclusion breakdown.")

# ----------------------------------------------------------- limits & roadmap
else:
    st.header("Limits and roadmap")
    st.subheader("What the data cannot show")
    st.markdown(
        "- No vehicle supply, wait times, cancellations or driver costs, so no claims about earnings or utilization.\n"
        "- Yellow taxis only; app-based and green-taxi trips are excluded, so outer-borough coverage reflects that.\n"
        "- One quarter of data; no weather, event or holiday features in the models.\n"
        "- Ranking and forecasts are evaluated offline; no evidence of real-world benefit."
    )
    st.subheader("Next steps")
    st.markdown(
        "1. Learned ranker, kept honest against the profile baselines.\n"
        "2. Add December 2024 for a congestion-pricing before/after analysis.\n"
        "3. Zone-level demand forecasts feeding the ranker.\n"
        "4. Weather and holiday features for the 24-hour forecast."
    )
    st.caption("Suggested wording: historical pickup-zone ranking and hourly demand forecasting on NYC TLC Yellow Taxi data.")

st.markdown("---")
st.caption("NYC TLC Yellow Taxi, Jan to Mar 2025. Descriptive analytics with offline forecasting and ranking evaluation.")