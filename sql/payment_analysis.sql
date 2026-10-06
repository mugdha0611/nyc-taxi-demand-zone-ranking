SELECT
    payment_type,
    COUNT(*) AS trips,
    ROUND(AVG(total_amount), 2) AS avg_total_amount,
    ROUND(AVG(tip_amount), 2) AS avg_tip_amount,
    ROUND(AVG(tip_rate) FILTER (WHERE payment_type = 1), 4) AS avg_card_tip_rate
FROM read_parquet('data/processed/yellow_tripdata_*_clean.parquet')
GROUP BY 1
ORDER BY trips DESC;
