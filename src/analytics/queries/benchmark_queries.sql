-- ============================================================
-- Cartaq Role 3 — ClickHouse Benchmark Query Suite
--
-- Purpose: Validate query latency on 5M-row tables and provide
--          copy-pasteable SQL for Superset chart configuration.
--
-- Run in: ClickHouse Play (http://localhost:8123/play)
--         or any ClickHouse SQL client pointed at localhost:9000
-- ============================================================


-- ── 1. Full-table count ──────────────────────────────────────
-- Expected: <50 ms on 5M rows (columnar count is near-instant).
SELECT count() AS total_forecast_rows
FROM cartaq.forecast_log;


-- ── 2. Top-20 hexes by average AQI ──────────────────────────
-- Columnar aggregation — baseline for all dashboard hex-heatmap queries.
-- Expected: <500 ms on 5M rows without index tuning.
SELECT
    h3_index,
    round(avg(predicted_aqi), 1)  AS avg_aqi,
    round(max(predicted_aqi), 1)  AS max_aqi,
    count()                        AS n_forecasts
FROM cartaq.forecast_log
GROUP BY h3_index
ORDER BY avg_aqi DESC
LIMIT 20;


-- ── 3. Daily peak AQI per hex (time-range filter) ────────────
-- Simulates the Superset "date filter" interaction on the dashboard.
-- The partition pruning on toYYYYMM(future_timestamp) is exercised here.
SELECT
    h3_index,
    toDate(future_timestamp)      AS day,
    max(predicted_aqi)            AS daily_peak_aqi,
    round(avg(predicted_aqi), 1)  AS daily_avg_aqi
FROM cartaq.forecast_log
WHERE future_timestamp BETWEEN '2026-01-01 00:00:00' AND '2026-07-01 00:00:00'
GROUP BY h3_index, day
ORDER BY day DESC, daily_peak_aqi DESC
LIMIT 100;


-- ── 4. Attribution: ranked sources by causal impact ─────────
-- Powers the "Source Attribution" bar chart in Superset.
SELECT
    source_type,
    round(avg(ate), 2)              AS mean_ate_aqi_pts,
    round(quantile(0.95)(ate), 2)   AS p95_ate,
    round(avg(confidence_upper - confidence_lower), 2) AS avg_ci_width,
    countIf(is_source_active = 1)   AS active_events,
    count()                         AS total_events
FROM cartaq.attribution_log
GROUP BY source_type
ORDER BY mean_ate_aqi_pts DESC;


-- ── 5. Spike detection: hexes with ≥3 high-AQI events/day ───
-- "Hot-spot" query — identifies chronic polluter zones.
SELECT
    h3_index,
    toDate(future_timestamp)             AS day,
    countIf(predicted_aqi > 150)         AS spike_count,
    round(max(predicted_aqi), 1)         AS peak_aqi
FROM cartaq.forecast_log
GROUP BY h3_index, day
HAVING spike_count >= 3
ORDER BY spike_count DESC, peak_aqi DESC
LIMIT 50;


-- ── 6. Cross-model comparison ────────────────────────────────
-- Compares a3tgcn_v1, a3tgcn_v2, and persistence_baseline
-- using a synthetic ground-truth of 100 AQI for RMSE calculation.
-- Replace 100 with avg(actual_aqi) once Role 1 delivers real labels.
SELECT
    model_version,
    count()                                              AS n,
    round(avg(predicted_aqi), 2)                        AS mean_predicted,
    round(sqrt(avg(pow(predicted_aqi - 100, 2))), 2)    AS pseudo_rmse_vs_100
FROM cartaq.forecast_log
GROUP BY model_version
ORDER BY pseudo_rmse_vs_100 ASC;


-- ── 7. Deck.gl H3 heatmap query ──────────────────────────────
-- Paste this SQL directly into a Superset "Deck.gl H3 Hexagon" chart.
-- Chart config:
--   H3 Hexagon Index col → h3_index
--   Metric              → avg_aqi  (already pre-aggregated below)
SELECT
    h3_index,
    round(avg(predicted_aqi), 1) AS avg_aqi
FROM cartaq.forecast_log
WHERE future_timestamp BETWEEN '{{since}}' AND '{{until}}'
GROUP BY h3_index
ORDER BY avg_aqi DESC;


-- ── 8. Hourly AQI from materialized view (fast path) ─────────
-- Reads from cartaq.hourly_aqi MV — sub-second even at 5M rows.
SELECT
    hour_bucket,
    round(avg(avg_aqi), 1)  AS mean_aqi,
    round(max(max_aqi), 1)  AS peak_aqi,
    sum(n_forecasts)         AS total_forecasts
FROM cartaq.hourly_aqi
WHERE hour_bucket >= (now() - INTERVAL 7 DAY)
GROUP BY hour_bucket
ORDER BY hour_bucket DESC;


-- ── 9. Attribution confidence interval coverage ──────────────
-- Flags hexes where the causal estimate CI width is too wide (>40 pts),
-- indicating low statistical confidence — useful for dispatch filtering.
SELECT
    h3_index,
    source_type,
    round(avg(ate), 2)                                        AS mean_ate,
    round(avg(confidence_upper - confidence_lower), 2)        AS avg_ci_width,
    countIf((confidence_upper - confidence_lower) > 40)       AS wide_ci_events,
    count()                                                    AS total_events
FROM cartaq.attribution_log
GROUP BY h3_index, source_type
HAVING avg_ci_width > 25
ORDER BY avg_ci_width DESC
LIMIT 30;


-- ── 10. Ingestion throughput health check ────────────────────
-- Checks how many rows landed in the last run (by run_timestamp).
SELECT
    toStartOfMinute(run_timestamp)  AS ingest_minute,
    count()                         AS rows_ingested
FROM cartaq.forecast_log
GROUP BY ingest_minute
ORDER BY ingest_minute DESC
LIMIT 10;
