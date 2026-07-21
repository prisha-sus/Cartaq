-- ============================================================
-- Cartaq Role 3 — ClickHouse Schema Initialization
-- Runs automatically on first container start via
-- /docker-entrypoint-initdb.d/00_init.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS cartaq;

-- ── 1. Forecast Log ──────────────────────────────────────────
-- Receives output from Role 1 (ST-GNN A3TGCN predictions).
-- Until real predictions are ready, populated from generate_fake_data.py.
CREATE TABLE IF NOT EXISTS cartaq.forecast_log
(
    h3_index            String       COMMENT 'H3 cell ID at resolution 8',
    future_timestamp    DateTime     COMMENT 'UTC forecasted timestamp',
    predicted_aqi       Float32      COMMENT 'AQI prediction (same scale as OpenAQ input)',
    horizon_hours       UInt8        COMMENT '24, 48, or 72',
    run_timestamp       DateTime     DEFAULT now() COMMENT 'When this prediction was generated',
    city                LowCardinality(String) DEFAULT 'pune',
    model_version       LowCardinality(String) DEFAULT 'a3tgcn_v1'
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(future_timestamp)
ORDER BY (h3_index, future_timestamp)
SETTINGS index_granularity = 8192;

-- ── 2. Attribution Log ───────────────────────────────────────
-- Receives output from Role 2 (DoWhy causal attribution engine).
-- Stores ATE + confidence intervals per source type per hex.
CREATE TABLE IF NOT EXISTS cartaq.attribution_log
(
    h3_index            String       COMMENT 'H3 cell ID at resolution 8',
    event_timestamp     DateTime     COMMENT 'UTC timestamp of the pollution spike',
    source_type         LowCardinality(String) COMMENT 'factory | traffic | construction | domestic | transport',
    ate                 Float32      COMMENT 'Average Treatment Effect (AQI delta from DoWhy)',
    confidence_lower    Float32      COMMENT '95% CI lower bound',
    confidence_upper    Float32      COMMENT '95% CI upper bound',
    wind_speed_kmh      Float32,
    is_source_active    UInt8        COMMENT '1=active, 0=inactive during event',
    run_timestamp       DateTime     DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_timestamp)
ORDER BY (h3_index, event_timestamp, source_type)
SETTINGS index_granularity = 8192;

-- ── 3. Dispatch Log ──────────────────────────────────────────
-- Receives output from Role 4 (dispatch orchestrator).
-- Tracks each advisory sent per hex.
CREATE TABLE IF NOT EXISTS cartaq.dispatch_log
(
    h3_index            String,
    facility_name       String,
    facility_category   LowCardinality(String),
    predicted_aqi       Float32,
    horizon_hours       UInt8,
    call_status         LowCardinality(String)  DEFAULT 'dry_run',
    mqtt_published      UInt8                   DEFAULT 0,
    dispatched_at       DateTime                DEFAULT now(),
    end_to_end_ms       Float32                 COMMENT 'End-to-end dispatch latency in milliseconds'
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(dispatched_at)
ORDER BY (dispatched_at, h3_index)
SETTINGS index_granularity = 8192;

-- ── 4. Materialized View: Hourly H3 AQI Aggregates ───────────
-- Drives the Superset deck.gl hexagonal heatmap.
-- Aggregates forecast_log into hourly buckets per hex.
CREATE MATERIALIZED VIEW IF NOT EXISTS cartaq.mv_hourly_aqi
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(hour_bucket)
ORDER BY (h3_index, hour_bucket)
POPULATE
AS
SELECT
    h3_index,
    toStartOfHour(future_timestamp)     AS hour_bucket,
    avgState(predicted_aqi)             AS avg_aqi_state,
    maxState(predicted_aqi)             AS max_aqi_state,
    minState(predicted_aqi)             AS min_aqi_state,
    countState()                        AS n_forecasts_state
FROM cartaq.forecast_log
GROUP BY h3_index, hour_bucket;

-- Companion SELECT view to query the MV cleanly
CREATE VIEW IF NOT EXISTS cartaq.hourly_aqi AS
SELECT
    h3_index,
    hour_bucket,
    avgMerge(avg_aqi_state)     AS avg_aqi,
    maxMerge(max_aqi_state)     AS max_aqi,
    minMerge(min_aqi_state)     AS min_aqi,
    countMerge(n_forecasts_state) AS n_forecasts
FROM cartaq.mv_hourly_aqi
GROUP BY h3_index, hour_bucket;

-- ── 5. Materialized View: Daily Attribution Summary ──────────
-- Aggregates ATE per source per day for Superset bar charts.
CREATE MATERIALIZED VIEW IF NOT EXISTS cartaq.mv_daily_attribution
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(day_bucket)
ORDER BY (source_type, day_bucket)
POPULATE
AS
SELECT
    source_type,
    toDate(event_timestamp)             AS day_bucket,
    avgState(ate)                       AS avg_ate_state,
    quantileState(0.95)(ate)            AS p95_ate_state,
    countState()                        AS n_events_state,
    sumState(is_source_active)          AS active_count_state
FROM cartaq.attribution_log
GROUP BY source_type, day_bucket;
