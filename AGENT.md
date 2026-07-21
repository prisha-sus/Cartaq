# Agent Prompt — Module: 24–72 Hour AQI Forecasting (ST-GNN)

## Module Overview
This module builds a 24–72 hour Air Quality Index forecasting system using an Attention Temporal Graph Convolutional Network (A3TGCN) via PyTorch Geometric Temporal. It processes spatial data mapped to an H3 hexagonal grid.

**What it does:**
- Ingests air quality data from OpenAQ and meteorological data from Open-Meteo.
- Maps spatial data to an H3 resolution 8 grid.
- Builds an unweighted k-ring adjacency graph.
- Trains an A3TGCN model to predict AQI at +24h, +48h, and +72h.

**What it explicitly does NOT do:**
- It does not handle source attribution, intervention simulation, alerting, or UI dashboards.
- Currently, wind-direction edge weighting is NOT implemented (edges are simple k-ring adjacency).

---

## Current State (All Roles)

### Role 1: ST-GNN AQI Forecasting
- [x] Initialized Git repository on branch `feature/stgnn-aqi-forecasting`.
- [x] Zero-blocker contract published (dummy data schema available at `output/dummy_forecast.parquet`).
- [x] Ingestion pipeline (`src/ingestion/openaq_client.py`, `src/ingestion/openmeteo_client.py`) built.
- [x] H3 mapper + adjacency matrix builder (`src/graph/h3_mapper.py`) built.
- [x] A3TGCN model (`src/model/a3tgcn.py`) and persistence baseline (`src/model/baseline.py`) built.
- [x] Training loop (`src/train.py`) with temporal split and metrics.json output built.
- [ ] **`src/graph/graph_builder.py` MISSING** — connects ingestion output → H3 tensor pipeline.
- [ ] **Training uses dummy random data** — graph_builder.py needed to use real ingested data.
- [ ] Real `output/forecast.parquet` not yet generated.

### Role 2: Causal AI & GIS Automation
- [x] Basic DoWhy ATE engine (`src/analysis/causal_engine.py`) on branch `feat/causal-engine`.
- [x] Synthetic causal dataset (`src/utils/ce_mock_data.py`) on branch `feat/causal-engine`.
- [ ] `feat/causal-engine` branch NOT merged to main.
- [ ] **Folium GIS headless pipeline** (intervention maps) MISSING.
- [ ] **WeasyPrint PDF dossier generation** MISSING.
- [ ] Confidence intervals not reported (only point estimate `estimate.value` returned).
- [ ] Document generation speed metrics MISSING.

### Role 3: Big Data Analytics & BI ✅ COMPLETE
- [x] ClickHouse schema (`infra/clickhouse/init.sql`) — 3 tables + 2 materialized views.
- [x] Apache Superset config (`infra/superset/superset_config.py`) with deck.gl H3 hexagon flag.
- [x] Docker Compose (`infra/docker-compose.yml`) — ClickHouse + Superset + PostGIS + Mosquitto.
- [x] 5M-row fake data generator (`src/analytics/generate_fake_data.py`).
- [x] ClickHouse ingestion pipeline (`src/analytics/ingest_to_clickhouse.py`).
- [x] Benchmark query suite (`src/analytics/queries/benchmark_queries.sql`) — 10 queries.
- [x] Metrics output to `metrics_role3.json`.

### Role 4: Agentic NLP & Systems Integration ✅ COMPLETE
- [x] Mock high-AQI event payload (`src/dispatch/mock_payload.json`) — 5 events.
- [x] PostGIS schema (`src/dispatch/postgis_setup.sql`) — 14 seeded Pune POIs, 3 tables.
- [x] Spatial query engine (`src/dispatch/spatial_query.py`) — ST_Intersects, graceful mock fallback.
- [x] IndicTrans2 translation pipeline (`src/dispatch/translation_pipeline.py`) — EN→HI, EN→MR.
- [x] Twilio dispatcher (`src/dispatch/twilio_dispatcher.py`) — DRY-RUN safe by default.
- [x] MQTT publisher (`src/dispatch/mqtt_publisher.py`) — retained advisory + compact IoT topics.
- [x] End-to-end orchestrator (`src/dispatch/dispatch_orchestrator.py`) with latency metrics.
- [x] Metrics output to `metrics_role4.json`.

---

## Zero-Blocker Contract (Output Schema)
The output of this module is a parquet file located at `output/forecast.parquet`
(currently using `output/dummy_forecast.parquet` as a placeholder).

The exact schema is:
- `h3_index` (string): H3 cell ID at resolution 8.
- `future_timestamp` (datetime): UTC, the forecasted point in time.
- `predicted_aqi` (float): model output, same AQI scale as ingested data.

---

## H3 Resolution
**Resolution 8** is chosen because it offers an appropriate balance between spatial
granularity (~0.73 km² area, ~461m edge length) and graph sparsity for city-scale
forecasting (e.g., Pune).

---

## Key File Locations

### Role 1
- `src/ingestion/openaq_client.py` — OpenAQ v3 API client
- `src/ingestion/openmeteo_client.py` — Open-Meteo historical weather client
- `src/graph/h3_mapper.py` — H3 indexing + k-ring adjacency builder
- `src/model/a3tgcn.py` — A3TGCN model definition
- `src/model/baseline.py` — Persistence baseline
- `src/train.py` — Training loop (currently uses dummy data)
- `src/create_dummy_contract.py` — Generates `output/dummy_forecast.parquet`
- `output/dummy_forecast.parquet` — Zero-blocker output (run create_dummy_contract.py to generate)
- `metrics.json` — Training metrics (RMSE vs baseline)

### Role 2 (branch: feat/causal-engine)
- `src/analysis/causal_engine.py` — DoWhy ATE calculation
- `src/utils/ce_mock_data.py` — Synthetic causal dataset generator

### Role 3
- `infra/docker-compose.yml` — All infrastructure services
- `infra/clickhouse/init.sql` — ClickHouse schema (tables + materialized views)
- `infra/superset/superset_config.py` — Apache Superset configuration
- `src/analytics/generate_fake_data.py` — 5M-row synthetic data generator
- `src/analytics/ingest_to_clickhouse.py` — Parquet → ClickHouse ingestion pipeline
- `src/analytics/queries/benchmark_queries.sql` — 10 benchmark queries
- `metrics_role3.json` — Ingestion throughput + query latency metrics

### Role 4
- `infra/docker-compose.yml` — PostGIS + Mosquitto services (shared with Role 3)
- `infra/mosquitto/mosquitto.conf` — Eclipse Mosquitto broker config
- `src/dispatch/mock_payload.json` — Mock high-AQI forecast events
- `src/dispatch/postgis_setup.sql` — PostGIS schema + 14 Pune vulnerability POIs
- `src/dispatch/spatial_query.py` — PostGIS ST_Intersects spatial intersection
- `src/dispatch/translation_pipeline.py` — AI4Bharat IndicTrans2 EN→Indic pipeline
- `src/dispatch/twilio_dispatcher.py` — Twilio automated voice call dispatcher
- `src/dispatch/mqtt_publisher.py` — Eclipse Mosquitto MQTT publisher
- `src/dispatch/dispatch_orchestrator.py` — End-to-end pipeline + metrics
- `metrics_role4.json` — Dispatch latency metrics

---

## Known Limitations
- Role 1: `graph_builder.py` is MISSING — `train.py` uses random dummy tensors, not real data.
- Role 1: Wind-direction edge weighting is NOT implemented; edges are unweighted k-ring adjacency.
- Role 2: Folium + WeasyPrint dossier pipeline is NOT built yet.
- Role 2: ATE confidence intervals are not reported (only point estimate).
- Role 3: Superset must have `clickhouse-connect` installed manually (done in Docker entrypoint).
- Role 3: Superset deck.gl H3 chart requires manual datasource + chart creation in the UI.
- Role 4: IndicTrans2 requires ~2 GB model download on first run.
- Role 4: All external services (Twilio, MQTT, PostGIS) default to DRY-RUN/mock mode.
- Role 4: `h3-pg` PostgreSQL extension is NOT installed; hex geometries are computed in Python.

---

## How to Run

### Infrastructure (Roles 3 & 4)
```bash
cd Cartaq
docker compose -f infra/docker-compose.yml up -d
# Wait ~60s, then:
#   ClickHouse: http://localhost:8123/play
#   Superset:   http://localhost:8088  (admin / admin)
```

### Role 3: Generate + Ingest fake data
```bash
pip install -r requirements.txt
python src/analytics/generate_fake_data.py        # writes data/fake_*_5m.parquet
python src/analytics/ingest_to_clickhouse.py      # loads → ClickHouse, writes metrics_role3.json
```

### Role 4: Run dispatch pipeline
```bash
# Initialize PostGIS schema (once):
psql "host=localhost port=5432 dbname=cartaq user=cartaq_user password=cartaq_pass" \
     -f src/dispatch/postgis_setup.sql

# Run end-to-end dispatch (dry-run by default):
python src/dispatch/dispatch_orchestrator.py
```

### Role 1: Training (dummy data)
```bash
cd Cartaq/src
python train.py     # uses random tensors; real pipeline needs graph_builder.py
```

### Role 1: Generate dummy contract
```bash
cd Cartaq/src
python create_dummy_contract.py    # writes output/dummy_forecast.parquet
```
