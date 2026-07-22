# CARTAQ — Urban Air Quality Intelligence Platform

> **C**ausal **A**I-driven **R**eal-time **T**rajectory for **A**ir **Q**uality

CARTAQ is a full-stack air quality intelligence platform that predicts AQI at +24h/+48h/+72h horizons using a spatio-temporal graph neural network (A3TGCN), attributes pollution sources with DoWhy causal inference, and triggers automated multi-lingual dispatch alerts via MQTT/Twilio.

Built for the **ET Hackathon — Problem Statement 2**: Environment & Smart City Air Quality Management.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                            │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  OpenAQ API  │  │ Open-Meteo   │  │  NASA FIRMS  │  │    OSM     │  │
│  │  (PM2.5)     │  │ (Weather)    │  │  (Fires)     │  │ (Road/Ind) │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │
│         │                 │                 │                │         │
│         ▼                 ▼                 ▼                ▼         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Raw Data (data/raw/*.parquet)                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       GRAPH & FORECAST PIPELINE                         │
│                                                                         │
│  ┌────────────────┐    ┌──────────────────┐    ┌────────────────────┐  │
│  │ graph_builder  │───▶│  A3TGCN Model    │───▶│ forecast_generator │  │
│  │ H3 hex → adj   │    │  (PyTorch Geo)   │    │  +24h/+48h/+72h   │  │
│  │ matrix + feats │    │  train/infer     │    │  → forecast.parq  │  │
│  └────────────────┘    └──────────────────┘    └─────────┬──────────┘  │
│                                                          │             │
│              ┌───────────────────────────────────────────┘             │
│              ▼                                                         │
│  ┌────────────────────┐    ┌──────────────────────────────────────┐    │
│  │    enrich.py       │───▶│  output/live-enriched_data.parquet   │    │
│  │  join forecast +   │    │  (weather + causal treatment vars)   │    │
│  │  weather + causal  │    └──────────────────────────────────────┘    │
│  └────────────────────┘                                                │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CAUSAL & DISPATCH PIPELINE                         │
│                                                                         │
│  ┌────────────────┐    ┌──────────────────┐    ┌────────────────────┐  │
│  │   DoWhy Causal │    │  Dispatch Orch.  │    │  MQTT Publisher   │  │
│  │  (src/causal/) │───▶│  spatial query   │───▶│  (Mosquitto)      │  │
│  │  ATE per source│    │  → advisory gen  │    │  + Twilio SMS     │  │
│  └────────────────┘    │  → translation   │    └────────────────────┘  │
│                        └──────────────────┘                            │
│                                    │                                   │
│                                    ▼                                   │
│                        ┌────────────────────┐                          │
│                        │  Alert Table +     │                          │
│                        │  metrics_role4.json│                          │
│                        └────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND (server.py)                      │
│                                                                         │
│  /api/status  /api/forecast/{h}  /api/causal  /api/analytics           │
│  /api/dispatch  /api/metrics  /api/pipeline/run  /api/download/pdf     │
│                                                                         │
│  Serves static/ SPA frontend                                   port 8000│
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   SPA FRONTEND (static/index.html)                      │
│                                                                         │
│  ┌────────┐ ┌────────┐ ┌──────┐ ┌──────┐ ┌────────┐ ┌────────┐        │
│  │ Globe  │ │Overview│ │ Map  │ │Causal│ │Analyt. │ │Dispatch│        │
│  │ 3D     │ │KPI +   │ │Hex   │ │ATE   │ │Box +   │ │Alerts +│        │
│  │ Three.js│Pipeline │ │scatter│ │bar   │ │scatter │ │MQTT    │        │
│  └────────┘ └────────┘ └──────┘ └──────┘ └────────┘ └────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
OpenAQ ──┐
Open-Meteo┼──▶ graph_builder ──▶ A3TGCN ──▶ forecast ──┐
OSM ─────┘                                              │
                                                  enrich.py ──▶ API ──▶ SPA
NASA FIRMS ──▶ causal ──▶ dispatch ────────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| **3D Globe Dashboard** | Three.js interactive globe with real-time AQI pins, city search, AI health advisor |
| **AQI Forecasting** | A3TGCN spatio-temporal graph neural network — +24h/+48h/+72h predictions |
| **Hex Map** | H3 hexagonal grid over Pune — scattermapbox colored by AQI with colorbar |
| **Causal Attribution** | DoWhy analysis — ATE per pollution source (traffic, industry, construction, etc.) |
| **Analytics** | Box plots, stacked bars, wind/temp scatter — all interactive Plotly charts |
| **Dispatch Center** | Alert table with HAZARDOUS row pulsing, MQTT topic expanders, real metrics |
| **Multi-lingual Alerts** | IndicTrans2 translation pipeline (English → Marathi, Hindi, etc.) |
| **SMS Dispatch** | Twilio integration for automated alert calls |
| **MQTT Pub/Sub** | Mosquitto broker — compressed binary payloads for IoT edge nodes |
| **Docker Infra** | ClickHouse, PostGIS, Mosquitto, Superset — one `docker compose up` |

---

## Tech Stack

**Frontend:** HTML/CSS/JS SPA, Three.js (3D globe), Plotly.js (charts), Tailwind CSS, FontAwesome

**Backend:** FastAPI (Python), uvicorn

**ML/AI:** PyTorch, PyTorch Geometric Temporal (A3TGCN), DoWhy (causal inference), H3 (Uber hex grid)

**Infrastructure:** Docker Compose, ClickHouse (analytics), PostGIS (spatial queries), Mosquitto (MQTT), Apache Superset (visualization)

**External APIs:** OpenAQ (PM2.5 data), Open-Meteo (weather), NASA FIRMS (thermal anomalies), Overpass (OSM features)

---

## Prerequisites

- Python 3.10+
- pip
- Git

Optional (for full infrastructure):
- Docker + Docker Compose

---

## Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/prisha-sus/Cartaq.git
cd Cartaq
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys (optional — ingestion falls back to synthetic data without keys):

| Variable | Required | Source |
|---|---|---|
| `OPENAQ_API_KEY` | No | https://openaq.org |
| `NASA_FIRMS_API_KEY` | No | https://firms.modaps.eosdis.nasa.gov |
| `TWILIO_ACCOUNT_SID` | No | https://twilio.com |
| `TWILIO_AUTH_TOKEN` | No | Twilio console |
| `TWILIO_FROM_NUMBER` | No | Twilio console |

### 3. Run the Server (Demo Mode)

```bash
uvicorn server:app --reload --port 8000
```

Open **http://localhost:8000** — the dashboard runs immediately with **synthetic demo data** pre-generated.

---

## Running the Full Pipeline with Real Data

### Step 1 — Ingest Real Data

```bash
# PM2.5 from OpenAQ
python src/ingestion/openaq_client.py

# Weather from Open-Meteo (free, no key needed)
python src/ingestion/openmeteo_client.py

# Road/industrial zones from OSM (static, run once)
python src/ingestion/osm_client.py

# Thermal anomalies from NASA FIRMS (optional)
python src/ingestion/nasa_firms_client.py
```

### Step 2 — Build Graph & Generate Forecast

```bash
python src/pipeline/graph_builder.py
python src/pipeline/forecast_generator.py
python src/pipeline/enrich.py
```

### Step 3 — Run Dispatch

```bash
python src/dispatch/dispatch_orchestrator.py
```

### Step 4 — Restart Server

```bash
uvicorn server:app --reload --port 8000
```

All `/api/*` endpoints now serve the real regenerated data.

---

## API Endpoints

| Endpoint | Description | Returns |
|---|---|---|
| `GET /` | SPA frontend | `index.html` |
| `GET /api/status` | Pipeline health + KPIs | `{ pipeline, kpis, is_demo, timestamp }` |
| `GET /api/forecast/{horizon}` | H3 hex AQI forecast | `{ horizon, data: [{h3_index, aqi, lat, lon, category, polygon}], count }` |
| `GET /api/causal` | DoWhy causal attribution | `{ results: [{source, ate, ci_lower, ci_upper}], total_ate }` |
| `GET /api/analytics` | Cross-horizon analytics | `{ by_horizon, weather_correlation, aqi_scale }` |
| `GET /api/dispatch` | Alerts + MQTT metrics | `{ metrics, alerts, mqtt_topics }` |
| `GET /api/metrics` | Model/dispatch metrics | `{ role3, role4 }` |
| `POST /api/pipeline/run?step=demo` | Trigger pipeline step | `{ ok, step, output }` |
| `GET /api/download/pdf` | Causal report PDF | `causal_report.pdf` |

---

## Project Structure

```
Cartaq/
├── server.py                  # FastAPI backend — serves API + static SPA
├── app.py                     # Entry point (uvicorn wrapper)
├── requirements.txt           # Python dependencies
├── .env                       # API keys (gitignored)
├── .env.example               # Template for .env
├── readme.md                  # This file
│
├── static/                    # Frontend SPA
│   └── index.html             # Single-page application (all 6 tabs)
│
├── src/
│   ├── ingestion/             # Data ingestion clients
│   │   ├── openaq_client.py   #   OpenAQ PM2.5 fetcher
│   │   ├── openmeteo_client.py#   Open-Meteo weather fetcher
│   │   ├── osm_client.py      #   OpenStreetMap features
│   │   └── nasa_firms_client.py # NASA FIRMS thermal anomalies
│   │
│   ├── graph/                 # H3 mapping & adjacency
│   │   └── h3_mapper.py       #   H3 hex → feature tensor → k-ring graph
│   │
│   ├── model/                 # ML models
│   │   ├── a3tgcn.py          #   A3TGCN spatio-temporal model
│   │   └── physics_loss.py    #   Physics-informed loss constraints
│   │
│   ├── pipeline/              # Orchestration pipeline
│   │   ├── graph_builder.py   #   Build H3 graph tensors from raw data
│   │   ├── forecast_generator.py # Run A3TGCN inference → forecast.parquet
│   │   ├── enrich.py          #   Join forecast + weather + causal vars
│   │   ├── pdf_generator.py   #   Generate causal report PDF
│   │   └── run_pipeline.py    #   Orchestrate all pipeline steps
│   │
│   ├── dispatch/              # Alert dispatch system
│   │   ├── dispatch_orchestrator.py # Main dispatch coordinator
│   │   ├── spatial_query.py   #   PostGIS proximity queries
│   │   ├── translation_pipeline.py # IndicTrans2 multi-lingual translation
│   │   ├── twilio_dispatcher.py    # Twilio SMS dispatch
│   │   ├── mqtt_publisher.py  #   MQTT broker publisher
│   │   └── postgis_setup.sql  #   PostGIS schema
│   │
│   ├── analytics/             # ClickHouse analytics
│   │   ├── ingest_to_clickhouse.py # Load parquet → ClickHouse
│   │   └── queries/
│   │       └── benchmark_queries.sql # Performance benchmark queries
│   │
│   └── train.py               # A3TGCN training loop
│
├── infra/                     # Docker infrastructure
│   ├── docker-compose.yml     #   ClickHouse + PostGIS + Mosquitto + Superset
│   ├── clickhouse/init.sql    #   ClickHouse schema
│   ├── mosquitto/mosquitto.conf # MQTT broker config
│   └── superset/superset_config.py # Superset dashboard config
│
├── data/                      # Data files (gitignored via .gitignore)
│   ├── raw/                   #   Raw ingestion output
│   ├── graph/                 #   Graph tensors
│   └── forecast.parquet       #   AQI forecast output
│
└── output/                    # Pipeline outputs (gitignored)
    ├── live-enriched_data.parquet
    ├── a3tgcn_checkpoint.pt
    └── causal_report.pdf
```

---

## Demo vs Real Data

The project ships with **synthetic demo data** pre-generated so the UI is immediately usable:

| Data | Demo (default) | Real (after ingestion) |
|---|---|---|
| Forecast | Procedural spatial gradient + noise | OpenAQ PM2.5 + A3TGCN inference |
| Causal | DoWhy on synthetic vars | DoWhy on real enriched data |
| Weather | `numpy.random` diurnal pattern | Open-Meteo historical archive |
| Dispatch | Static JSON alerts | PostGIS + computed advisories |

To switch from demo → real, run the ingestion scripts and pipeline (see [Running the Full Pipeline](#running-the-full-pipeline-with-real-data)).

---

## Deployment

### Basic (single VM)

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Add a cron job for periodic data refresh:

```bash
# every hour: fetch new data + rebuild forecast + restart server
0 * * * * cd /opt/Cartaq && \
  python src/ingestion/openaq_client.py && \
  python src/ingestion/openmeteo_client.py && \
  python src/pipeline/run_pipeline.py
```

### Full (Docker)

```bash
docker compose -f infra/docker-compose.yml up -d
```

This starts:
- **ClickHouse** — columnar analytics database
- **PostGIS** — spatial database with H3 indexing
- **Mosquitto** — MQTT message broker
- **Superset** — dashboard & visualization UI

---

## License

MIT — built for the ET Hackathon.
