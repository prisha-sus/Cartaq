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

## Current State
- [x] Initialized Git repository on branch `feature/stgnn-aqi-forecasting`.
- [x] Zero-blocker contract published (dummy data schema available at `output/dummy_forecast.parquet`).
- [ ] Ingestion pipeline (OpenAQ + Open-Meteo) built.
- [ ] Spatial Graph Construction (H3 mapping + adjacency) built.
- [ ] Model (A3TGCN + Baseline) built.
- [ ] Trained and evaluated.

## Zero-Blocker Contract (Output Schema)
The output of this module is a parquet file located at `output/forecast.parquet` (currently using `output/dummy_forecast.parquet` as a placeholder).
The exact schema is:
- `h3_index` (string): H3 cell ID at resolution 8.
- `future_timestamp` (datetime): UTC, the forecasted point in time.
- `predicted_aqi` (float): model output, same AQI scale as ingested data.

## H3 Resolution
**Resolution 8** is chosen because it offers an appropriate balance between spatial granularity (~0.73 km² area, ~461m edge length) and graph sparsity for city-scale forecasting (e.g., Los Angeles).

## Key File Locations
- `src/ingestion/`: Scripts for data ingestion from OpenAQ and Open-Meteo.
- `src/graph/`: Scripts for H3 mapping and graph construction.
- `src/model/`: PyTorch Geometric Temporal model definitions (A3TGCN and Baseline).
- `output/dummy_forecast.parquet`: Dummy zero-blocker output data.
- `metrics.json`: Performance metrics (RMSE vs baseline, graph construction latency).

## Known Limitations
- Wind-direction edge weighting is NOT implemented.
- Pipeline focuses on a single city bounding box by default (Pune, Shivajinagar) for development.
- `torch_geometric_temporal` requires `torch-sparse`, which must be installed manually using the correct wheel for your PyTorch/CUDA version.

## How to Run
*(To be updated once pipeline scripts are finalized)*
1. Set up the virtual environment and install `requirements.txt`.
2. Run data ingestion: `python src/ingestion/...`
3. Run graph builder: `python src/graph/...`
4. Train model: `python src/train.py`
