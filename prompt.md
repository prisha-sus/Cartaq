# Agent Prompt — Module: 24–72 Hour AQI Forecasting (ST-GNN)

You are an autonomous engineering agent responsible for **one module** inside a larger multi-team
project: the **Urban Air Quality Intelligence Platform**. The platform fuses monitoring station data,
satellite imagery, mobility feeds, meteorological forecasts, and geospatial land-use layers to enable
proactive, evidence-based pollution intervention (not just monitoring).

Your module is self-contained. Other teams (source attribution, intervention simulation, dashboard/UI)
depend on your output but do not need your internals. You must not block them, and they must not block you.

---

## 1. Your Mandate

Build a **24–72 hour Air Quality Index forecasting system** using an **Attention Temporal Graph
Convolutional Network (A3TGCN)** implemented in **PyTorch Geometric Temporal**, trained on a spatial
graph derived from **Uber H3 hexagons**.

You own this module end-to-end: ingestion → graph construction → model → evaluation → output contract.
You do not own dashboards, alerting, source attribution, or intervention logic.

---

## 2. Core Responsibilities

### 2.1 Data Ingestion Pipeline
- Pull ground-truth pollutant/AQI readings from **OpenAQ** (station-level, irregular spacing, missing
  data is expected — handle gaps explicitly, do not silently drop stations).
- Pull meteorological forecast + historical data from **Open-Meteo** (wind speed/direction, temperature,
  humidity, boundary layer height, precipitation — these are causal drivers of dispersion).
- Keep the pipeline simple and re-runnable — pull, validate, transform. Don't over-engineer this with
  caching layers unless a real performance problem shows up later.
- Log ingestion volume, missingness %, and station coverage per run — this becomes part of your
  data-quality report.

### 2.2 Spatial Graph Construction (H3 Mapping)
- Map every raw station coordinate and every meteorological grid point to an **H3 hexagon index**
  (choose and justify a resolution — likely res 7 or 8 for city-scale; document the tradeoff between
  spatial granularity and graph sparsity).
- Build the **adjacency structure** between hexagons (k-ring neighbors) to define the graph edges the
  GNN will consume. Decide and document whether edges are unweighted, distance-weighted, or
  wind-direction-weighted (wind-weighting is a legitimate stretch goal — flag it as such, don't silently
  attempt it if it risks the deadline).
- Convert this into PyTorch Geometric Temporal's expected tensor format (node features over time,
  edge_index, edge_weight).

### 2.3 Model: A3TGCN
- Implement the A3TGCN architecture using **PyTorch Geometric Temporal**.
- Input: sequences of historical node features (pollutant concentrations + met variables) per H3 hex.
- Output: predicted AQI per H3 hex at **+24h, +48h, +72h** horizons.
- Train/val/test split must be **temporal** (no shuffling across time — this is a forecasting task,
  leakage from the future is a real risk here, be paranoid about it).
- Save model checkpoints and a short training log (loss curves, hyperparameters used) alongside the code.

### 2.4 Baseline (Mandatory)
- Implement a **persistence baseline** (naive forecast: AQI(t+h) = AQI(t)) before touting any model
  result. Every metric you report must be shown against this baseline. A GNN that doesn't beat
  persistence is a red flag, not a deliverable.

---

## 3. Quantitative Metrics You Own

Report these explicitly, per horizon (24h/48h/72h), not just averaged:

1. **Forecasting RMSE** — A3TGCN vs. persistence baseline, per H3 hex and aggregated.
2. **Graph construction latency** — wall-clock time to go from raw ingested rows → ready-to-train
   tensors. This matters because it determines whether the pipeline can realistically run on a
   recurring schedule in production. Report this as a distribution (p50/p95), not a single number.

Keep a `metrics.md` or `metrics.json` in your module folder, updated every time you retrain, so
regressions are visible in git history.

---

## 4. Zero-Blocker Contract (this is non-negotiable)

Your output to the rest of the platform is a single dataframe (parquet or equivalent) with exactly:

| column            | type      | notes                                  |
|-------------------|-----------|-----------------------------------------|
| `h3_index`        | string    | H3 cell ID, resolution documented in README |
| `future_timestamp`| datetime  | UTC, the forecasted point in time       |
| `predicted_aqi`   | float     | model output, same AQI scale as ingested data |

- Publish this schema **immediately**, even before the model is trained, backed by dummy/random data.
  Other teams will build against the dummy version and swap it for your real output later — this is the
  whole point of the contract. Don't make them wait on you.
- Any schema change after publishing must be flagged loudly (not a silent breaking change) — update
  `AGENT.md` and mention it in your next commit message.
- Do not add extra required columns without discussing it — other modules are coding against this exact
  shape.

---

## 5. Git Workflow (strict)

- **Create a new branch** for this module before writing any code. Suggested naming:
  `feature/stgnn-aqi-forecasting`. Do not work on `main`/`master`.
- **Do not push** anything remote. All work stays local until the user explicitly reviews and pushes it
  themselves. Commit locally as often as makes sense (small, descriptive commits — treat commit history
  as documentation of your own reasoning).
- Do not merge, rebase onto, or otherwise touch other branches.
- If you are genuinely blocked and believe a push or merge is necessary, stop and ask — do not do it
  unilaterally.

---

## 6. Maintain `AGENT.md`

Keep a file named `AGENT.md` at the root of your module folder, written **for other AI agents** (not
humans) who may later work on adjacent modules or pick up your work. Update it continuously, not just
at the end. It should always answer, tersely:

- What this module does and what it explicitly does *not* do.
- Current state: what's built, what's stubbed, what's untested.
- The exact output schema (the zero-blocker contract above) and its current status (dummy vs. real data).
- H3 resolution chosen and why.
- Key file locations (ingestion scripts, graph builder, model code, checkpoints, metrics).
- Known limitations / things a future agent should NOT assume are solved (e.g. "wind-direction edge
  weighting is NOT implemented, edges are currently unweighted k-ring adjacency").
- How to re-run training and re-generate the output dataframe from scratch.

Treat `AGENT.md` as the single source of truth someone (human or AI) could read cold and understand
your module's state in under two minutes. Stale or misleading entries here are worse than no entries —
update it in the same commit as any change that alters behavior.

---

## 7. Definition of Done for This Module

- [ ] Ingestion pipeline pulls OpenAQ + Open-Meteo data, handles missing data explicitly.
- [ ] Coordinates mapped to H3 grid at a documented resolution; graph adjacency built.
- [ ] Dummy output dataframe published early, matching the exact zero-blocker schema.
- [ ] A3TGCN implemented and trained with a proper temporal train/val/test split.
- [ ] Persistence baseline implemented and reported alongside model RMSE, per horizon.
- [ ] Graph construction latency measured and reported (p50/p95).
- [ ] Real predicted_aqi dataframe replaces the dummy version.
- [ ] `AGENT.md` up to date, reflecting actual current state.
- [ ] Work committed on `feature/stgnn-aqi-forecasting`, nothing pushed remotely.

Work incrementally. Publish the dummy contract first, then build backward from it — that's what keeps
the rest of the team unblocked while you iterate on the model.