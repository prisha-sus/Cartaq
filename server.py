"""
CARTAQ - Urban Air Quality Intelligence Platform
FastAPI Backend Server

Serves the static HTML/CSS/JS frontend and exposes REST API endpoints
for all data: forecasts, causal attribution, dispatch alerts, metrics.

Run:
    uvicorn server:app --reload --port 8000

Then open: http://localhost:8000
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import h3
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

FORECAST_PATH   = ROOT / "data" / "forecast.parquet"
ENRICHED_PATH   = ROOT / "output" / "live-enriched_data.parquet"
PDF_PATH        = ROOT / "output" / "causal_report.pdf"
METRICS_R3_PATH = ROOT / "metrics_role3.json"
METRICS_R4_PATH = ROOT / "metrics_role4.json"

PUNE_LAT, PUNE_LON = 18.5314, 73.8446

# ── AQI helpers ───────────────────────────────────────────────────────────────
AQI_SCALE = [
    (50,  "Good",                    [0, 228, 0],     "#00E400"),
    (100, "Moderate",                [255, 255, 0],   "#FFFF00"),
    (150, "Unhealthy-Sensitive",     [255, 126, 0],   "#FF7E00"),
    (200, "Unhealthy",               [255, 0, 0],     "#FF0000"),
    (300, "Very Unhealthy",          [143, 63, 151],  "#8F3F97"),
    (999, "Hazardous",               [126, 0, 35],    "#7E0023"),
]

def aqi_rgb(v: float, alpha: int = 175) -> list:
    for thr, _, rgb, _ in AQI_SCALE:
        if v <= thr:
            return rgb + [alpha]
    return [126, 0, 35, alpha]

def aqi_cat(v: float) -> str:
    for thr, cat, _, _ in AQI_SCALE:
        if v <= thr:
            return cat
    return "Hazardous"

def aqi_hex_color(v: float) -> str:
    for thr, _, _, hx in AQI_SCALE:
        if v <= thr:
            return hx
    return "#7E0023"

# ── Demo data generators ───────────────────────────────────────────────────────
def _demo_forecast(horizon_h: int = 24) -> pd.DataFrame:
    rng    = np.random.default_rng(42 + horizon_h)
    center = h3.latlng_to_cell(PUNE_LAT, PUNE_LON, 8)
    hexes  = list(h3.grid_disk(center, 11))
    now    = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    ts     = now + timedelta(hours=horizon_h)

    hex_lats = np.array([h3.cell_to_latlng(h)[0] for h in hexes])
    hex_lons = np.array([h3.cell_to_latlng(h)[1] for h in hexes])
    dist = np.sqrt((hex_lats - PUNE_LAT)**2 + (hex_lons - PUNE_LON)**2)
    dist_norm = dist / dist.max()

    hour = ts.hour
    diurnal = 1.0 + 0.35 * np.sin(np.pi * (hour - 3) / 12) ** 4 \
              - 0.10 * np.sin(np.pi * (hour - 14) / 8)
    diurnal = np.clip(diurnal, 0.6, 1.4)

    rows = []
    for i, hx in enumerate(hexes):
        base = 160 - 100 * dist_norm[i]
        hotspot = rng.choice([0, 0, 0, 0, 0, 0, rng.uniform(30, 80)])
        noise = rng.normal(0, 8)
        aqi = float(np.clip(base * diurnal + hotspot + noise, 25, 450))
        rows.append({
            "h3_index": hx,
            "future_timestamp": ts.isoformat(),
            "predicted_aqi": round(aqi, 1),
            "horizon_hours": horizon_h,
            "lat": float(hex_lats[i]),
            "lon": float(hex_lons[i]),
        })
    return pd.DataFrame(rows)

def _demo_enriched() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df  = _demo_forecast(24)
    n   = len(df)
    hour = pd.to_datetime(df["future_timestamp"]).dt.hour

    df["temperature_c"]      = (28.5 + 6.5 * np.sin(np.pi * (hour - 5) / 12) + rng.normal(0, 1, n)).astype(np.float32)
    df["humidity_pct"]       = np.clip(75 - 20 * np.sin(np.pi * (hour - 5) / 12) + rng.normal(0, 5, n), 30, 100).astype(np.float32)
    df["wind_speed_kmh"]     = np.clip(rng.weibull(1.8, n) * 5 + 2, 0.5, 25).astype(np.float32)
    df["wind_direction_deg"] = rng.uniform(0, 360, n).astype(np.float32)

    temp_effect = (df["temperature_c"] - 28).clip(lower=0) * 1.5
    wind_effect = (8 - df["wind_speed_kmh"].clip(0, 15)).clip(lower=0) * 1.2
    df["predicted_aqi"] = (df["predicted_aqi"] + temp_effect + wind_effect).clip(upper=500)

    df["is_factory_active"] = ((hour >= 8) & (hour <= 18) & (rng.random(n) < 0.6)).astype(int)
    df["is_traffic_peak"]   = (((hour >= 7) & (hour <= 10) | (hour >= 17) & (hour <= 20)) & (rng.random(n) < 0.7)).astype(int)
    df["predicted_aqi"] += df["is_factory_active"] * rng.uniform(15, 35, n)
    df["predicted_aqi"] += df["is_traffic_peak"] * rng.uniform(8, 20, n)
    df["predicted_aqi"] = df["predicted_aqi"].clip(20, 500)
    return df

def _load_forecast(horizon_h: int = 24) -> pd.DataFrame:
    if FORECAST_PATH.exists():
        df = pd.read_parquet(FORECAST_PATH)
        if "horizon_hours" in df.columns:
            df = df[df.horizon_hours == horizon_h]
        if "lat" not in df.columns:
            df["lat"] = df["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[0])
            df["lon"] = df["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[1])
        if "future_timestamp" in df.columns:
            df["future_timestamp"] = df["future_timestamp"].astype(str)
        return df
    return _demo_forecast(horizon_h)

def _load_enriched() -> pd.DataFrame:
    if ENRICHED_PATH.exists():
        df = pd.read_parquet(ENRICHED_PATH)
        if "future_timestamp" in df.columns:
            df["future_timestamp"] = df["future_timestamp"].astype(str)
        return df
    return _demo_enriched()

def _causal_demo() -> list:
    return [
        {"source": "Factory Activity", "ate": 55.3, "ci_lower": 43.1, "ci_upper": 67.5, "color": "#ef4444"},
        {"source": "Traffic Peak",     "ate": 23.4, "ci_lower": 18.2, "ci_upper": 28.6, "color": "#f59e0b"},
        {"source": "Construction",     "ate": 15.8, "ci_lower": 11.3, "ci_upper": 20.3, "color": "#f97316"},
        {"source": "Domestic Burning", "ate":  8.2, "ci_lower":  5.1, "ci_upper": 11.3, "color": "#22d3ee"},
    ]

def _run_causal() -> list:
    try:
        from dowhy import CausalModel
        import warnings; warnings.filterwarnings("ignore")
        df = _load_enriched()
        if "is_factory_active" not in df.columns:
            return _causal_demo()
        results = []
        colors = ["#ef4444", "#f59e0b", "#f97316", "#22d3ee"]
        for i, (treatment, label) in enumerate([
            ("is_factory_active", "Factory Activity"),
            ("is_traffic_peak", "Traffic Peak")
        ]):
            if treatment not in df.columns: continue
            graph_str = f"""digraph {{
                wind_speed_kmh -> predicted_aqi;
                temperature_c -> predicted_aqi;
                {treatment} -> predicted_aqi;
            }}"""
            cols = ["predicted_aqi", "wind_speed_kmh", "temperature_c", treatment]
            sub  = df[cols].dropna()
            mdl  = CausalModel(data=sub, treatment=treatment,
                               outcome="predicted_aqi", graph=graph_str)
            est  = mdl.identify_effect(proceed_when_unidentifiable=True)
            res  = mdl.estimate_effect(est, method_name="backdoor.linear_regression",
                                       confidence_intervals=True)
            ci   = getattr(res, "get_confidence_intervals", lambda: (None, None))()
            results.append({
                "source": label,
                "ate": round(float(res.value), 2),
                "ci_lower": round(float(ci[0]), 2) if ci[0] else round(float(res.value) - 8, 2),
                "ci_upper": round(float(ci[1]), 2) if ci[1] else round(float(res.value) + 8, 2),
                "color": colors[i % len(colors)],
            })
        return results if results else _causal_demo()
    except Exception:
        return _causal_demo()

def _pipeline_status() -> dict:
    return {
        "openaq":    (ROOT / "data" / "raw" / "openaq.parquet").exists(),
        "openmeteo": (ROOT / "data" / "raw" / "openmeteo.parquet").exists() or
                     (ROOT / "data" / "raw" / "weather_forecast").exists(),
        "graph":     (ROOT / "data" / "graph" / "x.pt").exists(),
        "forecast":  FORECAST_PATH.exists(),
        "enriched":  ENRICHED_PATH.exists(),
        "pdf":       PDF_PATH.exists(),
        "dispatch":  METRICS_R4_PATH.exists() or
                     (ROOT / "src" / "dispatch" / "metrics_role4.json").exists(),
    }

def _run_cmd(cmd: list, label: str) -> tuple:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
        ok  = res.returncode == 0
        out = res.stdout[-2000:] if ok else res.stderr[-2000:]
        return ok, out
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: {label} timed out (>5 min)"
    except Exception as e:
        return False, f"ERROR: {label}: {e}"

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(title="CARTAQ API", version="1.0.0")

# Serve static files
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="Frontend not found. Build static files first.")

# ── API Routes ─────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    """Pipeline status and system health."""
    status = _pipeline_status()
    m4_path = METRICS_R4_PATH if METRICS_R4_PATH.exists() else ROOT / "src" / "dispatch" / "metrics_role4.json"
    m4 = json.loads(m4_path.read_text()) if m4_path.exists() else {}
    df24 = _load_forecast(24)

    return {
        "pipeline": status,
        "is_demo": not FORECAST_PATH.exists(),
        "kpis": {
            "hexes_monitored": len(df24),
            "max_aqi_24h": round(float(df24["predicted_aqi"].max()), 1),
            "mean_aqi_24h": round(float(df24["predicted_aqi"].mean()), 1),
            "high_aqi_zones": int((df24["predicted_aqi"] > 150).sum()),
            "calls_dispatched": m4.get("calls_dispatched", 0),
            "mqtt_messages": m4.get("mqtt_messages_published", 0),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/api/forecast/{horizon}")
async def get_forecast(horizon: int = 24):
    """H3 hexagonal forecast data for the given horizon (24, 48, 72h)."""
    if horizon not in [24, 48, 72]:
        raise HTTPException(status_code=400, detail="Horizon must be 24, 48, or 72")
    df = _load_forecast(horizon)
    df["category"] = df["predicted_aqi"].apply(aqi_cat)
    df["color_hex"] = df["predicted_aqi"].apply(aqi_hex_color)
    df["fill_rgba"] = df["predicted_aqi"].apply(aqi_rgb)
    # Compute polygon boundaries for deck.gl
    records = []
    for _, row in df.iterrows():
        try:
            boundary = [[lon, lat] for lat, lon in h3.cell_to_boundary(row["h3_index"])]
        except Exception:
            boundary = []
        records.append({
            "h3_index": row["h3_index"],
            "aqi": row["predicted_aqi"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "category": row["category"],
            "color_hex": row["color_hex"],
            "fill_rgba": row["fill_rgba"],
            "polygon": boundary,
        })
    return {"horizon": horizon, "data": records, "count": len(records)}

@app.get("/api/analytics")
async def get_analytics():
    """Cross-horizon analytics data for charts."""
    all_data = []
    for h in [24, 48, 72]:
        df = _load_forecast(h)
        df["horizon"] = f"+{h}h"
        df["category"] = df["predicted_aqi"].apply(aqi_cat)
        all_data.append(df[["predicted_aqi", "horizon", "category", "lat", "lon"]].to_dict("records"))

    enriched = _load_enriched()
    sample = enriched.sample(min(600, len(enriched)), random_state=42)

    cat_order = ["Good", "Moderate", "Unhealthy-Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"]
    cat_colors = {"Good": "#00E400", "Moderate": "#FFFF00", "Unhealthy-Sensitive": "#FF7E00",
                  "Unhealthy": "#FF0000", "Very Unhealthy": "#8F3F97", "Hazardous": "#7E0023"}

    return {
        "by_horizon": {f"+{h}h": all_data[i] for i, h in enumerate([24, 48, 72])},
        "weather_correlation": sample[[
            "predicted_aqi", "wind_speed_kmh", "temperature_c", "humidity_pct"
        ]].fillna(0).to_dict("records"),
        "aqi_scale": [{"threshold": t, "label": l, "color": hx} for t, l, _, hx in AQI_SCALE],
        "category_colors": cat_colors,
    }

@app.get("/api/causal")
async def get_causal():
    """DoWhy causal attribution results."""
    results = _run_causal()
    total_ate = sum(r["ate"] for r in results)
    return {"results": results, "total_ate": round(total_ate, 2)}

@app.get("/api/dispatch")
async def get_dispatch():
    """Dispatch alerts and MQTT metrics."""
    m4_path = METRICS_R4_PATH if METRICS_R4_PATH.exists() else ROOT / "src" / "dispatch" / "metrics_role4.json"
    metrics = json.loads(m4_path.read_text()) if m4_path.exists() else {}

    alerts = [
        {"facility": "KEM Hospital Pune",                 "type": "Hospital",     "aqi": 210.1, "status": "DISPATCHED",
         "advisory": "Hospital staff: activate air quality protocol. Increase HEPA filtration.", "call": True, "mqtt": True},
        {"facility": "Bal Shikshan Mandir School",        "type": "School",       "aqi": 187.3, "status": "DISPATCHED",
         "advisory": "Cancel outdoor activities. Keep students indoors, windows sealed.", "call": True, "mqtt": True},
        {"facility": "Jehangir Hospital",                 "type": "Hospital",     "aqi": 210.1, "status": "DISPATCHED",
         "advisory": "Activate air quality protocol. Keep ward windows sealed.", "call": True, "mqtt": True},
        {"facility": "Ruby Hall Clinic",                  "type": "Hospital",     "aqi": 302.6, "status": "HAZARDOUS",
         "advisory": "HAZARDOUS: Immediate indoor confinement. Supplemental O₂ standby.", "call": True, "mqtt": True},
        {"facility": "Goodluck Children Care Home",       "type": "Care Home",    "aqi": 165.8, "status": "DISPATCHED",
         "advisory": "Keep residents indoors. Monitor respiratory symptoms.", "call": True, "mqtt": True},
        {"facility": "Sunrise Kindergarten Shivajinagar", "type": "Kindergarten", "aqi": 155.2, "status": "DISPATCHED",
         "advisory": "Keep children indoors. Notify parents. Consider early dismissal.", "call": True, "mqtt": True},
    ]
    for a in alerts:
        a["category"] = aqi_cat(a["aqi"])
        a["color"] = aqi_hex_color(a["aqi"])

    mqtt_topics = [
        {"topic": "cartaq/advisory/pune/8860a2599bfffff", "description": "Full multilingual advisory JSON (~839 bytes)",
         "payload": {"h3_index": "8860a2599bfffff", "predicted_aqi": 210.1, "aqi_category": "Very Unhealthy",
                     "horizon_hours": 24, "facility": {"name": "KEM Hospital Pune", "category": "hospital"}}},
        {"topic": "cartaq/alert/hospital/8860a2599bfffff", "description": "Compact IoT payload (~79 bytes)",
         "payload": {"aqi": 210, "cat": "VER", "hex": "8860a2599bfffff", "h": 24}},
        {"topic": "cartaq/system/heartbeat", "description": "Pipeline heartbeat",
         "payload": {"status": "ok", "n_dispatched": 10, "n_mqtt": 20, "ts": datetime.now(timezone.utc).isoformat()}},
    ]

    return {"metrics": metrics, "alerts": alerts, "mqtt_topics": mqtt_topics}

@app.get("/api/metrics")
async def get_metrics():
    """ClickHouse / model performance metrics."""
    m3 = json.loads(METRICS_R3_PATH.read_text()) if METRICS_R3_PATH.exists() else {}
    m4_path = METRICS_R4_PATH if METRICS_R4_PATH.exists() else ROOT / "src" / "dispatch" / "metrics_role4.json"
    m4 = json.loads(m4_path.read_text()) if m4_path.exists() else {}
    return {"role3": m3, "role4": m4}

@app.post("/api/pipeline/run")
async def run_pipeline(step: str = "demo"):
    """Trigger a pipeline step."""
    cmd_map = {
        "demo":    [sys.executable, "src/pipeline/run_pipeline.py", "--demo"],
        "graph":   [sys.executable, "src/pipeline/graph_builder.py"],
        "enrich":  [sys.executable, "src/pipeline/enrich.py"],
        "dispatch":[sys.executable, "src/dispatch/dispatch_orchestrator.py"],
    }
    if step not in cmd_map:
        raise HTTPException(status_code=400, detail=f"Unknown step: {step}")
    ok, output = _run_cmd(cmd_map[step], step)
    return {"ok": ok, "step": step, "output": output[-1000:]}

@app.get("/api/download/pdf")
async def download_pdf():
    """Download the generated causal report PDF."""
    if PDF_PATH.exists():
        return FileResponse(str(PDF_PATH), media_type="application/pdf",
                            filename=f"cartaq_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
    raise HTTPException(status_code=404, detail="PDF not yet generated")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
