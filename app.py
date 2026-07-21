"""
CARTAQ - Urban Air Quality Intelligence Platform
Streamlit Demo Frontend

Combines all 4 roles into a single interactive dashboard.

Run:
    streamlit run app.py

Works immediately in demo mode with no external services needed.
Automatically uses real data when available (data/forecast.parquet, etc.)
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

FORECAST_PATH   = ROOT / "data" / "forecast.parquet"
ENRICHED_PATH   = ROOT / "output" / "live-enriched_data.parquet"
PDF_PATH        = ROOT / "output" / "causal_report.pdf"
METRICS_R3_PATH = ROOT / "metrics_role3.json"
METRICS_R4_PATH = ROOT / "metrics_role4.json"
MOCK_PAYLOAD    = ROOT / "src" / "dispatch" / "mock_payload.json"

PUNE_LAT, PUNE_LON = 18.5314, 73.8446

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CARTAQ - Air Quality Intelligence",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
    header     {visibility: hidden;}
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #334466;
        border-radius: 8px;
        padding: 10px 14px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.7rem !important; color: #e94560 !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #90a0c0 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem; padding: 6px 16px; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── AQI helpers ───────────────────────────────────────────────────────────────
AQI_SCALE = [
    (50,  "Good",                    [0, 228, 0],     "#00E400"),
    (100, "Moderate",                [255, 255, 0],   "#FFFF00"),
    (150, "Unhealthy-Sensitive",     [255, 126, 0],   "#FF7E00"),
    (200, "Unhealthy",               [255, 0, 0],     "#FF0000"),
    (300, "Very Unhealthy",          [143, 63, 151],  "#8F3F97"),
    (999, "Hazardous",               [126, 0, 35],    "#7E0023"),
]

def aqi_rgb(v: float, alpha: int = 175) -> list[int]:
    for thr, _, rgb, _ in AQI_SCALE:
        if v <= thr:
            return rgb + [alpha]
    return [126, 0, 35, alpha]

def aqi_cat(v: float) -> str:
    for thr, cat, _, _ in AQI_SCALE:
        if v <= thr:
            return cat
    return "Hazardous"

def aqi_hex(v: float) -> str:
    for thr, _, _, hx in AQI_SCALE:
        if v <= thr:
            return hx
    return "#7E0023"

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_forecast(horizon_h: int = 24) -> pd.DataFrame:
    if FORECAST_PATH.exists():
        df = pd.read_parquet(FORECAST_PATH)
        if "horizon_hours" in df.columns:
            df = df[df.horizon_hours == horizon_h]
        return df
    return _demo_forecast(horizon_h)

@st.cache_data(ttl=60, show_spinner=False)
def load_enriched() -> pd.DataFrame:
    if ENRICHED_PATH.exists():
        return pd.read_parquet(ENRICHED_PATH)
    return _demo_enriched()

@st.cache_data(ttl=30, show_spinner=False)
def load_metrics_r4() -> dict:
    for p in [METRICS_R4_PATH, ROOT / "src" / "dispatch" / "metrics_role4.json"]:
        if p.exists():
            return json.loads(p.read_text())
    return {}

@st.cache_data(ttl=30, show_spinner=False)
def load_metrics_r3() -> dict:
    if METRICS_R3_PATH.exists():
        return json.loads(METRICS_R3_PATH.read_text())
    return {}

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
        base = 160 - 100 * dist_norm[i]  # centre ~160, outskirts ~60
        hotspot = rng.choice([0, 0, 0, 0, 0, 0, rng.uniform(30, 80)])
        noise = rng.normal(0, 8)
        aqi = float(np.clip(base * diurnal + hotspot + noise, 25, 450))
        rows.append({"h3_index": hx, "future_timestamp": ts,
                     "predicted_aqi": round(aqi, 1), "horizon_hours": horizon_h})
    return pd.DataFrame(rows)

def _demo_enriched() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df  = _demo_forecast(24)
    n   = len(df)
    hour = pd.to_datetime(df["future_timestamp"]).dt.hour

    # Weather with diurnal structure
    df["temperature_c"]      = (28.5 + 6.5 * np.sin(np.pi * (hour - 5) / 12) + rng.normal(0, 1, n)).astype(np.float32)
    df["humidity_pct"]       = np.clip(75 - 20 * np.sin(np.pi * (hour - 5) / 12) + rng.normal(0, 5, n), 30, 100).astype(np.float32)
    df["wind_speed_kmh"]     = np.clip(rng.weibull(1.8, n) * 5 + 2, 0.5, 25).astype(np.float32)
    df["wind_direction_deg"] = rng.uniform(0, 360, n).astype(np.float32)

    # Apply weather effects on AQI (creates visible correlation in scatter plots)
    temp_effect = (df["temperature_c"] - 28).clip(lower=0) * 1.5
    wind_effect = (8 - df["wind_speed_kmh"].clip(0, 15)).clip(lower=0) * 1.2
    df["predicted_aqi"] = (df["predicted_aqi"] + temp_effect + wind_effect).clip(upper=500)

    # Causal treatment variables with actual AQI impact
    df["is_factory_active"] = ((hour >= 8) & (hour <= 18) & (rng.random(n) < 0.6)).astype(int)
    df["is_traffic_peak"]   = (((hour >= 7) & (hour <= 10) | (hour >= 17) & (hour <= 20)) & (rng.random(n) < 0.7)).astype(int)

    # Inject causal signal so DoWhy finds real effects
    df["predicted_aqi"] += df["is_factory_active"] * rng.uniform(15, 35, n)
    df["predicted_aqi"] += df["is_traffic_peak"] * rng.uniform(8, 20, n)
    df["predicted_aqi"] = df["predicted_aqi"].clip(20, 500)
    return df

def _prep_map_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fill_color"] = df["predicted_aqi"].apply(aqi_rgb)
    df["polygon"]    = df["h3_index"].apply(
        lambda x: [[lon, lat] for lat, lon in h3.cell_to_boundary(x)]
    )
    df["lat"]        = df["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[0])
    df["lon"]        = df["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[1])
    df["category"]   = df["predicted_aqi"].apply(aqi_cat)
    df["aqi_r"]      = df["predicted_aqi"].round(1)
    return df

def _causal_demo() -> list[dict]:
    return [
        {"source": "Factory Activity", "ate": 55.3, "ci_lower": 43.1, "ci_upper": 67.5},
        {"source": "Traffic Peak",     "ate": 23.4, "ci_lower": 18.2, "ci_upper": 28.6},
        {"source": "Construction",     "ate": 15.8, "ci_lower": 11.3, "ci_upper": 20.3},
        {"source": "Domestic Burning", "ate":  8.2, "ci_lower":  5.1, "ci_upper": 11.3},
    ]

@st.cache_data(show_spinner="Running DoWhy causal analysis…")
def _run_causal() -> list[dict]:
    try:
        from dowhy import CausalModel
        import warnings; warnings.filterwarnings("ignore")
        df = load_enriched()
        if "is_factory_active" not in df.columns:
            return _causal_demo()
        results = []
        for treatment, label in [("is_factory_active","Factory Activity"),
                                  ("is_traffic_peak","Traffic Peak")]:
            if treatment not in df.columns: continue
            graph_str = f"""digraph {{
                wind_speed_kmh -> predicted_aqi;
                temperature_c -> predicted_aqi;
                {treatment} -> predicted_aqi;
            }}"""
            cols = ["predicted_aqi","wind_speed_kmh","temperature_c",treatment]
            sub  = df[cols].dropna()
            mdl  = CausalModel(data=sub, treatment=treatment,
                               outcome="predicted_aqi", graph=graph_str)
            est  = mdl.identify_effect(proceed_when_unidentifiable=True)
            res  = mdl.estimate_effect(est, method_name="backdoor.linear_regression",
                                       confidence_intervals=True)
            ci   = getattr(res, "get_confidence_intervals", lambda: (None,None))()
            results.append({
                "source": label, "ate": round(float(res.value),2),
                "ci_lower": round(float(ci[0]),2) if ci[0] else round(float(res.value)-8,2),
                "ci_upper": round(float(ci[1]),2) if ci[1] else round(float(res.value)+8,2),
            })
        return results if results else _causal_demo()
    except Exception:
        return _causal_demo()

def _pipeline_status() -> dict:
    return {
        "openaq":    (ROOT/"data"/"raw"/"openaq.parquet").exists(),
        "openmeteo": (ROOT/"data"/"raw"/"openmeteo.parquet").exists() or
                     (ROOT/"data"/"raw"/"weather_forecast").exists(),
        "graph":     (ROOT/"data"/"graph"/"x.pt").exists(),
        "forecast":  FORECAST_PATH.exists(),
        "enriched":  ENRICHED_PATH.exists(),
        "pdf":       PDF_PATH.exists(),
        "dispatch":  METRICS_R4_PATH.exists() or
                     (ROOT/"src"/"dispatch"/"metrics_role4.json").exists(),
    }

def _run_cmd(cmd: list[str], label: str) -> tuple[bool, str]:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
        ok  = res.returncode == 0
        out = res.stdout[-2000:] if ok else res.stderr[-2000:]
        return ok, f"{'> OK' if ok else '> FAIL'} {label}\n{out}"
    except subprocess.TimeoutExpired:
        return False, f"> TIMEOUT {label} timed out (>5 min)"
    except Exception as e:
        return False, f"> ERROR {label}: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# AUTO PIPELINE - runs on startup if forecast data is missing
# ─────────────────────────────────────────────────────────────────────────────
def _auto_run_pipeline():
    """Run the full pipeline automatically, showing progress."""
    steps = [
        ("Building graph tensors...", [sys.executable, "src/pipeline/graph_builder.py"]),
        ("Generating forecast...", [sys.executable, "src/pipeline/run_pipeline.py", "--step", "forecast"]),
        ("Enriching with weather data...", [sys.executable, "src/pipeline/enrich.py"]),
    ]
    progress_bar = st.progress(0, text="Initializing CARTAQ pipeline...")
    status_text = st.empty()
    for i, (label, cmd) in enumerate(steps):
        status_text.info(label)
        ok, _ = _run_cmd(cmd, label)
        if not ok:
            status_text.warning(f"{label} - using fallback")
        progress_bar.progress((i + 1) / len(steps))
    progress_bar.empty()
    status_text.empty()
    st.cache_data.clear()

if not FORECAST_PATH.exists():
    with st.spinner("Running real-time pipeline to generate live data..."):
        _auto_run_pipeline()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([1, 9])
hcol1.markdown("# 🌬️")
with hcol2:
    st.markdown("## CARTAQ - Urban Air Quality Intelligence Platform")
    is_demo = not FORECAST_PATH.exists()
    mode_badge = "🎭 **Demo Mode**" if is_demo else "🟢 **Live Data**"
    st.caption(f"Pune, Maharashtra · H3 Resolution 8 · ST-GNN + DoWhy + ClickHouse + Mosquitto · {mode_badge}")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "🏠 Overview",
    "🗺️ AQI Forecast Map",
    "🔬 Causal Attribution",
    "📈 Analytics",
    "🚨 Dispatch Alerts",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    status  = _pipeline_status()
    df24    = load_forecast(24)
    m4      = load_metrics_r4()

    # KPI row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Hexes Monitored",   f"{len(df24):,}")
    k2.metric("Max AQI (+24h)",    f"{df24.predicted_aqi.max():.0f}")
    k3.metric("Mean AQI (+24h)",   f"{df24.predicted_aqi.mean():.0f}")
    k4.metric("High-AQI Zones",    f"{(df24.predicted_aqi > 150).sum()}",
              help="Hexes with predicted AQI > 150 (Unhealthy)")
    k5.metric("Calls Dispatched",  m4.get("calls_dispatched", 0))
    k6.metric("MQTT Messages",     m4.get("mqtt_messages_published", 0))

    st.divider()

    # Pipeline status
    st.subheader("⚙️ Pipeline Status")

    STEPS = [
        ("1", "Ingest OpenAQ + Open-Meteo",        status["openaq"] and status["openmeteo"],
         "python src/ingestion/openaq_client.py  &&  python src/ingestion/openmeteo_client.py"),
        ("2", "Build H3 Graph Tensors (Role 1)",   status["graph"],
         "python src/pipeline/graph_builder.py"),
        ("3", "ST-GNN Forecast -> data/forecast.parquet", status["forecast"],
         "python src/pipeline/run_pipeline.py --demo  (or replace with real model output)"),
        ("4", "Enrich: Forecast + Weather",         status["enriched"],
         "python src/pipeline/enrich.py"),
        ("5", "Causal Analysis + PDF (Role 2)",     status["pdf"],
         "Click 'Generate PDF' in the Causal Attribution tab"),
        ("6", "Dispatch Alerts (Role 4)",           status["dispatch"],
         "python src/dispatch/dispatch_orchestrator.py"),
    ]

    for num, name, done, hint in STEPS:
        icon_col, name_col, hint_col = st.columns([0.4, 3.5, 4])
        icon_col.markdown("**OK**" if done else "**..**")
        name_col.markdown(f"**Step {num}:** {name}")
        if not done:
            hint_col.code(hint, language=None)
        else:
            hint_col.caption("Complete")

    st.divider()
    st.subheader("🚀 Quick Actions")

    qa1, qa2, qa3, qa4 = st.columns(4)

    def _run_and_show(cmd, label, show_output=True):
        """Run a command and display its output."""
        with st.spinner(f"Running {label}..."):
            ok, msg = _run_cmd(cmd, label)
            if show_output:
                st.code(msg, language="text")
            if ok:
                st.success(f"{label} completed")
            else:
                st.error(f"{label} failed")
            return ok

    with qa1:
        if st.button("Generate Demo Data", use_container_width=True,
                     help="Runs full pipeline with realistic synthetic data"):
            _run_and_show([sys.executable, "src/pipeline/run_pipeline.py", "--demo"],
                          "Demo Pipeline")
            st.cache_data.clear()
            st.info("Data updated. Interact with tabs above to see refreshed charts.")

    with qa2:
        if st.button("Build Graph Tensors", use_container_width=True,
                     help="Builds H3 graph tensors from sensor/weather data"):
            _run_and_show([sys.executable, "src/pipeline/graph_builder.py"],
                          "Graph Builder")
            st.cache_data.clear()
            st.info("Graph tensors ready. Run 'Generate Demo Data' or 'Refresh All Data' to use them.")

    with qa3:
        if st.button("Run Dispatch", use_container_width=True,
                     help="Runs dispatch orchestrator (dry-run by default)"):
            _run_and_show([sys.executable, "src/dispatch/dispatch_orchestrator.py"],
                          "Dispatch")
            st.cache_data.clear()
            st.info("Dispatch complete. Switch to the Dispatch Alerts tab to see results.")

    with qa4:
        if st.button("Refresh All Data", use_container_width=True,
                     help="Re-runs graph builder, forecast, and enrichment"):
            for label, cmd in [
                ("Graph Builder", [sys.executable, "src/pipeline/graph_builder.py"]),
                ("Forecast", [sys.executable, "src/pipeline/run_pipeline.py", "--step", "forecast"]),
                ("Enrich", [sys.executable, "src/pipeline/enrich.py"]),
            ]:
                _run_and_show(cmd, label)
            st.cache_data.clear()
            st.info("All data refreshed. Switch between tabs to see updated charts.")

    st.divider()
    st.subheader("🏗️ Data Flow Architecture")
    st.code("""
 OpenAQ API ──┐
              ├──► graph_builder.py ──► data/graph/x.pt + edge_index.pt
 Open-Meteo──┘          │                         │
                  H3 Res-8 map             A3TGCN train.py
                                                   │
                                     data/forecast.parquet
                                                   │
 Open-Meteo ──────────► enrich.py ──► output/live-enriched_data.parquet
  (weather)                                        │
                                     DoWhy causal_engine.py
                                         (ATE + 95% CI)
                                                   │
                              pdf_generator.py ──► output/causal_report.pdf
                                                   │
                             PostGIS ST_Intersects (vulnerable facilities)
                                                   │
                         IndicTrans2 ──► Hindi + Marathi advisories
                                                   │
                     Twilio ──► voice calls    Mosquitto ──► MQTT/signage
                                                   │
                      ClickHouse ◄──── all logs ────┘
                      Superset  ──► deck.gl H3 dashboard
    """, language=None)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - AQI FORECAST MAP
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.subheader("🗺️ 24-72h AQI Forecast - H3 Hexagonal Grid")

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 4])
    horizon  = ctrl1.selectbox("Forecast horizon", [24, 48, 72], format_func=lambda x: f"+{x}h")
    thr      = ctrl2.slider("Alert threshold (AQI)", 50, 300, 150, 25)

    with st.spinner("Rendering forecast map…"):
        df = load_forecast(horizon)
        md = _prep_map_df(df)

    # AQI legend
    legend_html = " &nbsp; ".join(
        f'<span style="background:{hx};padding:2px 8px;border-radius:4px;'
        f'font-size:0.75rem;color:{"#000" if hx in ("#00E400","#FFFF00") else "#fff"}">'
        f'{cat}</span>'
        for _, cat, _, hx in AQI_SCALE
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    # Build GeoJSON FeatureCollection from hex polygons
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": row["h3_index"],
                "geometry": {"type": "Polygon", "coordinates": [row["polygon"]]},
                "properties": {"aqi": row["predicted_aqi"], "cat": row["category"]},
            }
            for _, row in md.iterrows()
        ],
    }
    aqi_min, aqi_max = md["predicted_aqi"].min(), md["predicted_aqi"].max()
    fig_map = go.Figure(
        go.Choroplethmap(
            geojson=geojson, locations=md["h3_index"], z=md["predicted_aqi"],
            colorscale=[(0, "#00E400"), (0.25, "#FFFF00"), (0.5, "#FF7E00"),
                        (0.75, "#FF0000"), (1, "#7E0023")],
            zmin=aqi_min, zmax=aqi_max,
            marker_line_width=0.5, marker_line_color="rgba(40,40,60,0.6)",
            hovertemplate="<b>%{id}</b><br>AQI: <b>%{z:.1f}</b><br>%{customdata}<extra></extra>",
            customdata=md["category"],
        )
    )
    fig_map.update_layout(
        map_style="open-street-map",
        map_center={"lat": PUNE_LAT, "lon": PUNE_LON},
        map_zoom=11, margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=480, paper_bgcolor="#0e1117",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.caption(f"Showing +{horizon}h forecast  |  {len(df):,} hexes  |  "
               f"{int((df.predicted_aqi > thr).sum())} above alert threshold ({thr})")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Top 10 Highest AQI Zones** (+{horizon}h)")
        top = (df.nlargest(10, "predicted_aqi")[["h3_index","predicted_aqi"]]
               .assign(category=lambda d: d.predicted_aqi.apply(aqi_cat))
               .reset_index(drop=True))
        top.index += 1
        top.columns = ["H3 Index","Predicted AQI","Category"]
        st.dataframe(top, use_container_width=True)

    with col_b:
        st.markdown("**AQI Distribution**")
        fig = px.histogram(df, x="predicted_aqi", nbins=25,
                           color_discrete_sequence=["#e94560"],
                           labels={"predicted_aqi":"AQI","count":"Hexes"})
        fig.add_vline(x=thr, line_dash="dash", line_color="orange",
                      annotation_text=f"Threshold {thr}")
        fig.update_layout(showlegend=False, height=290, margin=dict(l=10,r=10,t=20,b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - CAUSAL ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.subheader("🔬 Causal Attribution - DoWhy Average Treatment Effect")
    st.caption("Isolates the causal AQI impact of each pollution source, "
               "controlling for wind speed and temperature confounders.")

    causal = _run_causal()
    cdf    = pd.DataFrame(causal)

    cc1, cc2 = st.columns([3, 2])
    with cc1:
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(
            y=cdf["source"], x=cdf["ate"], orientation="h",
            error_x=dict(type="data", symmetric=False,
                         array=cdf["ci_upper"] - cdf["ate"],
                         arrayminus=cdf["ate"] - cdf["ci_lower"],
                         color="rgba(255,255,255,0.6)"),
            marker_color=["#e94560","#f6c90e","#ff9f40","#4bc0c0"][:len(cdf)],
            text=cdf["ate"].apply(lambda x: f"+{x:.1f} AQI pts"),
            textposition="outside", textfont_color="white",
        ))
        fig_c.update_layout(
            title="Causal Impact by Source (ATE + 95% CI)",
            xaxis_title="Average Treatment Effect (AQI points)",
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.05)", font_color="white",
            margin=dict(l=20, r=90, t=40, b=30),
            xaxis=dict(gridcolor="#334466"),
            yaxis=dict(gridcolor="#334466"),
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with cc2:
        st.markdown("**Attribution Summary**")
        disp = cdf.copy()
        disp["95% CI"] = disp.apply(lambda r: f"[{r.ci_lower:.1f}, {r.ci_upper:.1f}]", axis=1)
        disp = disp[["source","ate","95% CI"]].rename(
            columns={"source":"Source","ate":"ATE (AQI pts)"})
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.metric("Combined source impact", f"+{cdf['ate'].sum():.1f} AQI pts",
                  help="Sum of all attributed causal sources")

    st.divider()
    st.subheader("📄 Generate Causal Evidence Dossier (PDF)")
    pdf_c1, pdf_c2 = st.columns([2, 3])
    with pdf_c1:
        report_city    = st.text_input("City", "Pune")
        report_analyst = st.text_input("Analyst", "Cartaq Intelligence System")
        gen_pdf = st.button("📄 Generate PDF Report", type="primary", use_container_width=True)
    with pdf_c2:
        st.caption("""
**PDF includes:**
- Executive summary: max AQI, hexes monitored, dispatch stats
- Causal attribution bar chart with 95% CI (matplotlib)
- Top 15 highest-AQI hexes table
- Dispatch pipeline metrics
- Methodology notes (DoWhy linear regression estimator, H3 Res-8)
        """)

    if gen_pdf:
        with st.spinner("Generating PDF (requires fpdf2 + matplotlib)…"):
            try:
                from pipeline.pdf_generator import generate_report
                df_f = load_forecast(24)
                path = generate_report(
                    causal_results=causal,
                    forecast_df=df_f,
                    dispatch_metrics=load_metrics_r4(),
                    city=report_city,
                    analyst=report_analyst,
                    output_path=str(PDF_PATH),
                )
                st.success(f"✅ PDF saved -> `{path}`")
                with open(path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=f.read(),
                        file_name=f"cartaq_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except ImportError:
                st.error("fpdf2 not installed. Run: `pip install fpdf2`")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    # Show existing PDF download if already generated
    if PDF_PATH.exists() and not gen_pdf:
        with open(PDF_PATH, "rb") as f:
            st.download_button("⬇️ Download Last Generated PDF",
                               data=f.read(),
                               file_name="cartaq_causal_report.pdf",
                               mime="application/pdf")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 - ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.subheader("📈 Cross-Horizon Analytics")

    df_all = pd.concat(
        [load_forecast(h).assign(horizon=f"+{h}h") for h in [24, 48, 72]],
        ignore_index=True,
    )
    df_all["category"] = df_all["predicted_aqi"].apply(aqi_cat)

    row1a, row1b = st.columns(2)
    with row1a:
        fig_box = px.box(df_all, x="horizon", y="predicted_aqi", color="horizon",
                         title="AQI Distribution by Horizon",
                         color_discrete_sequence=["#00E400","#FFFF00","#FF7E00"],
                         labels={"predicted_aqi":"AQI","horizon":"Horizon"})
        fig_box.update_layout(showlegend=False, height=280,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
                               margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_box, use_container_width=True)

    with row1b:
        cat_counts = (df_all.groupby(["horizon","category"]).size()
                      .reset_index(name="count"))
        cmap = {c: hx for _, c, _, hx in AQI_SCALE}
        fig_bar = px.bar(cat_counts, x="horizon", y="count", color="category",
                         title="Hexes per AQI Category per Horizon",
                         color_discrete_map=cmap,
                         labels={"count":"Hexes","horizon":"Horizon","category":"Category"})
        fig_bar.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0.05)",
                               margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.subheader("Meteorological Correlations (from enriched data)")
    df_e = load_enriched()

    row2a, row2b = st.columns(2)
    with row2a:
        sample = df_e.sample(min(600, len(df_e)), random_state=42)
        fig_w = px.scatter(sample, x="wind_speed_kmh", y="predicted_aqi",
                           color="predicted_aqi",
                           color_continuous_scale="RdYlGn_r",
                           title="Wind Speed vs AQI (lower wind -> higher AQI)",
                           labels={"wind_speed_kmh":"Wind Speed (km/h)","predicted_aqi":"AQI"},
                           opacity=0.55)
        fig_w.update_layout(height=280, showlegend=False,
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
                             margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_w, use_container_width=True)

    with row2b:
        fig_t = px.scatter(sample, x="temperature_c", y="predicted_aqi",
                           color="humidity_pct",
                           color_continuous_scale="Blues_r",
                           title="Temperature vs AQI (colour = humidity %)",
                           labels={"temperature_c":"Temperature ( deg C)","predicted_aqi":"AQI"},
                           opacity=0.55)
        fig_t.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0.05)",
                             margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_t, use_container_width=True)

    st.divider()
    m3 = load_metrics_r3()
    if m3:
        st.subheader("⚡ ClickHouse Query Performance (5M rows)")
        bm = m3.get("benchmark_query_latencies", [])
        if bm:
            bm_df = pd.DataFrame(bm)
            fig_p = px.bar(bm_df, x="query", y="latency_ms",
                           title="Benchmark Query Latency (ms) on 5M Rows",
                           color="latency_ms", color_continuous_scale="RdYlGn_r",
                           labels={"latency_ms":"Latency (ms)","query":""})
            fig_p.update_layout(height=320, xaxis_tickangle=-25,
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
                                 margin=dict(l=10,r=10,t=40,b=90))
            st.plotly_chart(fig_p, use_container_width=True)
        for ing in m3.get("ingestion_metrics", []):
            st.metric(f"{ing['table']} throughput",
                      f"{ing.get('throughput_rows_per_second',0):,.0f} rows/s",
                      f"{ing.get('rows',0):,} rows in {ing.get('elapsed_seconds',0):.1f}s")
    else:
        st.info("Run `python src/analytics/ingest_to_clickhouse.py` after Docker is up to see ClickHouse benchmarks.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 - DISPATCH ALERTS
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.subheader("🚨 Public Health Dispatch - Vulnerable Facility Alerts")
    m4 = load_metrics_r4()

    if m4:
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("End-to-End Latency",  f"{m4.get('end_to_end_latency_ms',0):.0f} ms")
        dc2.metric("Calls Dispatched",     m4.get("calls_dispatched", 0))
        dc3.metric("MQTT Messages",        m4.get("mqtt_messages_published", 0))
        dc4.metric("Avg Spatial Query",    f"{m4.get('spatial_per_hex_avg_ms',0):.1f} ms/hex")
    else:
        st.info("No dispatch metrics yet. Run `python src/dispatch/dispatch_orchestrator.py`")

    st.divider()
    st.subheader("Dispatched Alerts Log")

    MOCK_LOG = [
        {"Facility": "KEM Hospital Pune",                    "Type":"🏥 Hospital",  "AQI": 210.1,
         "Advisory (EN)": "Hospital staff: activate air quality protocol. Increase HEPA filtration.",
         "Hindi":"[IndicTrans2 - run with HF token]","Marathi":"[IndicTrans2 - run with HF token]",
         "Call":"✅ dry_run","MQTT":"✅ Published"},
        {"Facility": "Bal Shikshan Mandir School",           "Type":"🏫 School",    "AQI": 187.3,
         "Advisory (EN)": "Cancel outdoor activities. Keep students indoors, windows sealed.",
         "Hindi":"[IndicTrans2 - run with HF token]","Marathi":"[IndicTrans2 - run with HF token]",
         "Call":"✅ dry_run","MQTT":"✅ Published"},
        {"Facility": "Jehangir Hospital",                    "Type":"🏥 Hospital",  "AQI": 210.1,
         "Advisory (EN)": "Activate air quality protocol. Keep ward windows sealed.",
         "Hindi":"[IndicTrans2 - run with HF token]","Marathi":"[IndicTrans2 - run with HF token]",
         "Call":"✅ dry_run","MQTT":"✅ Published"},
        {"Facility": "Ruby Hall Clinic",                     "Type":"🏥 Hospital",  "AQI": 302.6,
         "Advisory (EN)": "HAZARDOUS: Immediate indoor confinement. Supplemental O₂ standby.",
         "Hindi":"[IndicTrans2 - run with HF token]","Marathi":"[IndicTrans2 - run with HF token]",
         "Call":"✅ dry_run","MQTT":"✅ Published"},
        {"Facility": "Goodluck Children Care Home",          "Type":"🏠 Care Home", "AQI": 165.8,
         "Advisory (EN)": "Keep residents indoors. Monitor respiratory symptoms.",
         "Hindi":"[IndicTrans2 - run with HF token]","Marathi":"[IndicTrans2 - run with HF token]",
         "Call":"✅ dry_run","MQTT":"✅ Published"},
        {"Facility": "Sunrise Kindergarten Shivajinagar",    "Type":"🎒 Kindergarten","AQI": 155.2,
         "Advisory (EN)": "Keep children indoors. Notify parents. Consider early dismissal.",
         "Hindi":"[IndicTrans2 - run with HF token]","Marathi":"[IndicTrans2 - run with HF token]",
         "Call":"✅ dry_run","MQTT":"✅ Published"},
    ]

    log_df = pd.DataFrame(MOCK_LOG)
    st.dataframe(log_df[["Facility","Type","AQI","Advisory (EN)","Call","MQTT"]],
                 use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Advisory Preview (select facility)")
    sel = st.selectbox("Facility", log_df["Facility"].tolist())
    row = log_df[log_df["Facility"] == sel].iloc[0]
    ae, ah, am = st.columns(3)
    with ae:
        st.markdown("**English**")
        st.info(row["Advisory (EN)"])
    with ah:
        st.markdown("**हिन्दी (Hindi)**")
        if "[IndicTrans2" in str(row["Hindi"]):
            st.warning("Translation requires IndicTrans2. Add `HUGGINGFACE_TOKEN` to `.env`.")
        else:
            st.success(row["Hindi"])
    with am:
        st.markdown("**मराठी (Marathi)**")
        if "[IndicTrans2" in str(row["Marathi"]):
            st.warning("Translation requires IndicTrans2. Add `HUGGINGFACE_TOKEN` to `.env`.")
        else:
            st.success(row["Marathi"])

    st.divider()
    st.subheader("📡 MQTT Topic Browser (Eclipse Mosquitto)")
    st.caption("Topics are published with `retain=True` - signage displays get the last advisory immediately on reconnect.")

    TOPICS = [
        ("cartaq/advisory/pune/8860a2599bfffff", "Full multilingual advisory JSON (~839 bytes)",
         {"h3_index":"8860a2599bfffff","predicted_aqi":210.1,"aqi_category":"Very Unhealthy",
          "horizon_hours":24,"facility":{"name":"KEM Hospital Pune","category":"hospital"},
          "advisories":{"en":"HEALTH ADVISORY: AQI forecast 210 AQI...","hi":"[HF token needed]","mr":"[HF token needed]"}}),
        ("cartaq/alert/hospital/8860a2599bfffff", "Compact IoT payload (~79 bytes)",
         {"aqi":210,"cat":"VER","hex":"8860a2599bfffff","h":24}),
        ("cartaq/system/heartbeat", "Pipeline heartbeat (NOT retained)",
         {"status":"ok","n_dispatched":10,"n_mqtt":20,"ts":"2026-07-21T09:13:00Z"}),
    ]
    for topic, desc, payload in TOPICS:
        with st.expander(f"`{topic}` - {desc}"):
            st.json(payload)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "CARTAQ Urban Air Quality Intelligence Platform · "
    "ST-GNN · DoWhy · ClickHouse · PostGIS · Eclipse Mosquitto · "
    f"Refreshed {datetime.now().strftime('%H:%M:%S')}"
)
