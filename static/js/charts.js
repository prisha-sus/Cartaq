// ============================================================
// CARTAQ — Charts Module (Plotly.js)
// ============================================================

const CHART_BASE = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(255,255,255,0.02)',
  font:          { family: 'Plus Jakarta Sans, sans-serif', color: '#94a3b8' },
  margin:        { l:40, r:20, t:40, b:40 },
  xaxis:         { gridcolor:'rgba(255,255,255,0.06)', zerolinecolor:'rgba(255,255,255,0.08)' },
  yaxis:         { gridcolor:'rgba(255,255,255,0.06)', zerolinecolor:'rgba(255,255,255,0.08)' },
};

function mergeLayout(extra) {
  return Object.assign({}, CHART_BASE, extra);
}

const PLOTLY_CFG = { responsive: true, displayModeBar: false };

// ── AQI Gradient Colorscale ────────────────────────────────
const AQI_COLORSCALE = [
  [0,    '#10b981'],
  [0.2,  '#f59e0b'],
  [0.4,  '#f97316'],
  [0.6,  '#ef4444'],
  [0.8,  '#a855f7'],
  [1.0,  '#881337'],
];

// ── Overview Tab ───────────────────────────────────────────
let overviewLoaded = false;

PageLoaders.overview = async function() {
  if (overviewLoaded) return;
  overviewLoaded = true;

  // KPI cards
  try {
    const data = await API.status();
    updateKPIs(data);
    updatePipelineStatus(data.pipeline);
  } catch(e) {
    console.warn('Status API error', e);
    setKPIError();
  }
};

function updateKPIs(data) {
  const k = data.kpis || {};
  setKPI('kpiHexes',   k.hexes_monitored);
  setKPI('kpiMaxAqi',  k.max_aqi_24h);
  setKPI('kpiMeanAqi', k.mean_aqi_24h);
  setKPI('kpiHigh',    k.high_aqi_zones);
  setKPI('kpiCalls',   k.calls_dispatched);
  setKPI('kpiMqtt',    k.mqtt_messages);

  const mode = document.getElementById('demoModeBadge');
  if (mode) {
    mode.textContent = data.is_demo ? '🎭 Demo Mode' : '🟢 Live Data';
    mode.style.color = data.is_demo ? '#f59e0b' : '#34d399';
  }
}

function setKPI(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = (value ?? '—').toLocaleString();
}

function setKPIError() {
  ['kpiHexes','kpiMaxAqi','kpiMeanAqi','kpiHigh','kpiCalls','kpiMqtt'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '—';
  });
}

function updatePipelineStatus(status) {
  const steps = [
    { id:'step-openaq',    label:'Ingest OpenAQ + Open-Meteo',         done: status.openaq && status.openmeteo, hint:'python src/ingestion/openaq_client.py' },
    { id:'step-graph',     label:'Build H3 Graph Tensors (Role 1)',     done: status.graph,    hint:'python src/pipeline/graph_builder.py' },
    { id:'step-forecast',  label:'ST-GNN Forecast → forecast.parquet',  done: status.forecast, hint:'python src/pipeline/run_pipeline.py --demo' },
    { id:'step-enriched',  label:'Enrich: Forecast + Weather',          done: status.enriched, hint:'python src/pipeline/enrich.py' },
    { id:'step-pdf',       label:'Causal Analysis + PDF (Role 2)',       done: status.pdf,      hint:'Use Causal tab → Generate PDF' },
    { id:'step-dispatch',  label:'Dispatch Alerts (Role 4)',             done: status.dispatch, hint:'python src/dispatch/dispatch_orchestrator.py' },
  ];

  const container = document.getElementById('pipelineSteps');
  if (!container) return;

  container.innerHTML = steps.map(s => `
    <div class="pipeline-step fade-in">
      <div class="step-icon ${s.done ? 'done' : 'pending'}">
        <i class="fa-solid ${s.done ? 'fa-check' : 'fa-clock'}"></i>
      </div>
      <div class="step-info">
        <div class="step-name">${s.label}</div>
        <div class="step-hint">${s.done ? '✓ Complete' : s.hint}</div>
      </div>
    </div>
  `).join('');
}

// ── Quick Actions ──────────────────────────────────────────
async function runPipelineAction(step, label) {
  Toast.show(`Running ${label}…`, 'This may take a minute', 'info');
  try {
    const res = await API.runPipeline(step);
    if (res.ok) {
      Toast.success(`${label} complete`, res.output?.slice(-120) || '');
    } else {
      Toast.error(`${label} failed`, res.output?.slice(-120) || '');
    }
  } catch(e) {
    Toast.error('Network error', e.message);
  }
}

// ── Forecast Map Tab ───────────────────────────────────────
let mapLoaded   = false;
let mapHorizon  = 24;
let aqiThreshold = 150;

PageLoaders.map = async function() {
  if (mapLoaded) return;
  mapLoaded = true;

  const container = document.getElementById('mapContainer');
  if (!container) return;

  await loadForecastMap(mapHorizon);

  document.getElementById('horizonSelect')?.addEventListener('change', e => {
    mapHorizon = parseInt(e.target.value);
    loadForecastMap(mapHorizon);
  });

  document.getElementById('aqiThreshold')?.addEventListener('input', e => {
    aqiThreshold = parseInt(e.target.value);
    document.getElementById('thresholdLabel').textContent = aqiThreshold;
  });
};

async function loadForecastMap(horizon) {
  const container = document.getElementById('mapContainer');
  const summary   = document.getElementById('mapSummary');
  if (!container) return;

  container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;gap:12px;color:#64748b"><div class="spinner"></div><span>Loading forecast…</span></div>';

  try {
    const res = await API.forecast(horizon);
    const data = res.data || [];

    // Build GeoJSON
    const geojson = {
      type: 'FeatureCollection',
      features: data.map(d => ({
        type: 'Feature',
        id: d.h3_index,
        geometry: { type:'Polygon', coordinates:[d.polygon] },
        properties: { aqi: d.aqi, cat: d.category }
      }))
    };

    const aqi_vals = data.map(d => d.aqi);
    const aqi_min  = Math.min(...aqi_vals);
    const aqi_max  = Math.max(...aqi_vals);

    // Top 5 hotspots as scatter markers
    const top5 = [...data].sort((a,b) => b.aqi - a.aqi).slice(0,5);

    const traces = [
      {
        type: 'choroplethmap',
        geojson,
        locations: data.map(d => d.h3_index),
        z: data.map(d => d.aqi),
        customdata: data.map(d => d.category),
        colorscale: AQI_COLORSCALE,
        zmin: aqi_min, zmax: aqi_max,
        marker: { line: { width:0.8, color:'rgba(20,20,40,0.7)' }, opacity: 0.85 },
        colorbar: {
          title: { text:'AQI', font:{color:'#94a3b8',size:11} },
          tickfont: { color:'#94a3b8', size:10 },
          bgcolor: 'rgba(7,8,17,0.8)',
          bordercolor: 'rgba(255,255,255,0.1)',
          borderwidth: 1, thickness: 14, len: 0.7,
        },
        hovertemplate: '<b>%{id}</b><br>AQI: <b>%{z:.1f}</b><br>%{customdata}<extra></extra>',
      },
      {
        type: 'scattermap',
        lat: top5.map(d => d.lat),
        lon: top5.map(d => d.lon),
        mode: 'markers+text',
        marker: {
          size: top5.map(d => Math.min(30, 14 + d.aqi/25)),
          color: top5.map(d => d.color_hex),
          opacity: 0.9,
          symbol: 'circle',
        },
        text: top5.map(d => `AQI ${d.aqi}`),
        textposition: 'top center',
        textfont: { color:'#fff', size:9, family:'JetBrains Mono' },
        hovertemplate: '<b>Hotspot</b><br>AQI: <b>%{text}</b><extra></extra>',
        name: 'Hotspots',
      }
    ];

    const layout = {
      map: {
        style: 'carto-darkmatter',
        center: { lat: 18.5314, lon: 73.8446 },
        zoom: 10,
      },
      margin: { r:0, t:0, l:0, b:0 },
      height: container.offsetHeight || 480,
      paper_bgcolor: '#070811',
      showlegend: false,
    };

    container.innerHTML = '';
    Plotly.newPlot(container, traces, layout, PLOTLY_CFG);

    if (summary) {
      const aboveThr = data.filter(d => d.aqi > aqiThreshold).length;
      summary.textContent = `+${horizon}h forecast · ${data.length.toLocaleString()} hexes · ${aboveThr} above threshold (${aqiThreshold})`;
    }

    // Distribution histogram
    renderHistogram(data, horizon);
    renderTopTable(data);

  } catch(e) {
    container.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#ef4444">Error loading map: ${e.message}</div>`;
  }
}

function renderHistogram(data, horizon) {
  const el = document.getElementById('aqiHistogram');
  if (!el) return;
  Plotly.newPlot(el, [{
    type: 'histogram', x: data.map(d => d.aqi), nbinsx: 25,
    marker: { color: data.map(d => d.color_hex), opacity: 0.8 },
    name: 'Hexes',
  }], mergeLayout({
    title: { text:`AQI Distribution (+${horizon}h)`, font:{size:13, color:'#fff'} },
    height: 260,
    xaxis: { title:{text:'AQI',font:{size:11}}, gridcolor:'rgba(255,255,255,0.06)' },
    yaxis: { title:{text:'Hexes',font:{size:11}}, gridcolor:'rgba(255,255,255,0.06)' },
    margin: { l:40, r:16, t:40, b:36 },
  }), PLOTLY_CFG);
}

function renderTopTable(data) {
  const el = document.getElementById('topHexTable');
  if (!el) return;
  const top = [...data].sort((a,b) => b.aqi - a.aqi).slice(0,10);
  el.innerHTML = `
    <table class="dispatch-table" style="font-size:11px">
      <thead><tr><th>#</th><th>H3 Index</th><th>AQI</th><th>Category</th></tr></thead>
      <tbody>
        ${top.map((d,i) => `
          <tr>
            <td style="color:var(--text-muted)">${i+1}</td>
            <td class="text-mono" style="color:var(--cyan);font-size:10px">${d.h3_index}</td>
            <td><span class="aqi-badge" style="background:${d.color_hex}22;color:${d.color_hex};border:1px solid ${d.color_hex}44">${d.aqi}</span></td>
            <td style="color:var(--text-dim)">${d.category}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>`;
}

// ── Causal Tab ─────────────────────────────────────────────
let causalLoaded = false;

PageLoaders.causal = async function() {
  if (causalLoaded) return;
  causalLoaded = true;

  try {
    const data = await API.causal();
    renderCausalBars(data.results || []);
    renderCausalChart(data.results || []);
    const total = document.getElementById('totalAte');
    if (total) total.textContent = `+${data.total_ate} AQI pts`;
  } catch(e) {
    Toast.error('Causal data error', e.message);
  }
};

function renderCausalBars(results) {
  const container = document.getElementById('causalBars');
  if (!container) return;
  const maxAte = Math.max(...results.map(r => r.ci_upper));

  container.innerHTML = results.map(r => `
    <div class="causal-bar-row">
      <div class="causal-bar-label">${r.source}</div>
      <div style="flex:1">
        <div class="causal-bar-track">
          <div class="causal-bar-fill" style="width:${(r.ate/maxAte)*100}%;background:${r.color}"></div>
        </div>
        <div class="causal-ci">95% CI: [${r.ci_lower}, ${r.ci_upper}]</div>
      </div>
      <div class="causal-bar-value" style="color:${r.color}">+${r.ate}</div>
    </div>
  `).join('');
}

function renderCausalChart(results) {
  const el = document.getElementById('causalChart');
  if (!el) return;
  Plotly.newPlot(el, [{
    type: 'bar', orientation: 'h',
    y: results.map(r => r.source),
    x: results.map(r => r.ate),
    error_x: {
      type: 'data', symmetric: false,
      array: results.map(r => r.ci_upper - r.ate),
      arrayminus: results.map(r => r.ate - r.ci_lower),
      color: 'rgba(255,255,255,0.5)', thickness: 2,
    },
    marker: { color: results.map(r => r.color), opacity: 0.85 },
    text: results.map(r => `+${r.ate} AQI pts`),
    textposition: 'outside', textfont: { color:'#fff', size:11 },
  }], mergeLayout({
    title:  { text:'Causal Impact by Source (ATE + 95% CI)', font:{size:13,color:'#fff'} },
    xaxis:  { title:{text:'Average Treatment Effect (AQI pts)',font:{size:11}}, gridcolor:'rgba(255,255,255,0.06)' },
    height: 320, margin:{ l:150, r:100, t:48, b:36 },
  }), PLOTLY_CFG);
}

// ── Analytics Tab ──────────────────────────────────────────
let analyticsLoaded = false;

PageLoaders.analytics = async function() {
  if (analyticsLoaded) return;
  analyticsLoaded = true;

  try {
    const data = await API.analytics();
    renderBoxPlot(data.by_horizon);
    renderCategoryBar(data.by_horizon, data.category_colors);
    renderWeatherScatter(data.weather_correlation);
  } catch(e) {
    Toast.error('Analytics data error', e.message);
  }
};

function renderBoxPlot(byHorizon) {
  const el = document.getElementById('boxChart');
  if (!el) return;
  const colors = {'24':'#10b981','48':'#f59e0b','72':'#f97316'};
  const traces = Object.entries(byHorizon).map(([h, rows]) => ({
    type: 'box', name: `+${parseInt(h)}h`,
    y: rows.map(r => r.predicted_aqi),
    marker: { color: colors[parseInt(h)] || '#6366f1' },
    line: { color: colors[parseInt(h)] || '#6366f1', width:2 },
    fillcolor: (colors[parseInt(h)] || '#6366f1') + '33',
    boxmean: 'sd',
  }));
  Plotly.newPlot(el, traces, mergeLayout({
    title:  { text:'AQI Distribution by Forecast Horizon', font:{size:13,color:'#fff'} },
    yaxis:  { title:{text:'AQI',font:{size:11}}, gridcolor:'rgba(255,255,255,0.06)' },
    height: 280, showlegend: false, margin:{l:44,r:16,t:44,b:36},
  }), PLOTLY_CFG);
}

function renderCategoryBar(byHorizon, catColors) {
  const el = document.getElementById('catBar');
  if (!el) return;
  const horizons = Object.keys(byHorizon);
  const cats     = Object.keys(catColors);
  const traces   = cats.map(cat => ({
    type: 'bar', name: cat,
    x: horizons,
    y: horizons.map(h => byHorizon[h].filter(r => r.category === cat).length),
    marker: { color: catColors[cat], opacity: 0.8 },
  }));
  Plotly.newPlot(el, traces, mergeLayout({
    title:   { text:'Hexes per AQI Category per Horizon', font:{size:13,color:'#fff'} },
    barmode: 'stack', height: 280,
    legend:  { font:{color:'#94a3b8',size:10}, bgcolor:'rgba(0,0,0,0)', orientation:'h', y:-0.25 },
    margin:  { l:44, r:16, t:44, b:60 },
  }), PLOTLY_CFG);
}

function renderWeatherScatter(rows) {
  const windEl = document.getElementById('windScatter');
  const tempEl = document.getElementById('tempScatter');

  if (windEl) {
    Plotly.newPlot(windEl, [{
      type: 'scatter', mode: 'markers',
      x: rows.map(r => r.wind_speed_kmh),
      y: rows.map(r => r.predicted_aqi),
      marker: {
        color: rows.map(r => r.predicted_aqi),
        colorscale: AQI_COLORSCALE, size:6, opacity:0.65,
        showscale: false,
      },
      hovertemplate: 'Wind: %{x:.1f} km/h<br>AQI: %{y:.0f}<extra></extra>',
    }], mergeLayout({
      title: { text:'Wind Speed vs AQI (lower wind → higher AQI)', font:{size:12,color:'#fff'} },
      xaxis: { title:{text:'Wind Speed (km/h)',font:{size:11}}, gridcolor:'rgba(255,255,255,0.06)' },
      yaxis: { title:{text:'AQI',font:{size:11}}, gridcolor:'rgba(255,255,255,0.06)' },
      height:280, margin:{l:50,r:16,t:44,b:44},
    }), PLOTLY_CFG);
  }

  if (tempEl) {
    Plotly.newPlot(tempEl, [{
      type: 'scatter', mode: 'markers',
      x: rows.map(r => r.temperature_c),
      y: rows.map(r => r.predicted_aqi),
      marker: {
        color: rows.map(r => r.humidity_pct),
        colorscale: 'Blues_r', size:6, opacity:0.65,
        colorbar: { title:{text:'Humidity %',font:{size:10,color:'#94a3b8'}}, tickfont:{size:9,color:'#94a3b8'}, thickness:10, len:0.5 },
      },
      hovertemplate: 'Temp: %{x:.1f}°C<br>AQI: %{y:.0f}<extra></extra>',
    }], mergeLayout({
      title: { text:'Temperature vs AQI (color = humidity %)', font:{size:12,color:'#fff'} },
      xaxis: { title:{text:'Temperature (°C)',font:{size:11}}, gridcolor:'rgba(255,255,255,0.06)' },
      yaxis: { title:{text:'AQI',font:{size:11}}, gridcolor:'rgba(255,255,255,0.06)' },
      height:280, margin:{l:50,r:60,t:44,b:44},
    }), PLOTLY_CFG);
  }
}

// ── Dispatch Tab ───────────────────────────────────────────
let dispatchLoaded = false;

PageLoaders.dispatch = async function() {
  if (dispatchLoaded) return;
  dispatchLoaded = true;

  try {
    const data = await API.dispatch();
    renderDispatchMetrics(data.metrics || {});
    renderDispatchTable(data.alerts || []);
    renderMqttTopics(data.mqtt_topics || []);
  } catch(e) {
    Toast.error('Dispatch data error', e.message);
  }
};

function renderDispatchMetrics(m) {
  setKPI('dispLatency',  m.end_to_end_latency_ms ? m.end_to_end_latency_ms.toFixed(0)+' ms' : '—');
  setKPI('dispCalls',    m.calls_dispatched ?? '—');
  setKPI('dispMqtt',     m.mqtt_messages_published ?? '—');
  setKPI('dispHexMs',    m.spatial_per_hex_avg_ms ? m.spatial_per_hex_avg_ms.toFixed(1)+' ms/hex' : '—');
}

function renderDispatchTable(alerts) {
  const el = document.getElementById('dispatchTable');
  if (!el) return;
  el.innerHTML = `
    <table class="dispatch-table" style="width:100%">
      <thead><tr>
        <th>Facility</th><th>Type</th><th>AQI</th><th>Advisory</th><th>Call</th><th>MQTT</th>
      </tr></thead>
      <tbody>
        ${alerts.map(a => `
          <tr>
            <td style="font-weight:600;color:#fff">${a.facility}</td>
            <td style="color:var(--text-muted)">${a.type}</td>
            <td><span class="aqi-badge" style="background:${a.color}22;color:${a.color};border:1px solid ${a.color}44">${a.aqi}</span></td>
            <td style="color:var(--text-dim);font-size:11px;max-width:300px">${a.advisory}</td>
            <td><span class="status-pill ${a.status==='HAZARDOUS'?'hazardous':'dispatched'}">✓</span></td>
            <td><span class="status-pill dispatched">✓</span></td>
          </tr>
        `).join('')}
      </tbody>
    </table>`;
}

function renderMqttTopics(topics) {
  const el = document.getElementById('mqttTopics');
  if (!el) return;
  el.innerHTML = topics.map(t => `
    <div class="mqtt-topic">
      <div class="mqtt-topic-header">
        <div><code>${t.topic}</code><span style="color:var(--text-muted);font-size:10px;margin-left:10px">${t.description}</span></div>
        <i class="fa-solid fa-chevron-down mqtt-chevron" style="color:var(--text-muted);font-size:10px;transition:transform 0.2s"></i>
      </div>
      <div class="mqtt-topic-body">
        <pre>${JSON.stringify(t.payload, null, 2)}</pre>
      </div>
    </div>
  `).join('');

  // Re-init expanders for newly added elements
  el.querySelectorAll('.mqtt-topic-header').forEach(h => {
    h.addEventListener('click', () => {
      const body = h.nextElementSibling;
      body.classList.toggle('open');
      h.querySelector('.mqtt-chevron').style.transform = body.classList.contains('open') ? 'rotate(180deg)' : '';
    });
  });
}
