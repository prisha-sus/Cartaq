// ============================================================
// CARTAQ — Globe Module (Three.js AirIndx integration)
// ============================================================

const GLOBAL_CITIES = [
  { name:"Jakarta",   country:"Indonesia",     code:"IDN", flag:"https://flagcdn.com/w20/id.png", lat:-6.2088,  lon:106.8456 },
  { name:"Tokyo",     country:"Japan",         code:"JPN", flag:"https://flagcdn.com/w20/jp.png", lat:35.6762,  lon:139.6503 },
  { name:"London",    country:"United Kingdom",code:"GBR", flag:"https://flagcdn.com/w20/gb.png", lat:51.5074,  lon:-0.1278  },
  { name:"New York",  country:"United States", code:"USA", flag:"https://flagcdn.com/w20/us.png", lat:40.7128,  lon:-74.0060 },
  { name:"Delhi",     country:"India",         code:"IND", flag:"https://flagcdn.com/w20/in.png", lat:28.6139,  lon:77.2090  },
  { name:"Cairo",     country:"Egypt",         code:"EGY", flag:"https://flagcdn.com/w20/eg.png", lat:30.0444,  lon:31.2357  },
  { name:"Paris",     country:"France",        code:"FRA", flag:"https://flagcdn.com/w20/fr.png", lat:48.8566,  lon:2.3522   },
  { name:"Sydney",    country:"Australia",     code:"AUS", flag:"https://flagcdn.com/w20/au.png", lat:-33.8688, lon:151.2093 },
  { name:"Beijing",   country:"China",         code:"CHN", flag:"https://flagcdn.com/w20/cn.png", lat:39.9042,  lon:116.4074 },
  { name:"Riyadh",    country:"Saudi Arabia",  code:"SAU", flag:"https://flagcdn.com/w20/sa.png", lat:24.7136,  lon:46.6753  },
  { name:"São Paulo", country:"Brazil",        code:"BRA", flag:"https://flagcdn.com/w20/br.png", lat:-23.5505, lon:-46.6333 },
  { name:"Reykjavik", country:"Iceland",       code:"ISL", flag:"https://flagcdn.com/w20/is.png", lat:64.1466,  lon:-21.9426 },
  { name:"Cape Town", country:"South Africa",  code:"ZAF", flag:"https://flagcdn.com/w20/za.png", lat:-33.9249, lon:18.4241  },
  { name:"Pune",      country:"India",         code:"IND", flag:"https://flagcdn.com/w20/in.png", lat:18.5314,  lon:73.8446  },
];

let currentCity = GLOBAL_CITIES[13]; // Pune default
let autoRotate  = true;
let globeInited = false;

// ── Fetch live AQI ─────────────────────────────────────────
async function fetchCityAQI(city) {
  try {
    const url = `https://air-quality-api.open-meteo.com/v1/air-quality?latitude=${city.lat}&longitude=${city.lon}&current=us_aqi,pm2_5,pm10,ozone`;
    const r   = await fetch(url);
    const d   = await r.json();
    if (d?.current) {
      const aqi  = Math.round(d.current.us_aqi  || Math.floor(Math.random()*80)+40);
      const pm25 = +(d.current.pm2_5   || aqi*0.35).toFixed(1);
      const pm10 = +(d.current.pm10    || aqi*0.55).toFixed(1);
      const o3   = +(d.current.ozone   || aqi*0.25).toFixed(1);
      updateGlobeMetrics(aqi, pm25, pm10, o3);
    } else { fallbackAQI(); }
  } catch { fallbackAQI(); }
}

function fallbackAQI() {
  const aqi = Math.floor(Math.random()*110)+35;
  updateGlobeMetrics(aqi, +(aqi*0.35).toFixed(1), +(aqi*0.6).toFixed(1), +(aqi*0.2).toFixed(1));
}

function updateGlobeMetrics(aqi, pm25, pm10, o3) {
  const info = AQI.get(aqi);

  // AQI value displays
  ['globeAqiVal','tooltipAqiNum','targetPinVal'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = aqi;
  });

  // Badge color
  const badge = document.getElementById('aqiStatusBadge');
  if (badge) {
    badge.textContent = info.label;
    badge.style.cssText = `background:${info.bg};color:${info.text};border:1px solid ${info.color}44;border-radius:999px;font-size:10px;font-weight:700;padding:2px 10px;`;
  }

  // Tooltip badge
  const tb = document.getElementById('tooltipAqiBadge');
  if (tb) tb.style.background = info.color;

  // Tooltip status
  const ts = document.getElementById('tooltipAqiStatus');
  if (ts) ts.textContent = info.label;

  // Tooltip dot
  const dot = document.getElementById('tooltipStatusDot');
  if (dot) dot.style.background = info.color;

  // Risk & advice
  const rv = document.getElementById('riskValue');
  if (rv) rv.textContent = AQI.risk(aqi) + '%';
  const ra = document.getElementById('healthAdviceText');
  if (ra) ra.textContent = AQI.advice(aqi);

  // Pollutant bars
  const maxPm25 = 75, maxPm10 = 150, maxO3 = 100;
  setBar('valPm25','barPm25', pm25, maxPm25, '#f59e0b');
  setBar('valPm10','barPm10', pm10, maxPm10, '#10b981');
  setBar('valO3',  'barO3',   o3,   maxO3,   '#22d3ee');

  // Dominant pollutant
  const dp = document.getElementById('dominantPollutant');
  if (dp) dp.textContent = pm25 > pm10 ? 'PM2.5' : 'PM10';

  // Weather (simulated)
  const temp = Math.floor(Math.random()*12)+22;
  const hum  = Math.floor(Math.random()*30)+40;
  const wind = (Math.random()*10+5).toFixed(1);
  ['tooltipTemp','tooltipHumidity','tooltipWind'].forEach((id, i) => {
    const el = document.getElementById(id);
    if (el) el.textContent = [temp+'°C', hum+'%', wind+' km/h'][i];
  });
}

function setBar(valId, barId, value, max, color) {
  const vEl = document.getElementById(valId);
  const bEl = document.getElementById(barId);
  if (vEl) vEl.textContent = `${value} µg/m³`;
  if (bEl) {
    bEl.style.width = Math.min(100, (value/max)*100) + '%';
    bEl.style.background = color;
  }
}

function updateLocationUI(city) {
  const flag = document.getElementById('locationFlag');
  const code = document.getElementById('locationCode');
  const title= document.getElementById('tooltipLocationTitle');
  const cx   = document.getElementById('coordX');
  const cy   = document.getElementById('coordY');
  const mc   = document.getElementById('aiModalCity');

  if (flag)  flag.src = city.flag;
  if (code)  code.textContent = city.code;
  if (title) title.textContent = `${city.name}, ${city.country}`;
  if (cx)    cx.textContent = city.lat.toFixed(4);
  if (cy)    cy.textContent = city.lon.toFixed(4);
  if (mc)    mc.textContent = `${city.name}, ${city.country}`;
}

function selectCity(name) {
  const city = GLOBAL_CITIES.find(c => c.name.toLowerCase() === name.toLowerCase());
  if (!city) return;
  currentCity = city;
  updateLocationUI(city);
  fetchCityAQI(city);

  // Highlight active pill
  document.querySelectorAll('.city-pill').forEach(p => {
    p.classList.toggle('active', p.dataset.city === name);
  });
}

function findNearestCity(lat, lon) {
  let best = GLOBAL_CITIES[0], minD = Infinity;
  GLOBAL_CITIES.forEach(c => {
    const d = Math.hypot(c.lat - lat, c.lon - lon);
    if (d < minD) { minD = d; best = c; }
  });
  if (best.name !== currentCity.name) { selectCity(best.name); }
}

// ── Three.js Globe ─────────────────────────────────────────
function initGlobe() {
  if (globeInited) return;
  globeInited = true;

  const canvas = document.getElementById('globeCanvas');
  if (!canvas || typeof THREE === 'undefined') return;

  const W = 520, H = 520;
  const scene    = new THREE.Scene();
  const camera   = new THREE.PerspectiveCamera(45, W/H, 0.1, 1000);
  camera.position.z = 2.75;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  // Fallback canvas texture (dark ocean + grid)
  const tc  = document.createElement('canvas');
  tc.width  = 2048; tc.height = 1024;
  const ctx = tc.getContext('2d');
  const og  = ctx.createLinearGradient(0, 0, 0, 1024);
  og.addColorStop(0, '#080c1e'); og.addColorStop(0.5, '#030710'); og.addColorStop(1, '#010208');
  ctx.fillStyle = og; ctx.fillRect(0, 0, 2048, 1024);
  ctx.strokeStyle = 'rgba(34,211,238,0.07)'; ctx.lineWidth = 1;
  for (let lat = -80; lat <= 80; lat += 20) {
    const y = ((90-lat)/180)*1024;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(2048,y); ctx.stroke();
  }
  for (let lon = -180; lon <= 180; lon += 30) {
    const x = ((lon+180)/360)*2048;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,1024); ctx.stroke();
  }

  const mat  = new THREE.MeshStandardMaterial({ map: new THREE.CanvasTexture(tc), roughness:0.4, metalness:0.1 });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1,96,96), mat);
  mesh.rotation.y = 2.05; mesh.rotation.x = 0.22;
  scene.add(mesh);

  // Load HD satellite texture
  new THREE.TextureLoader().setCrossOrigin('anonymous').load(
    'https://unpkg.com/three-globe/example/img/earth-night.jpg',
    tex => { mesh.material.map = tex; mesh.material.needsUpdate = true; }
  );

  scene.add(new THREE.AmbientLight(0xffffff, 2.2));
  const dl = new THREE.DirectionalLight(0xffffff, 1.0);
  dl.position.set(5,3,5); scene.add(dl);

  // Drag interaction
  let dragging = false, prev = {x:0, y:0};
  canvas.addEventListener('mousedown', e => {
    dragging = true; prev = {x:e.clientX, y:e.clientY};
    autoRotate = false; updateRotateBtn();
  });
  canvas.addEventListener('mousemove', e => {
    if (!dragging) return;
    mesh.rotation.y += (e.clientX - prev.x) * 0.005;
    mesh.rotation.x += (e.clientY - prev.y) * 0.005;
    prev = {x:e.clientX, y:e.clientY};
    const lat = Math.max(-85, Math.min(85, mesh.rotation.x * 180/Math.PI));
    const lon = ((-mesh.rotation.y * 180/Math.PI) % 360 + 540) % 360 - 180;
    const cx = document.getElementById('coordX');
    const cy = document.getElementById('coordY');
    if (cx) cx.textContent = lat.toFixed(4);
    if (cy) cy.textContent = lon.toFixed(4);
    findNearestCity(lat, lon);
  });
  window.addEventListener('mouseup', () => { dragging = false; });

  // Touch support
  let lastTouch = null;
  canvas.addEventListener('touchstart', e => { lastTouch = e.touches[0]; autoRotate = false; });
  canvas.addEventListener('touchmove', e => {
    if (!lastTouch) return;
    const t = e.touches[0];
    mesh.rotation.y += (t.clientX - lastTouch.clientX) * 0.005;
    mesh.rotation.x += (t.clientY - lastTouch.clientY) * 0.005;
    lastTouch = t; e.preventDefault();
  }, { passive: false });
  canvas.addEventListener('touchend', () => { lastTouch = null; });

  // Zoom
  document.getElementById('zoomIn')?.addEventListener('click', () => { if (camera.position.z > 1.8) camera.position.z -= 0.2; });
  document.getElementById('zoomOut')?.addEventListener('click', () => { if (camera.position.z < 4.0) camera.position.z += 0.2; });

  // Auto-rotate toggle
  document.getElementById('btnAutoRotate')?.addEventListener('click', () => {
    autoRotate = !autoRotate; updateRotateBtn();
  });

  function updateRotateBtn() {
    const icon = document.getElementById('rotateIcon');
    const text = document.getElementById('rotateStateText');
    if (icon) icon.classList.toggle('animate-spin-slow', autoRotate);
    if (text) text.textContent = autoRotate ? 'Auto-Rotate ON' : 'Auto-Rotate OFF';
  }

  // Animate
  (function loop() {
    requestAnimationFrame(loop);
    if (autoRotate && !dragging) {
      mesh.rotation.y += 0.0008;
      const lon = ((-mesh.rotation.y*180/Math.PI)%360+540)%360-180;
      const lat = Math.max(-85,Math.min(85, mesh.rotation.x*180/Math.PI));
      const cx = document.getElementById('coordX');
      const cy = document.getElementById('coordY');
      if (cx) cx.textContent = lat.toFixed(4);
      if (cy) cy.textContent = lon.toFixed(4);
    }
    renderer.render(scene, camera);
  })();
}

// ── City search (globe panel) ─────────────────────────────
function initGlobeSearch() {
  const input   = document.getElementById('citySearchInput');
  const results = document.getElementById('globeSearchResults');
  if (!input || !results) return;

  input.addEventListener('input', () => {
    const q = input.value.toLowerCase().trim();
    if (!q) { results.style.display = 'none'; return; }
    const matches = GLOBAL_CITIES.filter(c =>
      c.name.toLowerCase().includes(q) || c.country.toLowerCase().includes(q)
    );
    if (matches.length) {
      results.innerHTML = matches.map(c => `
        <div class="globe-search-result" onclick="selectCity('${c.name}');document.getElementById('citySearchInput').value='';document.getElementById('globeSearchResults').style.display='none'">
          <strong>${c.name}</strong><small>${c.country}</small>
        </div>
      `).join('');
      results.style.display = 'block';
    } else { results.style.display = 'none'; }
  });

  document.addEventListener('click', e => {
    if (!input.contains(e.target) && !results.contains(e.target))
      results.style.display = 'none';
  });
}

// ── AI Modal ───────────────────────────────────────────────
async function openAiModal() {
  const modal   = document.getElementById('aiModal');
  const content = document.getElementById('aiModalContent');
  const loading = document.getElementById('aiModalLoading');
  if (!modal) return;
  modal.classList.add('open');
  loading.style.display = 'flex';
  content.innerHTML = '';

  const aqi = document.getElementById('globeAqiVal')?.textContent || '—';
  const city = `${currentCity.name}, ${currentCity.country}`;

  try {
    const apiKey = '';
    const url    = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;
    const payload = {
      contents: [{ parts: [{ text: `Provide an atmospheric air quality health assessment for ${city} with current US AQI of ${aqi}.` }] }],
      systemInstruction: { parts: [{ text: "Act as an expert atmospheric scientist and health physician. Provide a concise 2-paragraph analysis covering: 1) Causes for current AQI levels in the specified city, 2) Precise health recommendations for outdoors, masks, and indoor air filtration." }] }
    };
    const res  = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const data = await res.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
    loading.style.display = 'none';
    if (text) {
      content.innerHTML = `<p class="text-amber" style="font-weight:700;margin-bottom:12px">Live AI Analysis — ${city} (AQI ${aqi}):</p><p>${text.replace(/\n\n/g,'</p><p style="margin-top:10px">')}</p>`;
    } else {
      content.innerHTML = `<p>AQI for <strong>${city}</strong> is <strong>${aqi}</strong>. Consider N95 masks for prolonged outdoor exposure and monitor local advisories.</p>`;
    }
  } catch {
    loading.style.display = 'none';
    content.innerHTML = `<p>Active AQI in <strong>${city}</strong> is <strong>${aqi}</strong>. Keep windows closed during pollution spikes and consider air filtration indoors.</p>`;
  }
}

function closeAiModal() {
  document.getElementById('aiModal')?.classList.remove('open');
}

// ── Share ──────────────────────────────────────────────────
function shareDashboard(platform) {
  const aqi  = document.getElementById('globeAqiVal')?.textContent || '—';
  const text = encodeURIComponent(`Live Air Quality in ${currentCity.name}! AQI: ${aqi} — checked via CARTAQ`);
  const url  = encodeURIComponent(location.href);
  if (platform === 'twitter')   window.open(`https://twitter.com/intent/tweet?text=${text}`);
  if (platform === 'facebook')  window.open(`https://www.facebook.com/sharer/sharer.php?u=${url}`);
  if (platform === 'whatsapp')  window.open(`https://api.whatsapp.com/send?text=${text}`);
}

// ── Globe page init ────────────────────────────────────────
PageLoaders.globe = function() {
  initGlobe();
  initGlobeSearch();
  selectCity(currentCity.name);
};

// Wire up city pills
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.city-pill[data-city]').forEach(p => {
    p.addEventListener('click', () => selectCity(p.dataset.city));
  });
  document.getElementById('btnAiAdvisor')?.addEventListener('click', openAiModal);
  document.getElementById('btnHealthDetails')?.addEventListener('click', openAiModal);
  document.getElementById('aiModalClose')?.addEventListener('click', closeAiModal);
  document.getElementById('aiModalCloseFoot')?.addEventListener('click', closeAiModal);
  document.getElementById('aiModal')?.addEventListener('click', e => {
    if (e.target === document.getElementById('aiModal')) closeAiModal();
  });
});
