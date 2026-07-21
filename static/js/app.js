// ============================================================
// CARTAQ — API Client
// Fetches data from the FastAPI backend
// ============================================================

const API = {
  base: '', // same origin

  async get(path) {
    const res = await fetch(this.base + path);
    if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
    return res.json();
  },

  async post(path, body = {}) {
    const res = await fetch(this.base + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
    return res.json();
  },

  status:       ()          => API.get('/api/status'),
  forecast:     (h = 24)    => API.get(`/api/forecast/${h}`),
  analytics:    ()          => API.get('/api/analytics'),
  causal:       ()          => API.get('/api/causal'),
  dispatch:     ()          => API.get('/api/dispatch'),
  metrics:      ()          => API.get('/api/metrics'),
  runPipeline:  (step)      => API.post(`/api/pipeline/run?step=${step}`),
};

// ── Toast Notifications ────────────────────────────────────
const Toast = {
  el: null,
  timer: null,

  init() { this.el = document.getElementById('toast'); },

  show(title, body = '', type = 'info') {
    if (!this.el) return;
    this.el.querySelector('.toast-title').textContent = title;
    this.el.querySelector('.toast-body').textContent  = body;
    this.el.className = `show ${type}`;
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.hide(), 4000);
  },

  hide() {
    if (this.el) this.el.className = '';
  },

  success(t, b) { this.show(t, b, 'success'); },
  error(t, b)   { this.show(t, b, 'error'); },
};

// ── Tab / Page Router ─────────────────────────────────────
const Router = {
  pages: {},
  tabs: {},
  current: null,

  init() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
      const page = tab.dataset.page;
      this.tabs[page] = tab;
      tab.addEventListener('click', () => this.navigate(page));
    });

    document.querySelectorAll('.page').forEach(p => {
      this.pages[p.id.replace('page-', '')] = p;
    });

    // Load from hash
    const hash = location.hash.replace('#', '') || 'globe';
    this.navigate(hash);
  },

  navigate(page) {
    // Hide all pages
    Object.values(this.pages).forEach(p => p.classList.remove('active'));
    Object.values(this.tabs).forEach(t => t.classList.remove('active'));

    const target = this.pages[page] || this.pages['globe'];
    const targetKey = target ? Object.keys(this.pages).find(k => this.pages[k] === target) : 'globe';

    if (target) target.classList.add('active');
    if (this.tabs[targetKey]) this.tabs[targetKey].classList.add('active');

    location.hash = targetKey;
    this.current = targetKey;

    // Lazy-load page data
    PageLoaders[targetKey]?.();
  }
};

// ── AQI Helpers ────────────────────────────────────────────
const AQI = {
  scale: [
    { max: 50,  label: 'Good',                color: '#10b981', bg: 'rgba(16,185,129,0.2)',  text: 'rgba(52,211,153,1)'  },
    { max: 100, label: 'Moderate',             color: '#f59e0b', bg: 'rgba(245,158,11,0.2)',  text: 'rgba(251,191,36,1)'  },
    { max: 150, label: 'Unhealthy-Sensitive',  color: '#f97316', bg: 'rgba(249,115,22,0.2)',  text: 'rgba(251,146,60,1)'  },
    { max: 200, label: 'Unhealthy',            color: '#ef4444', bg: 'rgba(239,68,68,0.2)',   text: 'rgba(248,113,113,1)' },
    { max: 300, label: 'Very Unhealthy',       color: '#a855f7', bg: 'rgba(168,85,247,0.2)',  text: 'rgba(192,132,252,1)' },
    { max: 999, label: 'Hazardous',            color: '#881337', bg: 'rgba(136,19,55,0.2)',   text: 'rgba(251,113,133,1)' },
  ],

  get(v) {
    for (const s of this.scale) if (v <= s.max) return s;
    return this.scale[this.scale.length - 1];
  },

  advice(v) {
    if (v <= 50)  return 'Air quality is ideal. Enjoy all outdoor activities freely.';
    if (v <= 100) return 'Acceptable air quality. Unusually sensitive individuals should limit prolonged exertion.';
    if (v <= 150) return 'Sensitive groups may experience effects. General public unlikely to be impacted.';
    if (v <= 200) return 'Everyone may experience health effects. Sensitive groups: stay indoors.';
    if (v <= 300) return 'Health alert: serious effects for everyone. Avoid all outdoor activities.';
    return 'HAZARDOUS: Emergency conditions. Stay indoors with air filtration. Seal windows.';
  },

  risk(v) { return Math.min(95, Math.max(10, Math.round(v * 0.45))); },
};

// ── Clock ─────────────────────────────────────────────────
function startClock(elId) {
  const el = document.getElementById(elId);
  if (!el) return;
  const update = () => {
    el.textContent = new Date().toLocaleString('en-GB', {
      day: '2-digit', month: '2-digit', year: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
  };
  update();
  setInterval(update, 1000);
}

// ── MQTT Expanders ────────────────────────────────────────
function initMqttExpanders() {
  document.querySelectorAll('.mqtt-topic-header').forEach(h => {
    h.addEventListener('click', () => {
      const body = h.nextElementSibling;
      body.classList.toggle('open');
      h.querySelector('.mqtt-chevron').style.transform = body.classList.contains('open') ? 'rotate(180deg)' : '';
    });
  });
}

// ── Page Loaders (lazy) ────────────────────────────────────
const PageLoaders = {
  globe:    null, // handled by globe.js
  overview: null, // handled below
  map:      null,
  causal:   null,
  analytics:null,
  dispatch: null,
};

// DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  Toast.init();
  Router.init();
  startClock('liveClock');
  initMqttExpanders();
});
