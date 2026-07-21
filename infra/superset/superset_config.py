"""
Cartaq - Apache Superset Configuration
Mounted into the Superset container at /app/pythonpath/superset_config.py

Key setup steps after first boot:
    1. Browse to http://localhost:8088 and log in with admin / admin
    2. Settings -> Database Connections -> + Database
       Driver:  ClickHouse Connect (installed in the entrypoint)
       URI:     clickhousedb+connect://cartaq_user:cartaq_pass@clickhouse:8123/cartaq
    3. Create a dataset from cartaq.forecast_log
    4. Create a new chart -> Chart type: "Deck.gl H3 Hexagon"
       Longitude / Latitude columns -> leave blank (H3 index mode)
       H3 Hexagon Index -> h3_index
       Metric -> AVG(predicted_aqi)
    5. Add the chart to a dashboard
"""

import os

# ── Security ────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SUPERSET_SECRET_KEY", "cartaq_superset_secret_CHANGE_IN_PROD")

# ── Feature flags ────────────────────────────────────────────────────────────
FEATURE_FLAGS: dict = {
    # deck.gl / MapLibre geospatial charts
    "DECK_GL_MAPS": True,
    "GEOSPATIAL_TYPES": True,

    # Native dashboard filters (cross-filtering between charts)
    "DASHBOARD_NATIVE_FILTERS": True,
    "DASHBOARD_CROSS_FILTERS": True,

    # Template-based SQL (enables Jinja {{filter_values()}} macros)
    "ENABLE_TEMPLATE_PROCESSING": True,

    # Drag-and-drop chart builder
    "ENABLE_EXPLORE_DRAG_AND_DROP": True,

    # Async queries (improves UX on slow CH aggregation queries)
    "GLOBAL_ASYNC_QUERIES": False,  # Requires Redis; enable if you add redis service
}

# ── MapLibre / Mapbox tile config ────────────────────────────────────────────
# For Deck.gl maps. Using a free OpenStreetMap tile provider by default.
# To use Mapbox tiles, replace with your Mapbox token:
#   MAPBOX_API_KEY = "pk.eyJ1IjoiY2FydGFxIiwiYSI6I..."
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY", "")

# ── Session / CSRF ───────────────────────────────────────────────────────────
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SECURE = False     # Set True behind HTTPS in production
WTF_CSRF_ENABLED = True
WTF_CSRF_EXEMPT_LIST: list = []

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"

# ── SQL Lab / Query settings ─────────────────────────────────────────────────
# ClickHouse can return millions of rows; cap at a sane default.
SQL_MAX_ROW = 100_000
DISPLAY_MAX_ROW = 10_000

# ── Cache (file-based; upgrade to Redis for production) ──────────────────────
CACHE_CONFIG = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300,
}

# ── Embedded analytics ───────────────────────────────────────────────────────
GUEST_ROLE_NAME = "Public"
GUEST_TOKEN_JWT_SECRET = SECRET_KEY
GUEST_TOKEN_JWT_ALGO = "HS256"
GUEST_TOKEN_JWT_EXP_SECONDS = 300

# ── ClickHouse connection hint (for documentation; add via UI) ───────────────
# SQLAlchemy URI for Superset -> ClickHouse via clickhouse-connect:
#   clickhousedb+connect://cartaq_user:cartaq_pass@clickhouse:8123/cartaq
#
# Deck.gl H3 hexagon chart query template (paste into SQL Lab to verify data):
#   SELECT h3_index, avg(predicted_aqi) AS avg_aqi
#   FROM cartaq.forecast_log
#   WHERE future_timestamp BETWEEN '{{since}}' AND '{{until}}'
#   GROUP BY h3_index
#   ORDER BY avg_aqi DESC
