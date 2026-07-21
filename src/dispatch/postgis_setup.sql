-- ============================================================
-- Cartaq Role 4 — PostGIS Schema
-- Vulnerability vectors + forecast import + dispatch log.
--
-- Run against a PostGIS-enabled PostgreSQL instance:
--   psql "host=localhost port=5432 dbname=cartaq user=cartaq_user password=cartaq_pass" \
--        -f src/dispatch/postgis_setup.sql
--
-- The cartaq_postgis Docker container (infra/docker-compose.yml)
-- already has PostGIS pre-installed; run this script to initialize schema.
-- ============================================================

-- ── Extensions ───────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
-- NOTE: h3-pg (h3_cell_to_boundary_geometry) is NOT installed.
-- Hex boundaries are computed in Python via h3.cell_to_boundary()
-- and passed to PostGIS as WKT strings. No h3-pg dependency needed.

-- ── 1. Forecast Hexes ────────────────────────────────────────
-- Staging table: receives Role 1 output (or mock_payload.json events).
-- Each high-AQI hex is inserted here before spatial intersection.
CREATE TABLE IF NOT EXISTS forecast_hexes (
    h3_index          TEXT         NOT NULL,
    future_timestamp  TIMESTAMPTZ  NOT NULL,
    predicted_aqi     DOUBLE PRECISION NOT NULL,
    horizon_hours     SMALLINT,
    aqi_category      TEXT,
    -- Hex boundary polygon (populated by spatial_query.py)
    geom              GEOMETRY(POLYGON, 4326),
    ingested_at       TIMESTAMPTZ  DEFAULT NOW(),
    PRIMARY KEY (h3_index, future_timestamp)
);
CREATE INDEX IF NOT EXISTS idx_fh_geom      ON forecast_hexes USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_fh_aqi       ON forecast_hexes (predicted_aqi DESC);
CREATE INDEX IF NOT EXISTS idx_fh_timestamp ON forecast_hexes (future_timestamp);


-- ── 2. Vulnerability Points (OSM-derived) ───────────────────
-- Schools, hospitals, care homes — the targets that receive dispatch alerts.
-- In production: load from Overpass API export or osm2pgsql.
-- Here: seeded with real Pune POI coordinates.
CREATE TABLE IF NOT EXISTS vulnerability_points (
    id            SERIAL       PRIMARY KEY,
    osm_id        BIGINT       UNIQUE,
    name          TEXT         NOT NULL,
    category      TEXT         NOT NULL,  -- school | hospital | care_home | kindergarten
    lat           DOUBLE PRECISION NOT NULL,
    lon           DOUBLE PRECISION NOT NULL,
    geom          GEOMETRY(POINT, 4326),
    h3_index      TEXT,        -- populated post-insert
    phone_number  TEXT,        -- emergency contact (placeholder)
    inserted_at   TIMESTAMPTZ  DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_vp_geom     ON vulnerability_points USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_vp_category ON vulnerability_points (category);
CREATE INDEX IF NOT EXISTS idx_vp_h3       ON vulnerability_points (h3_index);

-- Seed with Pune POIs (real coordinates)
INSERT INTO vulnerability_points (osm_id, name, category, lat, lon, geom, phone_number)
VALUES
    (111001, 'Symbiosis International University Hospital', 'hospital',  18.5272, 73.8143,
        ST_SetSRID(ST_MakePoint(73.8143, 18.5272), 4326), '+912025674000'),
    (111002, 'KEM Hospital Pune',                           'hospital',  18.5195, 73.8553,
        ST_SetSRID(ST_MakePoint(73.8553, 18.5195), 4326), '+912026125600'),
    (111003, 'Jehangir Hospital',                           'hospital',  18.5316, 73.8767,
        ST_SetSRID(ST_MakePoint(73.8767, 18.5316), 4326), '+912026059333'),
    (111004, 'Ruby Hall Clinic',                            'hospital',  18.5362, 73.8867,
        ST_SetSRID(ST_MakePoint(73.8867, 18.5362), 4326), '+912026163391'),
    (111005, 'Sassoon General Hospital',                    'hospital',  18.5204, 73.8559,
        ST_SetSRID(ST_MakePoint(73.8559, 18.5204), 4326), '+912026128000'),
    (111006, 'Bal Shikshan Mandir English Medium School',   'school',    18.5334, 73.8562,
        ST_SetSRID(ST_MakePoint(73.8562, 18.5334), 4326), '+912024440500'),
    (111007, 'Pune Municipal Corporation Primary School',   'school',    18.5180, 73.8490,
        ST_SetSRID(ST_MakePoint(73.8490, 18.5180), 4326), '+912024453300'),
    (111008, 'The Orchid School Baner',                     'school',    18.5601, 73.8114,
        ST_SetSRID(ST_MakePoint(73.8114, 18.5601), 4326), '+912027294700'),
    (111009, 'Abhinava Vidyalaya English Medium School',    'school',    18.5480, 73.8374,
        ST_SetSRID(ST_MakePoint(73.8374, 18.5480), 4326), '+912025536000'),
    (111010, 'Goodluck Children Care Home',                 'care_home', 18.5295, 73.8592,
        ST_SetSRID(ST_MakePoint(73.8592, 18.5295), 4326), '+912024220000'),
    (111011, 'Poona Hospital and Research Centre',          'hospital',  18.5138, 73.8405,
        ST_SetSRID(ST_MakePoint(73.8405, 18.5138), 4326), '+912024330500'),
    (111012, 'Symbiosis School for Liberal Arts',           'school',    18.5263, 73.8117,
        ST_SetSRID(ST_MakePoint(73.8117, 18.5263), 4326), '+912020238777'),
    (111013, 'Deenanath Mangeshkar Hospital',               'hospital',  18.5050, 73.8396,
        ST_SetSRID(ST_MakePoint(73.8396, 18.5050), 4326), '+912024927777'),
    (111014, 'Sunrise Kindergarten Shivajinagar',           'kindergarten', 18.5314, 73.8446,
        ST_SetSRID(ST_MakePoint(73.8446, 18.5314), 4326), '+912020001234')
ON CONFLICT (osm_id) DO NOTHING;


-- ── 3. Dispatch Log ─────────────────────────────────────────
-- Audit trail of every alert sent (call + MQTT).
CREATE TABLE IF NOT EXISTS dispatch_log (
    id                  SERIAL       PRIMARY KEY,
    h3_index            TEXT         NOT NULL,
    vulnerability_id    INT          REFERENCES vulnerability_points(id),
    predicted_aqi       DOUBLE PRECISION,
    horizon_hours       SMALLINT,
    advisory_en         TEXT,
    advisory_hi         TEXT,   -- Hindi (IndicTrans2)
    advisory_mr         TEXT,   -- Marathi (IndicTrans2)
    twilio_call_sid     TEXT,
    twilio_status       TEXT,   -- 'queued' | 'dry_run' | 'failed'
    mqtt_topic          TEXT,
    mqtt_published      BOOLEAN  DEFAULT FALSE,
    dispatched_at       TIMESTAMPTZ DEFAULT NOW(),
    end_to_end_ms       DOUBLE PRECISION  -- pipeline latency
);
CREATE INDEX IF NOT EXISTS idx_dl_h3       ON dispatch_log (h3_index);
CREATE INDEX IF NOT EXISTS idx_dl_time     ON dispatch_log (dispatched_at DESC);
CREATE INDEX IF NOT EXISTS idx_dl_vuln_id  ON dispatch_log (vulnerability_id);


-- ── 4. Spatial query: vulnerabilities inside a given hex ────
-- Standalone query — replicated in spatial_query.py as a Python function.
-- Pass the hex boundary WKT via %s parameter substitution.
--
-- Example usage (psql):
--   SELECT vp.name, vp.category
--   FROM vulnerability_points vp
--   WHERE ST_Intersects(
--       vp.geom,
--       ST_GeomFromText('POLYGON((73.84 18.52, ...))', 4326)
--   );


-- ── 5. Useful diagnostic queries ────────────────────────────

-- Count vulnerable facilities per category
-- SELECT category, count(*) FROM vulnerability_points GROUP BY category;

-- Check all facilities within 1 km of Shivajinagar (hex centroid)
-- SELECT name, category,
--        ST_Distance(geom::geography,
--                    ST_SetSRID(ST_MakePoint(73.8446, 18.5314), 4326)::geography) AS dist_m
-- FROM vulnerability_points
-- WHERE ST_DWithin(geom::geography,
--                  ST_SetSRID(ST_MakePoint(73.8446, 18.5314), 4326)::geography,
--                  1000)
-- ORDER BY dist_m;
