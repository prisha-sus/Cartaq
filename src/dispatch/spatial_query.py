"""
Role 4: Agentic NLP & Systems Integration Engineer
PostGIS spatial query engine.

Maps each high-AQI H3 hex from the ST-GNN forecast to vulnerable facilities
(schools, hospitals, care homes) via a PostGIS ST_Intersects query.

Consumes:
    src/dispatch/mock_payload.json  (or output/forecast.parquet from Role 1)

Produces:
    A list of dispatch targets: (facility, h3_index, predicted_aqi) triples

Metrics tracked:
    - Spatial query execution time per hex (reported individually)
    - Total spatial query latency across all events

Usage:
    cd Cartaq
    python src/dispatch/spatial_query.py

Requires:
    pip install psycopg2-binary h3
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import h3

# ── Config ────────────────────────────────────────────────────────────────────
PG_DSN = os.getenv(
    "POSTGIS_DSN",
    "host=localhost port=5432 dbname=cartaq user=cartaq_user password=cartaq_pass",
)
AQI_ALERT_THRESHOLD = float(os.getenv("AQI_ALERT_THRESHOLD", "150"))
H3_RESOLUTION       = 8
# ─────────────────────────────────────────────────────────────────────────────


def load_high_aqi_events(
    mock_path: str = "",
) -> list[dict[str, Any]]:
    if not mock_path:
        mock_path = str(Path(__file__).parent / "mock_payload.json")
    """
    Load high-pollution events.

    In production: replace the mock JSON with:
        import pandas as pd
        df = pd.read_parquet("output/forecast.parquet")
        events = df[df.predicted_aqi >= AQI_ALERT_THRESHOLD].to_dict("records")
    """
    path = Path(mock_path)
    if not path.exists():
        raise FileNotFoundError(f"Mock payload not found: {path}")

    with open(path) as f:
        payload = json.load(f)

    events = [
        e for e in payload["high_pollution_events"]
        if e["predicted_aqi"] >= AQI_ALERT_THRESHOLD
    ]
    print(f"Loaded {len(events)} high-AQI events (>={AQI_ALERT_THRESHOLD} AQI)")
    return events


def hex_to_wkt_polygon(h3_index: str) -> str:
    """
    Convert an H3 cell to a WKT POLYGON string for PostGIS.
    Uses h3.cell_to_boundary() which returns (lat, lon) pairs;
    WKT requires (lon lat) ordering.
    """
    boundary = h3.cell_to_boundary(h3_index)
    # h3 returns [(lat, lon), ...]; WKT wants (lon lat)
    coords = [(lon, lat) for lat, lon in boundary]
    coords.append(coords[0])  # close the ring
    coord_str = ", ".join(f"{lon:.6f} {lat:.6f}" for lon, lat in coords)
    return f"POLYGON(({coord_str}))"


def query_vulnerabilities_in_hex(
    conn, h3_index: str
) -> list[dict[str, Any]]:
    """
    PostGIS ST_Intersects query: find all vulnerability_points inside an H3 hex.

    Returns a list of vulnerability dicts with spatial metadata.
    """
    import psycopg2.extras

    wkt = hex_to_wkt_polygon(h3_index)

    sql = """
        SELECT
            vp.id,
            vp.name,
            vp.category,
            vp.lat,
            vp.lon,
            vp.phone_number,
            round(
                ST_Distance(
                    vp.geom::geography,
                    ST_Centroid(ST_GeomFromText(%s, 4326))::geography
                )::numeric,
                1
            ) AS dist_from_hex_center_m
        FROM vulnerability_points vp
        WHERE ST_Intersects(
            vp.geom,
            ST_GeomFromText(%s, 4326)
        )
        ORDER BY dist_from_hex_center_m ASC;
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (wkt, wkt))
        return [dict(row) for row in cur.fetchall()]


def upsert_forecast_hex(conn, event: dict[str, Any]) -> None:
    """
    Insert the high-AQI hex (with its boundary geometry) into forecast_hexes.
    This makes the hex geometry available for future spatial joins.
    """
    wkt = hex_to_wkt_polygon(event["h3_index"])
    sql = """
        INSERT INTO forecast_hexes
            (h3_index, future_timestamp, predicted_aqi, horizon_hours, aqi_category, geom)
        VALUES (%s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326))
        ON CONFLICT (h3_index, future_timestamp) DO UPDATE
            SET predicted_aqi = EXCLUDED.predicted_aqi,
                aqi_category  = EXCLUDED.aqi_category,
                geom          = EXCLUDED.geom;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            event["h3_index"],
            event["future_timestamp"],
            event["predicted_aqi"],
            event.get("horizon_hours", 24),
            event.get("aqi_category", "Unhealthy"),
            wkt,
        ))
    conn.commit()


def run_spatial_queries(
    events: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    For each high-AQI event, run a PostGIS ST_Intersects query.
    Falls back to mock results if PostGIS is unavailable.

    Returns:
        dispatch_targets: flat list of (facility + event) dicts
        latency_stats: {"total_ms", "per_hex_avg_ms", "per_hex_max_ms"}
    """
    try:
        import psycopg2
        conn = psycopg2.connect(PG_DSN)
        print("  PostGIS connection OK")
        use_mock = False
    except Exception as e:
        print(f"  WARNING: PostGIS unavailable ({e}). Using mock spatial results.")
        use_mock = True

    dispatch_targets: list[dict] = []
    per_hex_latencies: list[float] = []

    for event in events:
        h3_idx = event["h3_index"]
        aqi    = event["predicted_aqi"]

        t0 = time.perf_counter()

        if use_mock:
            vulnerabilities = _mock_facilities(h3_idx)
        else:
            upsert_forecast_hex(conn, event)
            vulnerabilities = query_vulnerabilities_in_hex(conn, h3_idx)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_hex_latencies.append(elapsed_ms)

        mode_tag = "[MOCK]" if use_mock else "[PostGIS]"
        print(
            f"  {mode_tag} Hex {h3_idx}  AQI={aqi:.1f}  "
            f"-> {len(vulnerabilities)} facilities  ({elapsed_ms:.1f}ms)"
        )

        for vuln in vulnerabilities:
            dispatch_targets.append({
                **event,
                "vulnerability_id":      vuln.get("id"),
                "facility_name":         vuln["name"],
                "facility_category":     vuln["category"],
                "lat":                   vuln["lat"],
                "lon":                   vuln["lon"],
                "phone_number":          vuln.get("phone_number", ""),
                "dist_from_hex_center_m": vuln.get("dist_from_hex_center_m", 0),
            })

    if not use_mock:
        conn.close()

    latency_stats = {
        "total_ms":       round(sum(per_hex_latencies), 1),
        "per_hex_avg_ms": round(sum(per_hex_latencies) / len(per_hex_latencies), 1) if per_hex_latencies else 0,
        "per_hex_max_ms": round(max(per_hex_latencies), 1) if per_hex_latencies else 0,
    }

    return dispatch_targets, latency_stats


def _mock_facilities(h3_index: str) -> list[dict[str, Any]]:
    """
    Fallback mock vulnerability data when PostGIS is unavailable.
    Returns 2 representative facilities for every hex to allow full
    pipeline testing without a running database.
    """
    return [
        {
            "id": 2,
            "name": "KEM Hospital Pune",
            "category": "hospital",
            "lat": 18.5195,
            "lon": 73.8553,
            "phone_number": "+912026125600",
            "dist_from_hex_center_m": 320.0,
        },
        {
            "id": 6,
            "name": "Bal Shikshan Mandir English Medium School",
            "category": "school",
            "lat": 18.5334,
            "lon": 73.8562,
            "phone_number": "+912024440500",
            "dist_from_hex_center_m": 480.0,
        },
    ]


if __name__ == "__main__":
    events = load_high_aqi_events()
    targets, stats = run_spatial_queries(events)

    print(f"\nTotal dispatch targets: {len(targets)}")
    print(f"Spatial query latency:  total={stats['total_ms']}ms  "
          f"avg={stats['per_hex_avg_ms']}ms  max={stats['per_hex_max_ms']}ms")

    for t in targets:
        print(
            f"  [{t['facility_category'].upper():<12}] "
            f"{t['facility_name']:<45} "
            f"AQI={t['predicted_aqi']:.1f}  hex={t['h3_index']}"
        )
