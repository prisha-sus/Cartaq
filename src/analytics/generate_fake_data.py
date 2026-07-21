"""
Role 3: Big Data Analytics & BI Engineer
Generate 5 million rows of synthetic historical data for each of the two
ClickHouse tables (forecast_log, attribution_log), giving the BI layer
real data to benchmark against before the ML models are ready.

Zero-blocker contract for Role 3:
    "Generate 5 million rows of fake historical intervention data and
     spend your week optimizing ClickHouse queries and designing the
     Superset dashboards."

Output:
    data/fake_forecast_5m.parquet      (ingested -> cartaq.forecast_log)
    data/fake_attribution_5m.parquet   (ingested -> cartaq.attribution_log)

Metrics reported:
    - Generation throughput (rows / second) per table
    - Parquet file size on disk

Usage:
    cd Cartaq
    python src/analytics/generate_fake_data.py
"""

import os
import time
from datetime import datetime, timedelta

import h3
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
PUNE_CENTER_LAT = 18.5314
PUNE_CENTER_LON = 73.8446
H3_RESOLUTION   = 8
K_RING_RADIUS   = 25         # ~2000 hexes around Pune - realistic city coverage
TARGET_ROWS     = 5_000_000
RANDOM_SEED     = 42

# Categorical pools
SOURCE_TYPES   = ["factory", "traffic", "construction", "domestic", "transport"]
MODEL_VERSIONS = ["a3tgcn_v1", "a3tgcn_v2", "persistence_baseline"]
CITIES         = ["pune", "mumbai", "delhi", "bangalore", "hyderabad"]
HORIZONS       = [24, 48, 72]

# Date range: 18 months of synthetic history
START_TS = datetime(2025, 1, 1)
END_TS   = datetime(2026, 7, 1)
# ─────────────────────────────────────────────────────────────────────────────


def build_hex_pool(lat: float, lon: float, resolution: int, k: int) -> list[str]:
    """Return all H3 indices within k rings of the center cell."""
    center = h3.latlng_to_cell(lat, lon, resolution)
    return list(h3.grid_disk(center, k))


def _random_timestamps(n: int, rng: np.random.Generator) -> list[datetime]:
    """Sample n uniformly random datetimes between START_TS and END_TS."""
    total_hours = int((END_TS - START_TS).total_seconds() / 3600)
    offsets = rng.integers(0, total_hours, size=n)
    return [START_TS + timedelta(hours=int(h)) for h in offsets]


def generate_forecast_table(
    hex_pool: list[str], n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Synthetic cartaq.forecast_log rows.
    AQI distribution: lognormal centred around moderate-unhealthy range (50-200).
    """
    print(f"  Generating {n:,} forecast rows ({len(hex_pool)} hex pool)...")
    t0 = time.perf_counter()

    predicted_aqi = np.clip(
        rng.lognormal(mean=4.2, sigma=0.55, size=n), 10.0, 500.0
    ).astype(np.float32)

    df = pd.DataFrame({
        "h3_index":         rng.choice(hex_pool, size=n),
        "future_timestamp": _random_timestamps(n, rng),
        "predicted_aqi":    predicted_aqi,
        "horizon_hours":    rng.choice(HORIZONS, size=n).astype(np.uint8),
        "run_timestamp":    pd.Timestamp.utcnow(),
        "city":             rng.choice(CITIES, size=n),
        "model_version":    rng.choice(MODEL_VERSIONS, size=n),
    })
    df["future_timestamp"] = pd.to_datetime(df["future_timestamp"])

    elapsed = time.perf_counter() - t0
    print(f"  Done: {elapsed:.2f}s  |  {n / elapsed:,.0f} rows/s")
    return df


def generate_attribution_table(
    hex_pool: list[str], n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Synthetic cartaq.attribution_log rows.
    ATE (Average Treatment Effect): active sources add 20-80 AQI points,
    with 95% CI width of 10-30 points (realistic DoWhy output range).
    """
    print(f"  Generating {n:,} attribution rows...")
    t0 = time.perf_counter()

    is_active = rng.integers(0, 2, size=n, dtype=np.uint8)
    base_ate = (
        is_active * rng.uniform(20, 80, size=n)
        + rng.normal(0, 5, size=n)
    ).astype(np.float32)
    ate = np.clip(base_ate, -10.0, 150.0)

    ci_half = rng.uniform(5, 15, size=n).astype(np.float32)

    df = pd.DataFrame({
        "h3_index":         rng.choice(hex_pool, size=n),
        "event_timestamp":  _random_timestamps(n, rng),
        "source_type":      rng.choice(SOURCE_TYPES, size=n),
        "ate":              ate,
        "confidence_lower": (ate - ci_half),
        "confidence_upper": (ate + ci_half),
        "wind_speed_kmh":   rng.uniform(0.5, 30.0, size=n).astype(np.float32),
        "is_source_active": is_active,
        "run_timestamp":    pd.Timestamp.utcnow(),
    })
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])

    elapsed = time.perf_counter() - t0
    print(f"  Done: {elapsed:.2f}s  |  {n / elapsed:,.0f} rows/s")
    return df


def save_parquet(df: pd.DataFrame, path: str) -> float:
    """Save df to parquet (snappy) and return file size in MB."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  Saved -> {path}  ({size_mb:.1f} MB)")
    return size_mb


def main() -> None:
    print("=" * 60)
    print("  Cartaq Role 3 - Fake Data Generator")
    print(f"  Target: {TARGET_ROWS:,} rows per table")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"\nBuilding H3 hex pool (res={H3_RESOLUTION}, k={K_RING_RADIUS})...")
    hex_pool = build_hex_pool(PUNE_CENTER_LAT, PUNE_CENTER_LON, H3_RESOLUTION, K_RING_RADIUS)
    print(f"  Hex pool: {len(hex_pool)} cells\n")

    # Table 1: forecast_log
    print("[1/2] forecast_log")
    df_forecast = generate_forecast_table(hex_pool, TARGET_ROWS, rng)
    save_parquet(df_forecast, "data/fake_forecast_5m.parquet")
    del df_forecast  # free memory before generating table 2

    # Table 2: attribution_log
    print("\n[2/2] attribution_log")
    df_attr = generate_attribution_table(hex_pool, TARGET_ROWS, rng)
    save_parquet(df_attr, "data/fake_attribution_5m.parquet")

    print("\nOK - Generation complete.")
    print("   Next step: python src/analytics/ingest_to_clickhouse.py")


if __name__ == "__main__":
    main()
