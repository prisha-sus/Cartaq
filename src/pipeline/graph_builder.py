"""
Role 1 - graph_builder.py (THE MISSING PIECE)
Connects ingestion output -> H3-mapped PyTorch tensors for train.py.

Input:
    data/raw/openaq.parquet       lat, lon, timestamp, value (PM2.5 µg/m³), parameter
    data/raw/openmeteo.parquet    lat, lon, timestamp, temperature_2m,
                                  relative_humidity_2m, wind_speed_10m,
                                  wind_direction_10m, precipitation

Output (data/graph/):
    h3_indices.json    ordered list of H3 cell IDs
    x.pt               [batch_size, num_nodes, 4, 12]  - 4 features, 12-hour lookback
    y.pt               [batch_size, num_nodes, 3]       - PM2.5 at +24h, +48h, +72h
    edge_index.pt      [2, num_edges]
    metadata.json      shapes, feature names, latency metrics

After running this, update train.py to call load_real_graph_data() instead of
load_dummy_graph_data() - see comments in that file.

Usage:
    cd Cartaq
    python src/pipeline/graph_builder.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.h3_mapper import H3Mapper

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES          = ["pm25", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
LOOKBACK_HOURS    = 12
FORECAST_HORIZONS = [24, 48, 72]   # hours ahead; must match train.py's 3 horizons
H3_RESOLUTION     = 8
MIN_HEXES         = 10


def _load_openaq(path: str = "data/raw/openaq.parquet") -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    df = df[df.get("parameter", pd.Series(["pm25"] * len(df))) == "pm25"].copy()
    df = df.rename(columns={"value": "pm25"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("H")
    return df[["lat", "lon", "timestamp", "pm25"]].dropna()


def _load_openmeteo(path: str = "data/raw/openmeteo.parquet") -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("H")
    return df.dropna(subset=["lat", "lon", "timestamp"])


def _merge_to_hex_features(
    openaq_df: pd.DataFrame | None,
    openmeteo_df: pd.DataFrame | None,
    mapper: H3Mapper,
) -> pd.DataFrame:
    """Map both sources to H3 cells and merge into one hourly feature table."""
    frames = {}

    if openaq_df is not None and not openaq_df.empty:
        aq = mapper.add_h3_indices(openaq_df.copy())
        aq = (
            aq.groupby(["h3_index", "timestamp"])["pm25"]
            .median()
            .reset_index()
            .set_index(["h3_index", "timestamp"])
        )
        frames["openaq"] = aq

    if openmeteo_df is not None and not openmeteo_df.empty:
        met = mapper.add_h3_indices(openmeteo_df.copy())
        w_cols = [c for c in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
                  if c in met.columns]
        if w_cols:
            met_agg = (
                met.groupby(["h3_index", "timestamp"])[w_cols]
                .mean()
                .reset_index()
                .set_index(["h3_index", "timestamp"])
            )
            frames["openmeteo"] = met_agg

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return list(frames.values())[0].reset_index()
    merged = list(frames.values())[0].join(list(frames.values())[1], how="outer")
    return merged.reset_index()


def _build_tensors_from_df(
    feature_df: pd.DataFrame,
    mapper: H3Mapper,
    lookback: int = LOOKBACK_HOURS,
    horizons: list[int] = FORECAST_HORIZONS,
    output_dir: str = "data/graph",
) -> dict:
    """
    Slide a window over the time dimension to create batched (x, y) pairs.
    x: [batch, n_nodes, n_features, lookback]
    y: [batch, n_nodes, 3]  - PM2.5 at each horizon
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.perf_counter()

    # Ensure all feature columns exist
    for col in FEATURES:
        if col not in feature_df.columns:
            feature_df[col] = np.nan

    h3_indices = sorted(feature_df["h3_index"].dropna().unique().tolist())
    timestamps = sorted(feature_df["timestamp"].unique().tolist())

    if len(h3_indices) < MIN_HEXES:
        print(f"  Only {len(h3_indices)} hexes - falling back to synthetic tensors.")
        return _synthetic_tensors(output_dir)

    n_nodes = len(h3_indices)
    n_times = len(timestamps)
    idx_map = {h: i for i, h in enumerate(h3_indices)}
    ts_map  = {t: i for i, t in enumerate(timestamps)}

    # Build full feature cube [n_nodes, n_features, n_times]
    cube = np.zeros((n_nodes, len(FEATURES), n_times), dtype=np.float32)
    for _, row in feature_df.iterrows():
        ni = idx_map.get(row["h3_index"])
        ti = ts_map.get(row.get("timestamp"))
        if ni is None or ti is None:
            continue
        for fi, col in enumerate(FEATURES):
            v = row.get(col)
            if v is not None and not np.isnan(float(v)):
                cube[ni, fi, ti] = float(v)

    # Per-feature min-max normalisation
    for fi in range(len(FEATURES)):
        mn, mx = cube[:, fi, :].min(), cube[:, fi, :].max()
        if mx > mn:
            cube[:, fi, :] = (cube[:, fi, :] - mn) / (mx - mn)

    # Sliding window to create samples
    max_h = max(horizons)
    xs, ys = [], []
    for t in range(n_times - lookback - max_h):
        xs.append(cube[:, :, t : t + lookback])
        targets = [cube[:, 0, t + lookback + h - 1] for h in horizons]
        ys.append(np.stack(targets, axis=-1))

    if not xs:
        print("  Not enough time steps for windowing - falling back to synthetic tensors.")
        return _synthetic_tensors(output_dir)

    x_t = torch.tensor(np.stack(xs), dtype=torch.float32)   # [B, N, F, T]
    y_t = torch.tensor(np.stack(ys), dtype=torch.float32)   # [B, N, 3]
    ei  = torch.tensor(mapper.build_adjacency_matrix(h3_indices), dtype=torch.long)

    torch.save(x_t, os.path.join(output_dir, "x.pt"))
    torch.save(y_t, os.path.join(output_dir, "y.pt"))
    torch.save(ei,  os.path.join(output_dir, "edge_index.pt"))
    with open(os.path.join(output_dir, "h3_indices.json"), "w") as f:
        json.dump(h3_indices, f)

    latency_ms = (time.perf_counter() - t0) * 1000
    meta = {
        "n_nodes": n_nodes, "n_features": len(FEATURES),
        "lookback_hours": lookback, "forecast_horizons": horizons,
        "batch_size": len(xs), "features": FEATURES,
        "h3_resolution": H3_RESOLUTION,
        "x_shape": list(x_t.shape), "y_shape": list(y_t.shape),
        "edge_index_shape": list(ei.shape),
        "construction_latency_ms": round(latency_ms, 1),
        "synthetic": False,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  x.pt           {list(x_t.shape)}")
    print(f"  y.pt           {list(y_t.shape)}")
    print(f"  edge_index.pt  {list(ei.shape)}")
    print(f"  Latency: {latency_ms:.1f} ms")
    return meta


def _generate_realistic_synthetic(
    output_dir: str = "data/graph",
    seed: int = 42,
) -> dict:
    """
    Generate synthetic graph tensors with realistic spatial & temporal structure.

    Features:
        pm25:    diurnal cycle + spatial gradient (center high, outskirts low)
                 + industrial hotspots + auto-regressive noise
        temperature_2m:       diurnal (peak 2pm, min 5am) + spatial noise
        relative_humidity_2m: inverse temp + noise
        wind_speed_10m:       Weibull-distributed, slight diurnal pattern

    Targets (pm25 at +24h / +48h / +72h):
        Each horizon = current pm25 * decay_factor + synoptic drift + noise

    Edge weights: wind-directed from compute_wind_edge_weights (west->east).
    """
    import h3 as h3lib
    import numpy as np
    from model.physics_loss import compute_wind_edge_weights

    rng = np.random.default_rng(seed)

    # H3 hex layout
    center = h3lib.latlng_to_cell(18.5314, 73.8446, H3_RESOLUTION)
    hexes  = list(h3lib.grid_disk(center, 4))[:50]
    num_nodes = len(hexes)
    n_features = len(FEATURES)
    T = LOOKBACK_HOURS
    n_horizons = len(FORECAST_HORIZONS)
    batch_size = 64
    n_timesteps = T + max(FORECAST_HORIZONS) + batch_size + 5  # enough history + horizons

    # Centre-of-mass lat/lon for spatial gradient
    hex_lats  = np.array([h3lib.cell_to_latlng(h)[0] for h in hexes])
    hex_lons  = np.array([h3lib.cell_to_latlng(h)[1] for h in hexes])
    centre_lat, centre_lon = 18.5314, 73.8446
    dist_from_centre = np.sqrt((hex_lats - centre_lat)**2 + (hex_lons - centre_lon)**2)
    # Normalise distance to [0, 1]
    dist_norm = dist_from_centre / max(dist_from_centre.max(), 1e-6)

    # Industrial hotspots (2-3 random nodes with elevated base pollution)
    hotspots = rng.choice(num_nodes, size=rng.integers(2, 4), replace=False)
    hotspot_boost = np.zeros(num_nodes)
    hotspot_boost[hotspots] = rng.uniform(30, 60, size=len(hotspots))

    # Base PM2.5 per node (spatial pattern)
    base_pm25 = 140.0 - 80.0 * dist_norm + hotspot_boost  # centre ~140, outskirts ~60
    base_pm25 = np.clip(base_pm25, 30, 250)

    # Build full-feature time-series cube: [n_nodes, n_features, n_timesteps]
    cube = np.zeros((num_nodes, n_features, n_timesteps), dtype=np.float32)

    for t in range(n_timesteps):
        hour_of_day = t % 24
        # Diurnal PM2.5: peaks at 9am and 8pm, trough at 3am
        diurnal_pm25 = 1.0 + 0.35 * np.sin(np.pi * (hour_of_day - 3) / 12) ** 4 \
                       - 0.15 * np.sin(np.pi * (hour_of_day - 14) / 8)
        diurnal_pm25 = np.clip(diurnal_pm25, 0.5, 1.5)

        # Temperature diurnal: peak ~35C at 14h, min ~22C at 5h
        temp = 28.5 + 6.5 * np.sin(np.pi * (hour_of_day - 5) / 12)
        # Humidity inverse to temp
        humid = 75 - 20 * np.sin(np.pi * (hour_of_day - 5) / 12) + rng.normal(0, 5)
        humid = np.clip(humid, 30, 100)
        # Wind: Weibull-like with slight daytime increase
        wind = rng.weibull(1.8) * 5 + 2.0 + 2.0 * np.sin(np.pi * hour_of_day / 24)
        wind = np.clip(wind, 0.5, 25)

        for ni in range(num_nodes):
            # PM2.5: base * diurnal + auto-regressive + noise
            if t == 0:
                pm25 = base_pm25[ni] * diurnal_pm25 + rng.normal(0, 5)
            else:
                prev = cube[ni, 0, t-1]
                ar = 0.6 * prev + 0.15 * cube[ni, 0, max(0, t-2)]
                local_source = rng.normal(0, 3) + (hotspot_boost[ni] * 0.1 * np.sin(np.pi * hour_of_day / 12))
                pm25 = ar + local_source + base_pm25[ni] * (diurnal_pm25 - 1.0) * 0.5

            pm25 = np.clip(pm25, 20, 500)
            cube[ni, 0, t] = round(float(pm25), 2)  # pm25
            cube[ni, 1, t] = round(float(temp + rng.normal(0, 0.5)), 2)   # temperature
            cube[ni, 2, t] = round(float(humid + rng.normal(0, 2)), 2)     # humidity
            wind_v = max(0.5, wind + rng.normal(0, 0.3))
            cube[ni, 3, t] = round(float(wind_v), 2)                        # wind speed

    # Capture raw PM2.5 range before normalisation
    pm25_raw = cube[:, 0, :].copy()
    pm25_min = float(pm25_raw.min())
    pm25_max = float(pm25_raw.max())

    # Per-feature min-max normalisation (same as real data path)
    for fi in range(n_features):
        mn, mx = cube[:, fi, :].min(), cube[:, fi, :].max()
        if mx > mn:
            cube[:, fi, :] = (cube[:, fi, :] - mn) / (mx - mn)

    # Sliding window to create x/y pairs
    max_h = max(FORECAST_HORIZONS)
    xs, ys = [], []
    for t in range(n_timesteps - T - max_h):
        xs.append(cube[:, :, t : t + T])
        targets = [cube[:, 0, t + T + h - 1] for h in FORECAST_HORIZONS]
        ys.append(np.stack(targets, axis=-1))

    # Take first batch_size samples
    xs = np.stack(xs[:batch_size])
    ys = np.stack(ys[:batch_size])
    B_actual = xs.shape[0]

    # Edge index from H3 adjacency
    mapper = H3Mapper(H3_RESOLUTION)
    ei_np = mapper.build_adjacency_matrix(hexes)  # [2, E]

    # Wind-weighted edge weights (west->east wind at 15 km/h)
    wind_dir_deg = 270.0
    wind_speed_kmh = 15.0
    ei_wind, ew_wind = compute_wind_edge_weights(hexes, wind_dir_deg, wind_speed_kmh)

    # Use wind edge_index if it has edges, else fallback to unweighted adjacency
    if ei_wind.shape[1] > 0:
        edge_index = torch.tensor(ei_wind, dtype=torch.long)
        edge_weight = torch.tensor(ew_wind, dtype=torch.float32)
    else:
        edge_index = torch.tensor(ei_np, dtype=torch.long)
        edge_weight = None

    x_t = torch.tensor(xs, dtype=torch.float32)
    y_t = torch.tensor(ys, dtype=torch.float32)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(x_t, os.path.join(output_dir, "x.pt"))
    torch.save(y_t, os.path.join(output_dir, "y.pt"))
    torch.save(edge_index, os.path.join(output_dir, "edge_index.pt"))
    if edge_weight is not None:
        torch.save(edge_weight, os.path.join(output_dir, "edge_weight.pt"))
    with open(os.path.join(output_dir, "h3_indices.json"), "w") as f:
        json.dump(hexes, f)

    meta = {
        "n_nodes": num_nodes, "n_features": n_features,
        "lookback_hours": T, "forecast_horizons": FORECAST_HORIZONS,
        "batch_size": B_actual, "features": FEATURES,
        "h3_resolution": H3_RESOLUTION,
        "x_shape": list(x_t.shape), "y_shape": list(y_t.shape),
        "edge_index_shape": list(edge_index.shape),
        "synthetic": True, "realistic_synthetic": True,
        "wind_dir_deg": wind_dir_deg, "wind_speed_kmh": wind_speed_kmh,
        "pm25_min": round(pm25_min, 2),
        "pm25_max": round(pm25_max, 2),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Generated realistic synthetic data:")
    print(f"    x.pt           {list(x_t.shape)}")
    print(f"    y.pt           {list(y_t.shape)}")
    print(f"    edge_index.pt  {list(edge_index.shape)}")
    print(f"    edge_weight.pt {'saved' if edge_weight is not None else 'none'}")
    return meta


def _synthetic_tensors(output_dir: str = "data/graph") -> dict:
    """Fallback: delegates to realistic synthetic generator."""
    return _generate_realistic_synthetic(output_dir)


def load_real_graph_data(graph_dir: str = "data/graph"):
    """
    Drop-in replacement for train.py's load_dummy_graph_data().
    Returns (x, y, edge_index) tensors loaded from disk.
    """
    x          = torch.load(os.path.join(graph_dir, "x.pt"))
    y          = torch.load(os.path.join(graph_dir, "y.pt"))
    edge_index = torch.load(os.path.join(graph_dir, "edge_index.pt"))
    return x, y, edge_index


def main():
    print("=" * 60)
    print("  Cartaq Role 1 - Graph Builder")
    print("=" * 60)
    mapper = H3Mapper(H3_RESOLUTION)

    print("\n[1/3] Loading raw ingestion data...")
    aq  = _load_openaq()
    met = _load_openmeteo()

    if aq is None and met is None:
        print("  No raw data in data/raw/ - generating synthetic tensors.")
        print("  Run src/ingestion/openaq_client.py and openmeteo_client.py to ingest real data.")
        meta = _synthetic_tensors()
    else:
        print(f"  OpenAQ rows:    {len(aq) if aq is not None else 'N/A'}")
        print(f"  Open-Meteo rows: {len(met) if met is not None else 'N/A'}")

        print("\n[2/3] Mapping to H3 and building feature table...")
        feat_df = _merge_to_hex_features(aq, met, mapper)
        if feat_df.empty:
            print("  Empty feature table - falling back to synthetic tensors.")
            meta = _synthetic_tensors()
        else:
            print(f"  Rows: {len(feat_df):,}  |  Hexes: {feat_df.h3_index.nunique()}")
            print("\n[3/3] Building sliding-window tensors...")
            meta = _build_tensors_from_df(feat_df, mapper)

    print("\nOK - data/graph/ ready.")
    print("   To use real data in training, replace load_dummy_graph_data() in")
    print("   src/train.py with: from pipeline.graph_builder import load_real_graph_data")
    return meta


if __name__ == "__main__":
    main()
