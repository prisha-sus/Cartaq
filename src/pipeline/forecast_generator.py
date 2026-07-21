"""
Forecast Generator - uses trained A3TGCN model to produce AQI forecasts.

Takes the graph tensors (x.pt, edge_index.pt, edge_weight.pt, h3_indices.json)
and runs inference through the trained model to produce a forecast DataFrame
saved to data/forecast.parquet.

Usage:
    python src/pipeline/forecast_generator.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.a3tgcn import A3TGCNModel
from model.physics_loss import compute_wind_edge_weights

CHECKPOINT_PATH = Path("output/a3tgcn_checkpoint.pt")
GRAPH_DIR       = Path("data/graph")
OUTPUT_PATH     = "data/forecast.parquet"

WIND_DIR_DEG   = 270.0
WIND_SPEED_KMH = 15.0
FORECAST_HORIZONS = [24, 48, 72]


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def generate_forecast() -> pd.DataFrame:
    print("=" * 60)
    print("  Cartaq - Forecast Generator")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    t0 = time.perf_counter()

    # ── Load checkpoint ───────────────────────────────────────
    print("\n[1/4] Loading model checkpoint...")
    ckpt = load_checkpoint(CHECKPOINT_PATH)
    if ckpt is not None:
        print(f"  Loaded checkpoint from {CHECKPOINT_PATH} "
              f"(epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?'):.4f})")
        node_features  = ckpt.get("node_features", 4)
        periods        = ckpt.get("periods", 12)
        hidden_channels = ckpt.get("hidden_channels", 64)
    else:
        print(f"  No checkpoint found at {CHECKPOINT_PATH}")
        print("  Run `python src/train.py` first, or use demo mode.")
        return _demo_forecast_fallback()

    # ── Load graph data ───────────────────────────────────────
    print("\n[2/4] Loading graph tensors...")
    if not (GRAPH_DIR / "x.pt").exists():
        print(f"  Graph tensors not found in {GRAPH_DIR}")
        print("  Run `python src/pipeline/graph_builder.py` first, or use demo mode.")
        return _demo_forecast_fallback()

    x          = torch.load(GRAPH_DIR / "x.pt", map_location="cpu", weights_only=False)
    y          = torch.load(GRAPH_DIR / "y.pt", map_location="cpu", weights_only=False)
    edge_index = torch.load(GRAPH_DIR / "edge_index.pt", map_location="cpu", weights_only=False)

    # Compute train split (same ratio as train.py)
    B_total = x.size(0)
    train_end = int(0.70 * B_total)

    # Load or compute wind edge weights
    edge_weight = None
    if (GRAPH_DIR / "edge_weight.pt").exists():
        edge_weight = torch.load(GRAPH_DIR / "edge_weight.pt", map_location="cpu", weights_only=False)
    elif (GRAPH_DIR / "h3_indices.json").exists():
        with open(GRAPH_DIR / "h3_indices.json") as f:
            h3_indices = json.load(f)
        ei_np, ew_np = compute_wind_edge_weights(h3_indices, WIND_DIR_DEG, WIND_SPEED_KMH)
        if ei_np.shape[1] > 0:
            edge_index  = torch.tensor(ei_np, dtype=torch.long)
            edge_weight = torch.tensor(ew_np, dtype=torch.float32)

    # Load H3 indices
    h3_path = GRAPH_DIR / "h3_indices.json"
    h3_indices = json.loads(h3_path.read_text()) if h3_path.exists() else []
    num_nodes = x.size(1)

    with open(GRAPH_DIR / "metadata.json") as f:
        meta = json.load(f)
    print(f"  Loaded: x={list(x.shape)}  edge_index={list(edge_index.shape)}  "
          f"{len(h3_indices)} hexes")

    # ── Build model ───────────────────────────────────────────
    print("\n[3/4] Building model & running inference...")
    model = A3TGCNModel(
        node_features=node_features,
        periods=periods,
        hidden_channels=hidden_channels,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    x          = x.to(device)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        predictions = model(x, edge_index, edge_weight)  # [B, N, 3]
        predictions = predictions.cpu().numpy()
        # Get training targets for calibration
        y_train = y.cpu().numpy()[:train_end]

    # Average over batch dimension to get one forecast per node
    pred_mean = predictions.mean(axis=0)  # [N, 3]
    print(f"  Inference complete: {predictions.shape[0]} samples averaged, "
          f"{num_nodes} nodes x {len(FORECAST_HORIZONS)} horizons")

    # Denormalize predictions and targets
    pm25_min = meta.get("pm25_min", 0)
    pm25_max = meta.get("pm25_max", 500)
    pred_denorm = pred_mean * (pm25_max - pm25_min) + pm25_min

    # Calibrate: denormalize y targets then match statistics
    if y_train is not None and y_train.size > 0:
        y_denorm = y_train * (pm25_max - pm25_min) + pm25_min
        t_mean = y_denorm.mean()
        t_std  = y_denorm.std() + 1e-6
        p_mean = pred_denorm.mean()
        p_std  = pred_denorm.std() + 1e-6
        pred_denorm = (pred_denorm - p_mean) / p_std * t_std + t_mean
        t_range = y_denorm.max() - y_denorm.min()
        p_range = pred_denorm.max() - pred_denorm.min()
        print(f"  Calibrated: target mean={t_mean:.1f} std={t_std:.1f} range={t_range:.1f}, "
              f"pred range={p_range:.1f}")

    pred_denorm = np.clip(pred_denorm, 20, 500)

    # ── Build forecast DataFrame ──────────────────────────────
    print("\n[4/4] Writing forecast parquet...")
    base_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    rows = []
    for ni, h3_idx in enumerate(h3_indices):
        for hi, horizon in enumerate(FORECAST_HORIZONS):
            aqi = float(round(pred_denorm[ni, hi], 2))
            ts  = base_time + timedelta(hours=horizon)
            rows.append({
                "h3_index": h3_idx,
                "future_timestamp": ts,
                "predicted_aqi": aqi,
                "horizon_hours": horizon,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="snappy", index=False)

    elapsed = time.perf_counter() - t0
    print(f"  Saved {len(df):,} forecast rows -> {OUTPUT_PATH}")
    print(f"  AQI range: [{df.predicted_aqi.min():.1f}, {df.predicted_aqi.max():.1f}]")
    print(f"  Mean AQI:  {df.predicted_aqi.mean():.1f}")
    print(f"  Total time: {elapsed:.1f}s")
    return df


def _demo_forecast_fallback() -> pd.DataFrame:
    """Generate a demo forecast when no model checkpoint is available."""
    print("\n  -> Using demo mode with realistic synthetic data.")
    import h3 as h3lib
    rng = np.random.default_rng(42)

    center = h3lib.latlng_to_cell(18.5314, 73.8446, 8)
    hexes  = list(h3lib.grid_disk(center, 11))
    hex_lats = np.array([h3lib.cell_to_latlng(h)[0] for h in hexes])
    hex_lons = np.array([h3lib.cell_to_latlng(h)[1] for h in hexes])
    dist = np.sqrt((hex_lats - 18.5314)**2 + (hex_lons - 73.8446)**2)
    dist_norm = dist / dist.max()

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    rows = []
    for hx, d in zip(hexes, dist_norm):
        base_aqi = 160 - 100 * d  # centre ~160, outskirts ~60
        for hi, horizon in enumerate(FORECAST_HORIZONS):
            hour = (now + timedelta(hours=horizon)).hour
            diurnal = 1.0 + 0.3 * np.sin(np.pi * (hour - 3) / 12) ** 4
            noise = rng.normal(0, 8)
            aqi = float(np.clip(base_aqi * diurnal + noise, 25, 450))
            rows.append({
                "h3_index": hx,
                "future_timestamp": now + timedelta(hours=horizon),
                "predicted_aqi": round(aqi, 1),
                "horizon_hours": horizon,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="snappy", index=False)
    print(f"  Saved {len(df):,} demo forecast rows -> {OUTPUT_PATH}")
    print(f"  AQI range: [{df.predicted_aqi.min():.1f}, {df.predicted_aqi.max():.1f}]")
    print(f"  Mean AQI:  {df.predicted_aqi.mean():.1f}")
    return df


if __name__ == "__main__":
    generate_forecast()
