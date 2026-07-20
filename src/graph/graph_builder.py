import sys
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_DAYS_BACK
from graph.h3_mapper import H3Mapper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENMETEO_PATH = PROJECT_ROOT / "data/raw/openmeteo.parquet"
DEFAULT_OPENAQ_PATH = PROJECT_ROOT / "data/raw/openaq.parquet"

FEATURE_COLUMNS = [
    "aqi",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]
LOOKBACK_HOURS = 12
HORIZONS = [24, 48, 72]


def pm25_to_aqi(pm25: np.ndarray) -> np.ndarray:
    """Convert PM2.5 (µg/m³) to US EPA AQI."""
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    aqi = np.zeros_like(pm25, dtype=float)
    for c_low, c_high, i_low, i_high in breakpoints:
        mask = (pm25 >= c_low) & (pm25 <= c_high)
        aqi[mask] = ((i_high - i_low) / (c_high - c_low)) * (pm25[mask] - c_low) + i_low
    aqi[pm25 > 500.4] = 500.0
    return aqi


def estimate_pm25_from_met(df: pd.DataFrame) -> pd.Series:
    """
    Estimate PM2.5 from meteorology when ground-truth AQI is unavailable.
    Uses dispersion proxies: stagnant/low-wind and dry conditions raise PM2.5.
    """
    pm25 = (
        20.0
        + 0.5 * (30.0 - df["wind_speed_10m"].fillna(10.0))
        + 0.15 * (df["relative_humidity_2m"].fillna(50.0) - 50.0)
        - 2.0 * df["precipitation"].fillna(0.0)
        + 0.08 * (df["temperature_2m"].fillna(25.0) - 25.0)
    )
    return pm25.clip(lower=5.0, upper=250.0)


def load_openmeteo(path: Path = DEFAULT_OPENMETEO_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Open-Meteo data not found at {path}")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_openaq(path: Path = DEFAULT_OPENAQ_PATH, met_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if met_df is not None and not met_df.empty:
        start = met_df["timestamp"].min()
        end = met_df["timestamp"].max()
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    else:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=DEFAULT_DAYS_BACK)
        df = df[df["timestamp"] >= cutoff]

    return df.sort_values("timestamp").reset_index(drop=True)


def build_hourly_aqi_table(
    met_df: pd.DataFrame,
    aq_df: pd.DataFrame,
    resolution: int = 8,
) -> tuple[pd.DataFrame, list[str], str]:
    mapper = H3Mapper(resolution=resolution)

    met_hourly = (
        met_df.groupby(pd.Grouper(key="timestamp", freq="h"), observed=True)[FEATURE_COLUMNS[1:]]
        .mean()
        .reset_index()
    )

    center_lat = float(met_df["lat"].iloc[0])
    center_lon = float(met_df["lon"].iloc[0])
    center_h3 = h3.latlng_to_cell(center_lat, center_lon, resolution)
    h3_indices = list(h3.grid_disk(center_h3, 3))
    
    met_df = met_df.copy()
    met_df["pm25"] = estimate_pm25_from_met(met_df)
    met_df["aqi"] = pm25_to_aqi(met_df["pm25"].to_numpy())
    
    aq_h3_indices = set()
    aq_by_h3 = {}
    source = "openmeteo_proxy"
    
    if not aq_df.empty and "value" in aq_df.columns:
        aq_df = mapper.add_h3_indices(aq_df)
        aq_df = aq_df.dropna(subset=["h3_index", "value"])

        overlap_start = max(aq_df["timestamp"].min(), met_hourly["timestamp"].min())
        overlap_end = min(aq_df["timestamp"].max(), met_hourly["timestamp"].max())
        if overlap_start >= overlap_end:
            raise ValueError(
                "OpenAQ and Open-Meteo timestamps do not overlap. "
                "Re-pull data for the same city and date range."
            )

        aq_df = aq_df[
            (aq_df["timestamp"] >= overlap_start) & (aq_df["timestamp"] <= overlap_end)
        ]
        met_hourly = met_hourly[
            (met_hourly["timestamp"] >= overlap_start) & (met_hourly["timestamp"] <= overlap_end)
        ]

        hourly = (
            aq_df.groupby(["h3_index", pd.Grouper(key="timestamp", freq="h")], observed=True)["value"]
            .mean()
            .reset_index()
            .rename(columns={"value": "pm25"})
        )
        hourly["aqi"] = pm25_to_aqi(hourly["pm25"].to_numpy())
        aq_h3_indices = set(hourly["h3_index"].unique())
        
        for h3_idx in aq_h3_indices:
            node_aq = hourly[hourly["h3_index"] == h3_idx].set_index("timestamp")[["pm25", "aqi"]]
            aq_by_h3[h3_idx] = node_aq
        
        source = "hybrid"
    
    met_indexed = met_hourly.set_index("timestamp")
    rows = []
    for h3_idx in h3_indices:
        if h3_idx in aq_by_h3:
            node_aq = aq_by_h3[h3_idx]
            node_df = met_indexed.join(node_aq, how="outer").sort_index()
            node_df[["pm25", "aqi"]] = node_df[["pm25", "aqi"]].interpolate(method="time").ffill().bfill()
        else:
            node_df = met_indexed.copy()
            node_df["pm25"] = met_df.set_index("timestamp")["pm25"].reindex(node_df.index)
            node_df["aqi"] = met_df.set_index("timestamp")["aqi"].reindex(node_df.index)
        
        node_df[FEATURE_COLUMNS[1:]] = node_df[FEATURE_COLUMNS[1:]].interpolate(method="time").ffill().bfill()
        node_df = node_df.dropna(subset=["aqi"]).reset_index()
        node_df["h3_index"] = h3_idx
        rows.append(node_df)
    panel = pd.concat(rows, ignore_index=True)

    panel = panel.sort_values(["timestamp", "h3_index"]).reset_index(drop=True)
    return panel, h3_indices, source


def panel_to_sequences(
    panel: pd.DataFrame,
    h3_indices: list[str],
    lookback: int = LOOKBACK_HOURS,
    horizons: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[pd.Timestamp]]:
    if horizons is None:
        horizons = HORIZONS

    node_to_idx = {h: i for i, h in enumerate(h3_indices)}
    timestamps = sorted(panel["timestamp"].unique())
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

    feature_cube = np.full(
        (len(h3_indices), len(timestamps), len(FEATURE_COLUMNS)),
        np.nan,
        dtype=np.float32,
    )
    aqi_cube = np.full((len(h3_indices), len(timestamps)), np.nan, dtype=np.float32)

    for row in panel.itertuples(index=False):
        node_idx = node_to_idx[row.h3_index]
        time_idx = ts_to_idx[row.timestamp]
        feature_cube[node_idx, time_idx, 0] = row.aqi
        for feat_idx, col in enumerate(FEATURE_COLUMNS[1:], start=1):
            feature_cube[node_idx, time_idx, feat_idx] = getattr(row, col)
        aqi_cube[node_idx, time_idx] = row.aqi

    for node_idx in range(feature_cube.shape[0]):
        node_df = pd.DataFrame(feature_cube[node_idx], index=timestamps, columns=FEATURE_COLUMNS)
        node_df = node_df.interpolate(method="time").ffill().bfill()
        feature_cube[node_idx] = node_df.to_numpy(dtype=np.float32)
        aqi_cube[node_idx] = node_df["aqi"].to_numpy(dtype=np.float32)

    max_horizon = max(horizons)
    samples_x = []
    samples_y = []
    sample_timestamps = []

    for t in range(lookback, len(timestamps) - max_horizon):
        x_window = feature_cube[:, t - lookback : t, :]
        y_targets = np.stack([aqi_cube[:, t + h] for h in horizons], axis=-1)
        if np.isnan(x_window).any() or np.isnan(y_targets).any():
            continue
        samples_x.append(np.transpose(x_window, (0, 2, 1)))
        samples_y.append(y_targets)
        sample_timestamps.append(timestamps[t])

    if not samples_x:
        raise ValueError(
            "Not enough complete timesteps to build training sequences. "
            "Ensure OpenAQ and Open-Meteo overlap and cover at least "
            f"{lookback + max(horizons)} hours."
        )

    x = torch.tensor(np.stack(samples_x), dtype=torch.float32)
    y = torch.tensor(np.stack(samples_y), dtype=torch.float32)
    return x, y, sample_timestamps


def normalize_splits(splits: dict) -> tuple[dict, dict, dict]:
    x_train = splits["train"][0]
    feat_mean = x_train.mean(dim=(0, 1, 3), keepdim=True)
    feat_std = x_train.std(dim=(0, 1, 3), keepdim=True).clamp(min=1e-6)
    target_mean = splits["train"][1].mean()
    target_std = splits["train"][1].std().clamp(min=1e-6)

    norm_stats = {
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }

    normalized = {}
    for split_name, (x, y, ts) in splits.items():
        x_norm = (x - feat_mean) / feat_std
        y_norm = (y - target_mean) / target_std
        normalized[split_name] = (x_norm, y_norm, ts)
    return normalized, norm_stats, splits


def temporal_split(x: torch.Tensor, y: torch.Tensor, timestamps: list[pd.Timestamp]):
    n = x.size(0)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    splits = {
        "train": (x[:train_end], y[:train_end], timestamps[:train_end]),
        "val": (x[train_end:val_end], y[train_end:val_end], timestamps[train_end:val_end]),
        "test": (x[val_end:], y[val_end:], timestamps[val_end:]),
    }
    return splits


def build_graph_tensors(
    openmeteo_path: Path = DEFAULT_OPENMETEO_PATH,
    openaq_path: Path = DEFAULT_OPENAQ_PATH,
    resolution: int = 8,
) -> dict:
    latencies = []
    start = time.perf_counter()

    met_df = load_openmeteo(openmeteo_path)
    aq_df = load_openaq(openaq_path, met_df=met_df)
    panel, h3_indices, source = build_hourly_aqi_table(met_df, aq_df, resolution=resolution)

    mapper = H3Mapper(resolution=resolution)
    edge_index = mapper.build_adjacency_matrix(h3_indices)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    x, y, sample_timestamps = panel_to_sequences(panel, h3_indices)
    splits = temporal_split(x, y, sample_timestamps)

    latencies.append(time.perf_counter() - start)

    return {
        "splits": splits,
        "edge_index": edge_index,
        "h3_indices": h3_indices,
        "feature_columns": FEATURE_COLUMNS,
        "lookback_hours": LOOKBACK_HOURS,
        "horizons": HORIZONS,
        "source": source,
        "panel": panel,
        "graph_construction_latency_s": latencies[0],
    }


def save_processed(data: dict, output_dir: Path = PROJECT_ROOT / "data/processed") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "splits": data["splits"],
            "edge_index": data["edge_index"],
            "h3_indices": data["h3_indices"],
            "feature_columns": data["feature_columns"],
            "lookback_hours": data["lookback_hours"],
            "horizons": data["horizons"],
            "source": data["source"],
        },
        output_dir / "graph_tensors.pt",
    )
    data["panel"].to_parquet(output_dir / "hourly_panel.parquet", index=False)


if __name__ == "__main__":
    graph_data = build_graph_tensors()
    save_processed(graph_data)
    print(f"Built graph from {graph_data['source']} data.")
    print(f"Nodes: {len(graph_data['h3_indices'])}, edges: {graph_data['edge_index'].shape[1]}")
    print(f"Samples: {graph_data['splits']['train'][0].shape[0]} train / "
          f"{graph_data['splits']['val'][0].shape[0]} val / "
          f"{graph_data['splits']['test'][0].shape[0]} test")
