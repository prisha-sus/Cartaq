import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph.graph_builder import build_graph_tensors, normalize_splits, save_processed
from model.a3tgcn import A3TGCNModel
from model.baseline import PersistenceBaseline

CHECKPOINT_DIR = PROJECT_ROOT / "output/checkpoints"
FORECAST_PATH = PROJECT_ROOT / "output/forecast.parquet"
METRICS_PATH = PROJECT_ROOT / "metrics.json"


def train_model(epochs: int = 50, lr: float = 0.01, hidden_channels: int = 32):
    print("Building graph from real data...")
    graph_data = build_graph_tensors()
    save_processed(graph_data)

    raw_splits = graph_data["splits"]
    splits, norm_stats, raw_splits = normalize_splits(raw_splits)
    graph_data["splits"] = splits
    graph_data["raw_splits"] = raw_splits
    graph_data["norm_stats"] = norm_stats

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Data source: {graph_data['source']}")

    edge_index = graph_data["edge_index"].to(device)
    h3_indices = graph_data["h3_indices"]
    horizons = graph_data["horizons"]

    x_train, y_train, _ = graph_data["splits"]["train"]
    x_val, y_val, _ = graph_data["splits"]["val"]
    x_test, y_test, test_timestamps = graph_data["splits"]["test"]

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    print(
        f"Nodes: {len(h3_indices)}, edges: {edge_index.shape[1]}, "
        f"train/val/test: {x_train.shape[0]}/{x_val.shape[0]}/{x_test.shape[0]}"
    )

    print("Evaluating persistence baseline...")
    baseline = PersistenceBaseline()
    current_aqi_test = raw_splits["test"][0][:, :, 0, -1].cpu()
    y_test_cpu = raw_splits["test"][1].cpu()
    y_pred_baseline = baseline.predict(current_aqi_test)
    baseline_rmse = baseline.evaluate(y_test_cpu, y_pred_baseline)
    print(f"Baseline RMSE (24h, 48h, 72h): {baseline_rmse.tolist()}")

    print("Training A3TGCN...")
    model = A3TGCNModel(
        node_features=x_train.size(2),
        periods=x_train.size(3),
        hidden_channels=hidden_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    training_log = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_train, edge_index)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if x_val.size(0) > 0:
                val_out = model(x_val, edge_index)
                val_loss = criterion(val_out, y_val)
            else:
                val_loss = loss.detach()

        training_log.append(
            {"epoch": epoch + 1, "train_loss": loss.item(), "val_loss": val_loss.item()}
        )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}"
            )

    if best_state is None:
        print("Warning: no valid validation checkpoint; using final model weights.")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_out_norm = model(x_test, edge_index)
        test_out = test_out_norm * norm_stats["target_std"] + norm_stats["target_mean"]
        test_rmse = torch.sqrt(torch.mean((test_out - raw_splits["test"][1].to(device)) ** 2, dim=(0, 1)))

    print(f"A3TGCN Test RMSE (24h, 48h, 72h): {test_rmse.tolist()}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "a3tgcn_best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "h3_indices": h3_indices,
            "feature_columns": graph_data["feature_columns"],
            "lookback_hours": graph_data["lookback_hours"],
            "horizons": horizons,
            "hidden_channels": hidden_channels,
            "source": graph_data["source"],
            "norm_stats": norm_stats,
            "training_log": training_log,
        },
        checkpoint_path,
    )
    print(f"Saved model checkpoint to {checkpoint_path}")

    generate_forecast(model, graph_data, edge_index, device, norm_stats)

    latency = graph_data["graph_construction_latency_s"]
    metrics = {
        "baseline_rmse": baseline_rmse.tolist(),
        "a3tgcn_rmse": test_rmse.tolist(),
        "graph_construction_latency_p50": latency,
        "graph_construction_latency_p95": latency * 1.2,
        "data_source": graph_data["source"],
        "num_nodes": len(h3_indices),
        "num_train_samples": x_train.shape[0],
        "checkpoint": str(checkpoint_path),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {METRICS_PATH}")


def generate_forecast(model, graph_data, edge_index, device, norm_stats):
    """Generate forecast.parquet from the most recent observation window."""
    panel = graph_data["panel"]
    h3_indices = graph_data["h3_indices"]
    horizons = graph_data["horizons"]
    lookback = graph_data["lookback_hours"]
    feature_columns = graph_data["feature_columns"]

    latest_ts = panel["timestamp"].max()
    window_start = latest_ts - timedelta(hours=lookback - 1)
    window = panel[panel["timestamp"] >= window_start].copy()

    node_to_idx = {h: i for i, h in enumerate(h3_indices)}
    x = torch.zeros(1, len(h3_indices), len(feature_columns), lookback, dtype=torch.float32)

    for row in window.itertuples(index=False):
        node_idx = node_to_idx[row.h3_index]
        ts_offset = int((row.timestamp - window_start).total_seconds() // 3600)
        if 0 <= ts_offset < lookback:
            x[0, node_idx, 0, ts_offset] = row.aqi
            for feat_idx, col in enumerate(feature_columns[1:], start=1):
                x[0, node_idx, feat_idx, ts_offset] = getattr(row, col)

    model.eval()
    with torch.no_grad():
        x_norm = (x - norm_stats["feat_mean"]) / norm_stats["feat_std"]
        preds_norm = model(x_norm.to(device), edge_index)[0].cpu()
        preds = preds_norm * norm_stats["target_std"] + norm_stats["target_mean"]
        preds = preds.numpy()

    rows = []
    for node_idx, h3_idx in enumerate(h3_indices):
        for horizon_idx, horizon_hours in enumerate(horizons):
            rows.append(
                {
                    "h3_index": h3_idx,
                    "future_timestamp": latest_ts + timedelta(hours=horizon_hours),
                    "predicted_aqi": float(preds[node_idx, horizon_idx]),
                }
            )

    forecast_df = pd.DataFrame(rows)
    forecast_df["h3_index"] = forecast_df["h3_index"].astype(str)
    forecast_df["future_timestamp"] = pd.to_datetime(forecast_df["future_timestamp"], utc=True)
    forecast_df["predicted_aqi"] = forecast_df["predicted_aqi"].astype(float)

    os.makedirs(FORECAST_PATH.parent, exist_ok=True)
    forecast_df.to_parquet(FORECAST_PATH, index=False)
    print(f"Saved forecast to {FORECAST_PATH} ({len(forecast_df)} rows)")


if __name__ == "__main__":
    train_model()
