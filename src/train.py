"""
ST-GNN Training Loop with Physics-Constrained Loss

What changed from the original:
    1. Accepts real graph data from graph_builder.py when available
    2. Uses PhysicsLoss (MSE + wind advection penalty) instead of plain MSELoss
    3. Logs both MSE and physics penalty per epoch
    4. Saves model checkpoint alongside metrics.json

Run from Cartaq/src/:
    python train.py

Features:
    - Temporal split (no shuffle) - no future leakage
    - Persistence baseline reported alongside model RMSE
    - Physics penalty shows how much the wind constraint contributed
    - p50/p95 graph construction latency from metadata.json
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))

from model.a3tgcn import A3TGCNModel
from model.baseline import PersistenceBaseline
from model.physics_loss import PhysicsLoss, compute_wind_edge_weights
from pipeline.graph_builder import _generate_realistic_synthetic

# ── Config ────────────────────────────────────────────────────────────────────
GRAPH_DIR        = Path("../data/graph")
CHECKPOINT_PATH  = Path("../output/a3tgcn_checkpoint.pt")
EPOCHS           = 30
LEARNING_RATE    = 0.005
HIDDEN_CHANNELS  = 64
LAMBDA_PHYSICS   = 0.10    # weight of physics penalty term
WIND_DIR_DEG     = 270.0   # default wind direction (from west); override with real data
WIND_SPEED_KMH   = 15.0    # default wind speed km/h
# ─────────────────────────────────────────────────────────────────────────────


def load_graph_data():
    """
    Load tensors from graph_builder.py output.
    Falls back to generating realistic synthetic data if no saved tensors exist.
    """
    if not (GRAPH_DIR / "x.pt").exists():
        print("  No saved graph tensors found - generating realistic synthetic data.")
        os.makedirs(str(GRAPH_DIR), exist_ok=True)
        _generate_realistic_synthetic(output_dir=str(GRAPH_DIR))
        print()

    x          = torch.load(GRAPH_DIR / "x.pt")
    y          = torch.load(GRAPH_DIR / "y.pt")
    edge_index = torch.load(GRAPH_DIR / "edge_index.pt")

    # Try loading pre-computed wind edge weights
    edge_weight = None
    if (GRAPH_DIR / "edge_weight.pt").exists():
        edge_weight = torch.load(GRAPH_DIR / "edge_weight.pt")
    else:
        # Compute on the fly from H3 indices
        h3_path = GRAPH_DIR / "h3_indices.json"
        if h3_path.exists():
            with open(h3_path) as f:
                h3_indices = json.load(f)
            ei_np, ew_np = compute_wind_edge_weights(
                h3_indices, WIND_DIR_DEG, WIND_SPEED_KMH
            )
            if ei_np.shape[1] > 0:
                edge_index  = torch.tensor(ei_np,  dtype=torch.long)
                edge_weight = torch.tensor(ew_np,  dtype=torch.float32)

    meta = {}
    if (GRAPH_DIR / "metadata.json").exists():
        with open(GRAPH_DIR / "metadata.json") as f:
            meta = json.load(f)

    print(f"  Loaded: x={list(x.shape)}  y={list(y.shape)}  "
          f"edge_index={list(edge_index.shape)}  "
          f"edge_weight={'yes' if edge_weight is not None else 'none'}")
    return x, y, edge_index, edge_weight, meta


def train_model():
    print("=" * 60)
    print("  Cartaq - A3TGCN Training with Physics-Constrained Loss")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Load data ──────────────────────────────────────────────
    print("\n[1/4] Loading graph data...")
    x, y, edge_index, edge_weight, meta = load_graph_data()
    print(f"  {meta.get('n_nodes','?')} nodes, "
          f"{meta.get('batch_size','?')} samples")

    x          = x.to(device)
    y          = y.to(device)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # ── Temporal split (no shuffle - this is forecasting) ─────
    B          = x.size(0)
    train_end  = int(0.70 * B)
    val_end    = int(0.85 * B)

    x_train,   y_train   = x[:train_end],           y[:train_end]
    x_val,     y_val     = x[train_end:val_end],     y[train_end:val_end]
    x_test,    y_test    = x[val_end:],              y[val_end:]

    print(f"\n  Train: {train_end} | Val: {val_end - train_end} | Test: {B - val_end} samples")

    # ── Persistence baseline ──────────────────────────────────
    print("\n[2/4] Evaluating persistence baseline...")
    baseline = PersistenceBaseline()
    # Feature 0 is PM2.5/AQI - last time step of the lookback window
    current_aqi_test = x_test[:, :, 0, -1].cpu()
    y_pred_base      = baseline.predict(current_aqi_test)
    baseline_rmse    = baseline.evaluate(y_test.cpu(), y_pred_base)
    print(f"  Baseline RMSE (+24h, +48h, +72h): {[f'{v:.4f}' for v in baseline_rmse.tolist()]}")

    # ── Model + optimizer + loss ──────────────────────────────
    print("\n[3/4] Training A3TGCN with physics-constrained loss...")
    model     = A3TGCNModel(
        node_features=x.size(2),
        periods=x.size(3),
        hidden_channels=HIDDEN_CHANNELS,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = PhysicsLoss(lambda_physics=LAMBDA_PHYSICS)

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Physics lambda: {LAMBDA_PHYSICS}")
    print(f"  Wind: {WIND_DIR_DEG} deg from, {WIND_SPEED_KMH} km/h\n")

    history = []
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(x_train, edge_index, edge_weight)
        loss, comps = criterion(out, y_train, edge_index, edge_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(x_val, edge_index, edge_weight)
            val_loss, val_comps = criterion(val_out, y_val, edge_index, edge_weight)

        scheduler.step()

        history.append({
            "epoch":        epoch,
            "train_loss":   round(comps["total"],   4),
            "train_mse":    round(comps["mse"],     4),
            "train_phys":   round(comps["physics"], 4),
            "val_loss":     round(val_comps["total"], 4),
        })

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS} | "
                f"Train: {comps['total']:.4f} "
                f"(MSE={comps['mse']:.4f}, Phys={comps['physics']:.4f}) | "
                f"Val: {val_comps['total']:.4f}"
            )

    # ── Test evaluation ───────────────────────────────────────
    print("\n[4/4] Test evaluation...")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_out  = model(x_test, edge_index, edge_weight)
        test_rmse = torch.sqrt(
            torch.mean((test_out - y_test) ** 2, dim=(0, 1))
        )

    print(f"  A3TGCN RMSE  (+24h, +48h, +72h): {[f'{v:.4f}' for v in test_rmse.tolist()]}")
    print(f"  Baseline RMSE (+24h, +48h, +72h): {[f'{v:.4f}' for v in baseline_rmse.tolist()]}")

    improvement = [(b - m) / b * 100 for b, m in
                   zip(baseline_rmse.tolist(), test_rmse.tolist())]
    print(f"  Improvement over baseline:         {[f'{v:.1f}%' for v in improvement]}")

    # ── Save checkpoint ──────────────────────────────────────
    os.makedirs(CHECKPOINT_PATH.parent, exist_ok=True)
    torch.save({
        "model_state_dict":  best_state,
        "epoch":             EPOCHS,
        "val_loss":          best_val_loss,
        "node_features":     x.size(2),
        "periods":           x.size(3),
        "hidden_channels":   HIDDEN_CHANNELS,
    }, CHECKPOINT_PATH)
    print(f"\n  Checkpoint saved -> {CHECKPOINT_PATH}")

    # ── Save metrics ─────────────────────────────────────────
    metrics = {
        "baseline_rmse":                 baseline_rmse.tolist(),
        "a3tgcn_rmse":                   test_rmse.tolist(),
        "improvement_over_baseline_pct": improvement,
        "lambda_physics":                LAMBDA_PHYSICS,
        "final_physics_penalty":         history[-1]["train_phys"],
        "training_history":              history,
        "data_type":                     "realistic_synthetic" if meta.get("realistic_synthetic") else ("real" if not meta.get("synthetic") else "synthetic"),
        "n_nodes":                       meta.get("n_nodes", x.size(1)),
        "batch_size":                    meta.get("batch_size", x.size(0)),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  Metrics saved -> metrics.json")

    return metrics


if __name__ == "__main__":
    train_model()
