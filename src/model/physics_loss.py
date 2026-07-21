"""
Physics-Constrained Loss Function (The "PINN Hack")

This is the scientific core that makes the ST-GNN physics-aware without
needing a full PINN. From Prisha's description:

  L_total = L_pred + lambda * L_physics

Where:
  L_pred    = standard MSE between predicted and true AQI
  L_physics = penalty for predictions that violate wind advection physics

The Physics:
  The advection-diffusion equation states that pollution concentration C
  at a downwind node i should be >= the wind-transported concentration
  from its upwind neighbor j (plus local sources can add, not subtract).

  Simplified discrete form:
    C_i(t+h) >= w_ji * C_j(t)

  Where w_ji = max(0, cos(wind_dir - bearing(j->i))) * wind_speed / normalization

  If the model predicts C_i(t+h) < w_ji * C_j(t), it is predicting
  LESS pollution downwind than physics allows - this is penalized.

  The penalty is one-sided (relu): we only penalize violations, not
  reward the model for getting physics right (that's what L_pred is for).

Usage in train.py:
    from model.physics_loss import PhysicsLoss
    criterion = PhysicsLoss(lambda_physics=0.1)
    loss = criterion(predictions, targets, edge_index, edge_weight)
"""

import torch
import torch.nn as nn
import numpy as np


class PhysicsLoss(nn.Module):
    """
    Combined MSE + wind-advection physics penalty.

    Args:
        lambda_physics: weight of the physics penalty term (0.05 - 0.2 works well)
        reduction:      'mean' or 'sum' for the MSE component
    """

    def __init__(self, lambda_physics: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred:         [batch, num_nodes, num_horizons]  - model output
            target:       [batch, num_nodes, num_horizons]  - ground truth AQI
            edge_index:   [2, num_edges]                    - (src, dst) pairs
            edge_weight:  [num_edges]                       - wind transport weights
                          (positive = wind blows FROM src TO dst)

        Returns:
            total_loss: scalar tensor
            loss_components: dict with 'mse', 'physics', 'total' for logging
        """
        l_pred = self.mse(pred, target)

        if edge_weight is None or edge_index.size(1) == 0:
            return l_pred, {"mse": l_pred.item(), "physics": 0.0, "total": l_pred.item()}

        l_phys = self._advection_penalty(pred, edge_index, edge_weight)
        total  = l_pred + self.lambda_physics * l_phys

        return total, {
            "mse":     l_pred.item(),
            "physics": l_phys.item(),
            "total":   total.item(),
        }

    def _advection_penalty(
        self,
        pred: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each directed edge (src -> dst) with positive wind weight:

        Physics says: C_dst >= transport_factor * C_src
        (you can't have LESS pollution downwind than the wind brought)

        Penalty = ReLU(transport_factor * C_src - C_dst)
                = how much the downwind prediction is below what physics expects

        This is one-sided: the model isn't penalized for predicting MORE
        pollution downwind (local sources could cause that). It's only
        penalized for predicting LESS than the wind would transport.
        """
        src, dst = edge_index[0], edge_index[1]

        # Only edges where wind is actually blowing (positive transport weight)
        upwind_mask = edge_weight > 0
        if not upwind_mask.any():
            return torch.tensor(0.0, device=pred.device)

        src_uw  = src[upwind_mask]
        dst_uw  = dst[upwind_mask]
        w_uw    = edge_weight[upwind_mask].clamp(0, 1)   # transport factor in [0,1]

        # pred: [batch, nodes, horizons]
        # Index along the node dimension
        c_src = pred[:, src_uw, :]    # [batch, n_upwind_edges, horizons]
        c_dst = pred[:, dst_uw, :]    # [batch, n_upwind_edges, horizons]

        # Expected downwind concentration from advection
        expected = w_uw.unsqueeze(0).unsqueeze(-1) * c_src   # broadcast over batch + horizons

        # Physics violation: C_dst < expected
        violation = torch.relu(expected - c_dst)

        return violation.mean()


def compute_wind_edge_weights(
    h3_indices: list[str],
    wind_dir_deg: float,
    wind_speed_kmh: float,
) -> np.ndarray:
    """
    Compute wind-transport edge weights for a set of H3 hex indices.

    For each directed edge (src -> dst):
      weight = max(0, cos(wind_dir - bearing(src->dst))) * wind_speed

    Positive weight = wind blows FROM src TOWARD dst (src is upwind).
    Zero / negative = wind does not transport from src to dst (pruned to 0).

    Args:
        h3_indices:    ordered list of H3 cell IDs
        wind_dir_deg:  meteorological wind direction (degrees FROM which wind blows,
                       e.g. 270 = wind blowing FROM west)
        wind_speed_kmh: wind speed in km/h

    Returns:
        edge_index:    [2, num_edges] with ALL directed k-ring edges
        edge_weight:   [num_edges]    wind transport factor (>0 = downwind)
    """
    import h3 as h3lib

    idx_map    = {h: i for i, h in enumerate(h3_indices)}
    edges      = []
    weights    = []

    # Convert met convention (FROM direction) to TO direction
    # Wind FROM 270 means air moves TOWARD 90 (east)
    wind_to_deg = (wind_dir_deg + 180) % 360
    wind_to_rad = np.radians(wind_to_deg)

    for h3_idx in h3_indices:
        src = idx_map[h3_idx]
        lat_s, lon_s = h3lib.cell_to_latlng(h3_idx)

        neighbors = list(h3lib.grid_disk(h3_idx, 1))
        for nb in neighbors:
            if nb == h3_idx or nb not in idx_map:
                continue

            dst = idx_map[nb]
            lat_d, lon_d = h3lib.cell_to_latlng(nb)

            # Bearing FROM src TO dst (north=0, clockwise)
            dlat = np.radians(lat_d - lat_s)
            dlon = np.radians(lon_d - lon_s)
            bearing_rad = np.arctan2(
                np.sin(dlon) * np.cos(np.radians(lat_d)),
                np.cos(np.radians(lat_s)) * np.sin(np.radians(lat_d))
                - np.sin(np.radians(lat_s)) * np.cos(np.radians(lat_d)) * np.cos(dlon),
            )

            # How aligned is the wind direction with this edge?
            # cos(angle) = 1 when wind blows exactly along this edge direction
            angle_diff = bearing_rad - wind_to_rad
            transport  = np.cos(angle_diff)  # [-1, 1]

            # Normalize: weight = max(0, cos) * wind_speed / max_possible
            weight = max(0.0, transport) * (wind_speed_kmh / 30.0)  # normalize by ~max wind
            weight = min(weight, 1.0)

            edges.append([src, dst])
            weights.append(weight)

    if not edges:
        return np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float32)

    edge_index  = np.array(edges, dtype=np.int64).T
    edge_weight = np.array(weights, dtype=np.float32)
    return edge_index, edge_weight


if __name__ == "__main__":
    # Quick self-test
    torch.manual_seed(42)

    batch, nodes, horizons = 4, 10, 3
    num_edges = 20

    pred   = torch.abs(torch.randn(batch, nodes, horizons))
    target = torch.abs(torch.randn(batch, nodes, horizons))

    edge_index = torch.randint(0, nodes, (2, num_edges))
    # Realistic wind weights: most near 0 (cross-wind), some positive (downwind)
    edge_weight = torch.clamp(torch.randn(num_edges) * 0.3 + 0.2, 0, 1)

    criterion = PhysicsLoss(lambda_physics=0.1)
    loss, components = criterion(pred, target, edge_index, edge_weight)

    print("=== PhysicsLoss self-test ===")
    print(f"  MSE loss:      {components['mse']:.4f}")
    print(f"  Physics penalty: {components['physics']:.4f}")
    print(f"  Total loss:    {components['total']:.4f}")
    print(f"  Physics is {components['physics'] / components['mse'] * 100:.1f}% of MSE")

    # Test wind edge weights
    print("\n=== Wind edge weights ===")
    import h3
    center = h3.latlng_to_cell(18.5314, 73.8446, 8)
    hexes  = list(h3.grid_disk(center, 2))[:15]
    ei, ew = compute_wind_edge_weights(hexes, wind_dir_deg=270, wind_speed_kmh=15)
    print(f"  {ei.shape[1]} directed edges")
    print(f"  Weight range: [{ew.min():.3f}, {ew.max():.3f}]")
    print(f"  Positive (downwind) edges: {(ew > 0).sum()}")
    print(f"  Zero (crosswind/upwind):   {(ew == 0).sum()}")
