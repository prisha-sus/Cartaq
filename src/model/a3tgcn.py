"""
A3TGCN-equivalent model in 100% pure PyTorch (no torch_geometric needed).

Architecture:
  - Temporal branch: GRU over T time-steps per node
  - Temporal attention: soft attention over GRU outputs
  - Spatial branch: two manual Graph Convolutional layers (Kipf & Welling 2017)
  - Output head: 3 forecast horizons (+24h, +48h, +72h)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_adj(edge_index: torch.Tensor, num_nodes: int, edge_weight=None) -> torch.Tensor:
    """
    Build a normalised (symmetric) adjacency matrix A_hat from COO edge_index.
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    Returns: dense [N, N] float32 tensor on the same device as edge_index.
    """
    device = edge_index.device
    N = num_nodes

    # Add self-loops
    self_loop = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)
    ei = torch.cat([edge_index, self_loop], dim=1)

    if edge_weight is not None:
        ew = torch.cat([edge_weight, torch.ones(N, device=device)])
    else:
        ew = torch.ones(ei.size(1), device=device)

    # Build sparse adjacency, then densify
    A = torch.zeros(N, N, device=device)
    A[ei[0], ei[1]] = ew

    # Symmetric normalisation
    deg = A.sum(dim=1).clamp(min=1e-6)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt


class _GCNLayer(nn.Module):
    """One GCN layer: H' = A_hat H W"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)

    def forward(self, h: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # h: [N, F]   A_hat: [N, N]
        return self.W(A_hat @ h)


class _TemporalAttention(nn.Module):
    """Soft attention over T time steps."""
    def __init__(self, hidden: int, periods: int):
        super().__init__()
        self.fc = nn.Linear(hidden * periods, periods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BN, H, T]
        BN, H, T = x.shape
        alpha = torch.softmax(self.fc(x.reshape(BN, H * T)), dim=-1)  # [BN, T]
        return (x * alpha.unsqueeze(1)).sum(dim=-1)                    # [BN, H]


class A3TGCNModel(nn.Module):
    """
    Attention Temporal Graph Convolutional Network - pure PyTorch.

    Input
    -----
    x:           [B, N, F, T]  batch × nodes × features × lookback time-steps
    edge_index:  [2, E]        COO graph edges
    edge_weight: [E]           optional edge weights

    Output
    ------
    [B, N, 3]   predicted AQI at +24h, +48h, +72h per node
    """

    def __init__(self, node_features: int, periods: int, hidden_channels: int = 64):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Temporal branch
        self.gru  = nn.GRU(node_features, hidden_channels, batch_first=True)
        self.attn = _TemporalAttention(hidden_channels, periods)

        # Spatial branch
        self.gcn1 = _GCNLayer(hidden_channels, hidden_channels)
        self.gcn2 = _GCNLayer(hidden_channels, hidden_channels)

        # Forecast head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 3),
        )

        # Cache: avoid recomputing A_hat every forward pass unless graph changes
        self._cached_A_hat = None
        self._cached_ei_shape = None

    def forward(self, x, edge_index, edge_weight=None):
        B, N, n_feat, T = x.shape

        # ── Build (or reuse) normalised adjacency ─────────────────────────
        ei_shape = (edge_index.shape, None if edge_weight is None else edge_weight.shape)
        if self._cached_A_hat is None or self._cached_ei_shape != ei_shape:
            self._cached_A_hat    = _build_adj(edge_index, N, edge_weight)
            self._cached_ei_shape = ei_shape
        A_hat = self._cached_A_hat                              # [N, N]

        # ── Temporal encoding ──────────────────────────────────────────────
        # [B, N, n_feat, T] -> [B*N, T, n_feat]
        xt = x.permute(0, 1, 3, 2).reshape(B * N, T, n_feat)
        gru_out, _ = self.gru(xt)                               # [B*N, T, H]
        gru_out    = gru_out.permute(0, 2, 1)                  # [B*N, H, T]
        h_temp     = self.attn(gru_out)                         # [B*N, H]
        h_temp     = h_temp.reshape(B, N, self.hidden_channels) # [B, N, H]

        # ── Spatial encoding (shared graph, applied per sample in batch) ───
        all_h = []
        for b in range(B):
            h_b = F.relu(self.gcn1(h_temp[b], A_hat))          # [N, H]
            h_b = F.relu(self.gcn2(h_b, A_hat))
            all_h.append(h_b)
        h_spat = torch.stack(all_h, dim=0)                      # [B, N, H]

        # ── Forecast head ──────────────────────────────────────────────────
        return self.head(h_spat)                                 # [B, N, 3]
