import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

class A3TGCNModel(torch.nn.Module):
    def __init__(self, node_features, periods, hidden_channels=32):
        super(A3TGCNModel, self).__init__()
        # A3TGCN takes (in_channels, out_channels, periods)
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=hidden_channels, 
                           periods=periods)
        
        # We need to output +24h, +48h, +72h, so 3 horizons
        # The output of A3TGCN is [batch_size, num_nodes, hidden_channels]
        self.linear = torch.nn.Linear(hidden_channels, 3)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x shape: (batch_size, num_nodes, num_features, periods)
        edge_index shape: (2, num_edges)
        edge_weight shape: (num_edges,)
        """
        batch_size, num_nodes, num_features, periods = x.size()
        
        # Flatten batch and node dimensions for PyTorch Geometric Temporal
        x_batched = x.view(batch_size * num_nodes, num_features, periods)
        
        # Tile edge_index for the batched graph
        num_edges = edge_index.size(1)
        edge_index_batched = edge_index.repeat(1, batch_size)
        offsets = torch.arange(batch_size, device=x.device).view(1, -1).repeat_interleave(num_edges, dim=1) * num_nodes
        edge_index_batched = edge_index_batched + offsets
        
        if edge_weight is not None:
            edge_weight_batched = edge_weight.repeat(batch_size)
        else:
            edge_weight_batched = None
            
        # A3TGCN expects x shape: [total_nodes, in_channels, periods]
        h = self.tgnn(x_batched, edge_index_batched, edge_weight_batched)
        # h shape: [batch_size * num_nodes, hidden_channels]
        
        h = F.relu(h)
        out = self.linear(h)
        # out shape: [batch_size * num_nodes, 3]
        
        # Reshape back to [batch_size, num_nodes, 3]
        return out.view(batch_size, num_nodes, 3)
