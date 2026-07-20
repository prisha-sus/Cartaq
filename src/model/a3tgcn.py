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
        # A3TGCN expects x shape: [batch_size, num_nodes, in_channels, periods]
        # So we just pass it directly.
        h = self.tgnn(x, edge_index, edge_weight)
        # h shape: [batch_size, num_nodes, hidden_channels]
        
        h = F.relu(h)
        out = self.linear(h)
        # out shape: [batch_size, num_nodes, 3]
        return out
