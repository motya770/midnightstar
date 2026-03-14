# src/models/graph_transformer.py
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_adj


class RWSEEncoder(nn.Module):
    def __init__(self, walk_length: int = 16, out_dim: int = 8):
        super().__init__()
        self.walk_length = walk_length
        self.linear = nn.Linear(walk_length, out_dim)

    def forward(self, edge_index, num_nodes):
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        rw = adj / deg
        rw_diag = torch.zeros(num_nodes, self.walk_length, device=edge_index.device)
        power = torch.eye(num_nodes, device=edge_index.device)
        for k in range(self.walk_length):
            power = power @ rw
            rw_diag[:, k] = power.diagonal()
        return self.linear(rw_diag)


class GraphTransformerLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2,
                 num_heads: int = 4, rwse_dim: int = 16, rwse_walk_length: int = 16):
        super().__init__()
        self.rwse = RWSEEncoder(walk_length=rwse_walk_length, out_dim=rwse_dim)
        self.input_proj = nn.Linear(in_channels + rwse_dim, hidden_channels)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerConv(hidden_channels, hidden_channels // num_heads, heads=num_heads))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

    def encode(self, data):
        num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
        pe = self.rwse(data.edge_index, num_nodes)
        x = torch.cat([data.x, pe], dim=-1)
        x = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x, data.edge_index))
            x = torch.relu(x)
        return x

    def decode(self, z, src, dst):
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))

    def forward(self, data, src, dst):
        z = self.encode(data)
        return self.decode(z, src, dst)
