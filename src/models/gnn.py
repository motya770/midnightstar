# src/models/gnn.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GNNLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2, aggr: str = "mean"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

    def encode(self, data) -> torch.Tensor:
        x = data.x
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = torch.relu(x)
        return x

    def decode(self, z, src, dst):
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))

    def forward(self, data, src, dst):
        z = self.encode(data)
        return self.decode(z, src, dst)
