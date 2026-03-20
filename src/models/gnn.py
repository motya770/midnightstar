# src/models/gnn.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GWASEncoder(nn.Module):
    """Encode GWAS trait tokens into a fixed-size vector per node."""

    def __init__(self, vocab_size: int, num_categories: int, embed_dim: int = 32):
        super().__init__()
        self.trait_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cat_embed = nn.Embedding(num_categories, 8)
        self.proj = nn.Linear(embed_dim + 8 + 1, embed_dim)  # +1 for score

    def forward(self, token_ids, scores, cat_ids):
        """
        token_ids: [num_nodes, max_tokens] (0 = padding)
        scores: [num_nodes, max_tokens]
        cat_ids: [num_nodes, max_tokens]
        Returns: [num_nodes, embed_dim]
        """
        mask = token_ids != 0  # [num_nodes, max_tokens]
        t_emb = self.trait_embed(token_ids)        # [N, T, embed_dim]
        c_emb = self.cat_embed(cat_ids)            # [N, T, 8]
        s = scores.unsqueeze(-1)                   # [N, T, 1]
        combined = torch.cat([t_emb, c_emb, s], dim=-1)  # [N, T, embed_dim+9]
        combined = self.proj(combined)             # [N, T, embed_dim]
        # Score-weighted mean pooling (masked)
        combined = combined * s * mask.unsqueeze(-1).float()
        denom = (s * mask.unsqueeze(-1).float()).sum(dim=1).clamp(min=1e-8)
        return combined.sum(dim=1) / denom         # [N, embed_dim]


class GNNLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2, aggr: str = "mean",
                 gwas_vocab_size: int = 0, gwas_num_categories: int = 0, gwas_embed_dim: int = 32):
        super().__init__()
        self.gwas_encoder = None
        total_in = in_channels
        if gwas_vocab_size > 1:
            self.gwas_encoder = GWASEncoder(gwas_vocab_size, gwas_num_categories, gwas_embed_dim)
            total_in = in_channels + gwas_embed_dim

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(total_in, hidden_channels, aggr=aggr))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

    def encode(self, data) -> torch.Tensor:
        x = data.x
        if self.gwas_encoder is not None and hasattr(data, "gwas_token_ids"):
            gwas_feat = self.gwas_encoder(data.gwas_token_ids, data.gwas_scores, data.gwas_cat_ids)
            x = torch.cat([x, gwas_feat], dim=-1)
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = torch.relu(x)
        return x

    def decode(self, z, src, dst):
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))

    def forward(self, data, src, dst):
        z = self.encode(data)
        return self.decode(z, src, dst)
