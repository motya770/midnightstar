"""Heterogeneous gene–trait link predictor.

Genes and traits are projected into one hidden space, message passing runs over
STRING (gene–gene) and GWAS (gene–trait, both directions) edges with a SAGEConv
per relation, and a decoder scores (gene, trait) pairs. The decoder logit is the
ranking score for disease-gene discovery.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

EDGE_TYPES = (
    ("gene", "interacts", "gene"),
    ("gene", "has", "trait"),
    ("trait", "rev_has", "gene"),
)
DECODERS = ("dot", "bilinear", "mlp")


class HeteroGNNLinkPredictor(nn.Module):
    def __init__(self, gene_in: int, num_traits: int, trait_in: int = 6, hidden_channels: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, decoder: str = "dot"):
        super().__init__()
        if decoder not in DECODERS:
            raise ValueError(f"decoder must be one of {DECODERS}, got {decoder!r}")
        self.decoder_kind = decoder
        self.hidden_channels = hidden_channels

        self.gene_in = nn.Linear(gene_in, hidden_channels)
        self.trait_in = nn.Linear(trait_in, hidden_channels)
        self.trait_embedding = nn.Embedding(num_traits, hidden_channels)
        self.trait_bias = nn.Parameter(torch.zeros(num_traits))

        self.convs = nn.ModuleList([
            HeteroConv({et: SAGEConv((hidden_channels, hidden_channels), hidden_channels) for et in EDGE_TYPES},
                       aggr="sum")
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({"gene": nn.LayerNorm(hidden_channels), "trait": nn.LayerNorm(hidden_channels)})
            for _ in range(num_layers - 1)
        ])
        self.dropout = nn.Dropout(dropout)

        if decoder == "bilinear":
            self.bilinear = nn.Linear(hidden_channels, hidden_channels, bias=False)
        elif decoder == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(3 * hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, 1),
            )

    # ---- encoder ------------------------------------------------------------

    def encode(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_gene, z_trait)."""
        num_traits = data["trait"].x.size(0)
        trait_ids = torch.arange(num_traits, device=data["trait"].x.device)
        x = {
            "gene": self.gene_in(data["gene"].x),
            "trait": self.trait_in(data["trait"].x) + self.trait_embedding(trait_ids),
        }
        edge_index = {et: data[et].edge_index for et in EDGE_TYPES}
        last = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            out = conv(x, edge_index)
            # A node type that received no messages keeps its current representation.
            x = {k: out.get(k, x[k]) for k in x}
            if i < last:
                x = {k: self.dropout(torch.relu(self.norms[i][k](v))) for k, v in x.items()}
        return x["gene"], x["trait"]

    # ---- decoders -----------------------------------------------------------

    def decode_logits(self, z_gene, z_trait, gene_idx, trait_idx) -> torch.Tensor:
        g, t = z_gene[gene_idx], z_trait[trait_idx]
        if self.decoder_kind == "dot":
            score = (g * t).sum(dim=-1)
        elif self.decoder_kind == "bilinear":
            score = (self.bilinear(g) * t).sum(dim=-1)
        else:
            score = self.mlp(torch.cat([g, t, g * t], dim=-1)).squeeze(-1)
        return score + self.trait_bias[trait_idx]

    def decode(self, z_gene, z_trait, gene_idx, trait_idx) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(z_gene, z_trait, gene_idx, trait_idx))

    def score_matrix(self, z_gene, z_trait, trait_idx) -> torch.Tensor:
        """Logits of every gene against each trait in ``trait_idx``: (n_genes, len(trait_idx))."""
        trait_idx = torch.as_tensor(trait_idx, device=z_gene.device, dtype=torch.long)
        t = z_trait[trait_idx]
        bias = self.trait_bias[trait_idx]
        if self.decoder_kind == "dot":
            return z_gene @ t.T + bias
        if self.decoder_kind == "bilinear":
            return self.bilinear(z_gene) @ t.T + bias
        n = z_gene.size(0)
        columns = [
            self.decode_logits(z_gene, z_trait, torch.arange(n, device=z_gene.device),
                               torch.full((n,), int(ti), device=z_gene.device, dtype=torch.long))
            for ti in trait_idx.tolist()
        ]
        return torch.stack(columns, dim=1) if columns else z_gene.new_zeros((n, 0))

    def forward(self, data, gene_idx, trait_idx):
        z_gene, z_trait = self.encode(data)
        return self.decode(z_gene, z_trait, gene_idx, trait_idx)
