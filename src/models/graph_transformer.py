# src/models/graph_transformer.py
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class RWSEEncoder(nn.Module):
    """Random Walk Structural Encoding using sparse matrix operations."""

    def __init__(self, walk_length: int = 8, out_dim: int = 8, num_probes: int = 64):
        super().__init__()
        self.walk_length = walk_length
        self.num_probes = num_probes
        self.linear = nn.Linear(walk_length, out_dim)

    def compute_rw_diag(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute raw random walk return probabilities (no gradients needed)."""
        import time as _time

        t0 = _time.time()
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
        deg = deg.clamp(min=1)

        row, col = edge_index
        rw_values = 1.0 / deg[row]
        rw_diag = torch.zeros(num_nodes, self.walk_length, device=edge_index.device)

        num_probes = self.num_probes
        print(
            f"[DEBUG] RWSE: random probe mode ({num_nodes} nodes, {num_probes} probes, {self.walk_length} walks)...",
            flush=True,
        )

        probes_orig = torch.sign(torch.randn(num_nodes, num_probes, device=edge_index.device))
        probes = probes_orig.clone()

        for k in range(self.walk_length):
            src_vals = probes[row] * rw_values.unsqueeze(1)
            new_probes = torch.zeros_like(probes)
            new_probes.scatter_add_(0, col.unsqueeze(1).expand_as(src_vals), src_vals)
            probes = new_probes
            rw_diag[:, k] = (probes_orig * probes).mean(dim=1)

        print(f"[DEBUG] RWSE: done in {_time.time()-t0:.1f}s", flush=True)
        return rw_diag

    def forward(self, rw_diag: torch.Tensor) -> torch.Tensor:
        return self.linear(rw_diag)


class GraphTransformerBlock(nn.Module):
    """Pre-norm attention block with a feed-forward residual path."""

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        edge_dim: int,
        dropout: float,
        ff_mult: int,
    ):
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.attn = TransformerConv(
            hidden_channels,
            hidden_channels // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim if edge_dim > 0 else None,
        )
        self.drop1 = nn.Dropout(dropout)

        ff_hidden = hidden_channels * ff_mult
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, hidden_channels),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        h = self.norm1(x)
        if edge_attr is not None:
            h = self.attn(h, edge_index, edge_attr)
        else:
            h = self.attn(h, edge_index)
        x = x + self.drop1(h)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class GraphTransformerLinkPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        rwse_dim: int = 8,
        rwse_walk_length: int = 8,
        rwse_probes: int = 64,
        edge_dim: int = 5,
        dropout: float = 0.1,
        ff_mult: int = 2,
        gwas_vocab_size: int = 0,
        gwas_num_categories: int = 0,
        gwas_embed_dim: int = 32,
    ):
        super().__init__()
        from .gnn import GWASEncoder

        self.edge_dim = edge_dim
        self.gwas_encoder = None
        gwas_channels = 0
        if gwas_vocab_size > 1:
            self.gwas_encoder = GWASEncoder(gwas_vocab_size, gwas_num_categories, gwas_embed_dim)
            gwas_channels = gwas_embed_dim

        self.rwse = None
        rwse_channels = 0
        if rwse_dim > 0 and rwse_walk_length > 0:
            self.rwse = RWSEEncoder(
                walk_length=rwse_walk_length,
                out_dim=rwse_dim,
                num_probes=rwse_probes,
            )
            rwse_channels = rwse_dim

        total_in = in_channels + gwas_channels + rwse_channels
        self.input_proj = nn.Sequential(
            nn.Linear(total_in, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                GraphTransformerBlock(
                    hidden_channels=hidden_channels,
                    num_heads=num_heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    ff_mult=ff_mult,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_channels)
        self._rw_diag_cache = None
        self._rw_cache_edge_hash = None

    @staticmethod
    def _edge_hash(edge_index: torch.Tensor) -> int:
        ei = edge_index.cpu()
        return hash((ei.shape[0], ei.shape[1], ei.sum().item(), ei[0, -1].item() if ei.size(1) > 0 else 0))

    def precompute_rwse(self, data):
        """Compute and cache RWSE walk probabilities. Call once before training."""
        if self.rwse is None:
            return
        import time as _time

        num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
        t0 = _time.time()
        print(f"[DEBUG] precompute_rwse: RWSE ({num_nodes} nodes)...", end=" ", flush=True)
        with torch.no_grad():
            rw_diag = self.rwse.compute_rw_diag(data.edge_index.cpu(), num_nodes)
        self._rw_diag_cache = rw_diag.to(data.x.device)
        self._rw_cache_edge_hash = self._edge_hash(data.edge_index)
        print(f"{_time.time()-t0:.1f}s (cached)", flush=True)

    def _cache_valid(self, data, num_nodes: int) -> bool:
        if self._rw_diag_cache is None:
            return False
        if self._rw_diag_cache.size(0) != num_nodes:
            return False
        return self._rw_cache_edge_hash == self._edge_hash(data.edge_index)

    def _get_rwse(self, data, num_nodes: int, target_device: torch.device) -> torch.Tensor | None:
        if self.rwse is None:
            return None
        if self._cache_valid(data, num_nodes):
            return self._rw_diag_cache.to(target_device)

        import time as _time

        t0 = _time.time()
        print(f"[DEBUG] encode: RWSE ({num_nodes} nodes, no cache)...", end=" ", flush=True)
        with torch.no_grad():
            rw_diag = self.rwse.compute_rw_diag(data.edge_index.cpu(), num_nodes)
        rw_diag = rw_diag.to(target_device)
        print(f"{_time.time()-t0:.1f}s", flush=True)
        return rw_diag

    def encode(self, data):
        import time as _time

        num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
        target_device = data.x.device

        rw_diag = self._get_rwse(data, num_nodes, target_device)

        t0 = _time.time()
        print("[DEBUG] encode: feature projection...", end=" ", flush=True)
        parts = [data.x]
        if self.gwas_encoder is not None and hasattr(data, "gwas_token_ids"):
            gwas_feat = self.gwas_encoder(data.gwas_token_ids, data.gwas_scores, data.gwas_cat_ids)
            parts.append(gwas_feat)
        if rw_diag is not None:
            parts.append(self.rwse(rw_diag))
        x = self.input_proj(torch.cat(parts, dim=-1))
        print(f"{_time.time()-t0:.1f}s", flush=True)

        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None and self.edge_dim > 0:
            edge_attr = x.new_zeros((data.edge_index.size(1), self.edge_dim))

        for i, layer in enumerate(self.layers):
            t0 = _time.time()
            print(f"[DEBUG] encode: Transformer block {i}...", end=" ", flush=True)
            x = layer(x, data.edge_index, edge_attr)
            print(f"{_time.time()-t0:.1f}s", flush=True)

        return self.output_norm(x)

    def decode_logits(self, z, src, dst):
        return (z[src] * z[dst]).sum(dim=-1)

    def decode(self, z, src, dst):
        return torch.sigmoid(self.decode_logits(z, src, dst))

    def forward(self, data, src, dst):
        z = self.encode(data)
        return self.decode(z, src, dst)
