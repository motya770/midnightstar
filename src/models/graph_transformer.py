# src/models/graph_transformer.py
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_torch_csr_tensor


class RWSEEncoder(nn.Module):
    """Random Walk Structural Encoding using sparse matrix operations."""

    def __init__(self, walk_length: int = 16, out_dim: int = 8):
        super().__init__()
        self.walk_length = walk_length
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

        # Hutchinson estimator: approximate diag(RW^k) using random Rademacher probes
        # E[z_i * (RW^k z)_i] = (RW^k)_{ii} = return probability at step k
        num_probes = 32
        print(f"[DEBUG] RWSE: random probe mode ({num_nodes} nodes, {num_probes} probes, {self.walk_length} walks)...", flush=True)

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
        """Project cached walk probabilities through trainable linear layer."""
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
        self._rw_diag_cache = None  # cached raw walk probabilities

    def precompute_rwse(self, data):
        """Compute and cache RWSE walk probabilities. Call once before training."""
        import time as _time
        num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
        t0 = _time.time()
        print(f"[DEBUG] precompute_rwse: RWSE ({num_nodes} nodes)...", end=" ", flush=True)
        with torch.no_grad():
            rw_diag = self.rwse.compute_rw_diag(data.edge_index.cpu(), num_nodes)
        self._rw_diag_cache = rw_diag.to(data.x.device)
        print(f"{_time.time()-t0:.1f}s (cached)", flush=True)

    def encode(self, data):
        import time as _time
        num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
        target_device = data.x.device

        # Use cached walk probabilities if available, otherwise compute
        if self._rw_diag_cache is not None and self._rw_diag_cache.size(0) == num_nodes:
            rw_diag = self._rw_diag_cache.to(target_device)
        else:
            t0 = _time.time()
            print(f"[DEBUG] encode: RWSE ({num_nodes} nodes, no cache)...", end=" ", flush=True)
            with torch.no_grad():
                rw_diag = self.rwse.compute_rw_diag(data.edge_index.cpu(), num_nodes)
            rw_diag = rw_diag.to(target_device)
            print(f"{_time.time()-t0:.1f}s", flush=True)

        # Linear projection (trainable — receives gradients)
        t0 = _time.time()
        print("[DEBUG] encode: RWSE proj + input_proj...", end=" ", flush=True)
        pe = self.rwse(rw_diag)
        x = torch.cat([data.x, pe], dim=-1)
        x = self.input_proj(x)
        print(f"{_time.time()-t0:.1f}s", flush=True)

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            t0 = _time.time()
            print(f"[DEBUG] encode: TransformerConv layer {i}...", end=" ", flush=True)
            x = norm(x + layer(x, data.edge_index))
            x = torch.relu(x)
            print(f"{_time.time()-t0:.1f}s", flush=True)

        return x

    def decode_logits(self, z, src, dst):
        return (z[src] * z[dst]).sum(dim=-1)

    def decode(self, z, src, dst):
        return torch.sigmoid(self.decode_logits(z, src, dst))

    def forward(self, data, src, dst):
        z = self.encode(data)
        return self.decode(z, src, dst)
