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

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # Build sparse row-normalized adjacency (random walk matrix)
        adj_sparse = to_torch_csr_tensor(edge_index, size=(num_nodes, num_nodes))
        adj_dense_row = adj_sparse.to_dense() if num_nodes <= 5000 else None

        # Compute degree for row normalization
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
        deg = deg.clamp(min=1)

        # Build sparse random walk matrix: RW = D^{-1} A
        row, col = edge_index
        rw_values = 1.0 / deg[row]
        rw_sparse = torch.sparse_coo_tensor(
            edge_index, rw_values, (num_nodes, num_nodes)
        ).coalesce()

        # Compute RW diagonal landing probabilities via sparse matrix-vector products
        # Instead of full matrix power, track per-node probability vectors
        rw_diag = torch.zeros(num_nodes, self.walk_length, device=edge_index.device)

        # For each walk step k, we need diag(RW^k)
        # Use: p_k = RW^T @ p_{k-1} where p_0 = I (each node starts at itself)
        # diag(RW^k)[i] = (RW^k @ e_i)[i] = probability of returning to node i after k steps
        # Batch all nodes: P_0 = I, P_k = RW^T @ P_{k-1}, diag = diagonal of P_k
        # But storing full P is NxN. Instead, track only the diagonal using:
        # diag(RW^k) = sum over paths. Approximate with sparse power iteration on diagonal.

        # Efficient approach: use the fact that diag(A^k) = trace contribution per node
        # Compute via repeated sparse mat-vec: for each node, track return probability
        # Use batched sparse multiplication on identity columns

        rw_t = rw_sparse.t()

        import time as _time
        if num_nodes <= 10000:
            print(f"[DEBUG] RWSE: small graph mode ({num_nodes} nodes, {self.walk_length} walks)...", flush=True)
            current = torch.eye(num_nodes, device=edge_index.device)
            for k in range(self.walk_length):
                current = torch.sparse.mm(rw_t, current)
                rw_diag[:, k] = current.diagonal()
        else:
            chunk_size = 2000
            total_chunks = (num_nodes + chunk_size - 1) // chunk_size
            print(f"[DEBUG] RWSE: large graph mode ({num_nodes} nodes, {total_chunks} chunks, {self.walk_length} walks)...", flush=True)
            t0 = _time.time()
            for ci, start in enumerate(range(0, num_nodes, chunk_size)):
                end = min(start + chunk_size, num_nodes)
                current = torch.zeros(num_nodes, end - start, device=edge_index.device)
                current[start:end] = torch.eye(end - start, device=edge_index.device)
                for k in range(self.walk_length):
                    current = torch.sparse.mm(rw_t, current)
                    rw_diag[start:end, k] = current[range(start, end), range(end - start)]
                if (ci + 1) % 5 == 0 or ci == total_chunks - 1:
                    elapsed = _time.time() - t0
                    print(f"[DEBUG] RWSE: chunk {ci+1}/{total_chunks} ({elapsed:.1f}s)", flush=True)

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
        import time as _time
        num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
        target_device = data.x.device

        # RWSE uses sparse ops not supported on MPS — always compute on CPU
        t0 = _time.time()
        print(f"[DEBUG] encode: RWSE ({num_nodes} nodes)...", end=" ", flush=True)
        with torch.no_grad():
            rwse_device = next(self.rwse.parameters()).device
            self.rwse.cpu()
            pe = self.rwse(data.edge_index.cpu(), num_nodes)
            self.rwse.to(rwse_device)
            pe = pe.to(target_device)
        print(f"{_time.time()-t0:.1f}s", flush=True)

        t0 = _time.time()
        print("[DEBUG] encode: input_proj...", end=" ", flush=True)
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

    def decode(self, z, src, dst):
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))

    def forward(self, data, src, dst):
        z = self.encode(data)
        return self.decode(z, src, dst)
