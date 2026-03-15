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

        if num_nodes <= 10000:
            # For moderate graphs: batch sparse matmul
            # P shape: (num_nodes, num_nodes), start as identity
            # Track as dense but operate with sparse RW
            current = torch.eye(num_nodes, device=edge_index.device)
            for k in range(self.walk_length):
                current = torch.sparse.mm(rw_t, current)
                rw_diag[:, k] = current.diagonal()
        else:
            # For large graphs: sample-based approximation
            # Process nodes in chunks to avoid OOM
            chunk_size = 2000
            for start in range(0, num_nodes, chunk_size):
                end = min(start + chunk_size, num_nodes)
                # Start with indicator vectors for this chunk
                current = torch.zeros(num_nodes, end - start, device=edge_index.device)
                current[start:end] = torch.eye(end - start, device=edge_index.device)
                for k in range(self.walk_length):
                    current = torch.sparse.mm(rw_t, current)
                    # Extract diagonal: for node i in [start, end), its return prob is current[i, i-start]
                    for local_i, global_i in enumerate(range(start, end)):
                        rw_diag[global_i, k] = current[global_i, local_i]

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
        with torch.no_grad():
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
