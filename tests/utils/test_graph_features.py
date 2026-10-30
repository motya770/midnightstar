# tests/utils/test_graph_features.py
import networkx as nx
import torch

from src.utils.graph_features import nx_to_pyg_data


def test_nx_to_pyg_tolerates_none_alphafold_values():
    G = nx.Graph()
    G.add_edge("A", "B")
    G.nodes["A"].update(mean_plddt=80.0, disordered_fraction=0.1, sequence_length=100)
    G.nodes["B"].update(mean_plddt=None, disordered_fraction=None, sequence_length=None)
    data, nodes, _ = nx_to_pyg_data(G)
    assert data.x.shape[0] == 2
    assert torch.isfinite(data.x).all()


def test_nx_to_pyg_exposes_degree_column():
    G = nx.relabel_nodes(nx.path_graph(4), {i: f"G{i}" for i in range(4)})
    data, nodes, _ = nx_to_pyg_data(G)
    deg = torch.tensor([G.degree(n) for n in nodes], dtype=torch.float)
    assert data.degree_col == data.x.size(1) - 1
    assert torch.allclose(data.x[:, data.degree_col], deg / deg.max())
