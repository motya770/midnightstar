# tests/models/test_gnn.py
import torch
from torch_geometric.data import Data
from src.models.gnn import GNNLinkPredictor


def _make_data(num_nodes=5, in_channels=8):
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def test_gnn_forward_shape():
    data = _make_data()
    model = GNNLinkPredictor(in_channels=8, hidden_channels=32)
    src = torch.tensor([0, 2])
    dst = torch.tensor([1, 3])
    out = model(data, src, dst)
    assert out.shape == (2,), f"Expected shape (2,), got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "Output values should be in [0, 1]"


def test_gnn_different_configs():
    data = _make_data(num_nodes=6, in_channels=4)
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([3, 4, 5])
    for num_layers, hidden, aggr in [(1, 16, "mean"), (2, 32, "max"), (3, 64, "sum")]:
        model = GNNLinkPredictor(in_channels=4, hidden_channels=hidden, num_layers=num_layers, aggr=aggr)
        out = model(data, src, dst)
        assert out.shape == (3,), f"Config ({num_layers}, {hidden}, {aggr}): expected (3,), got {out.shape}"


def test_gnn_encode_produces_embeddings():
    data = _make_data(num_nodes=5, in_channels=8)
    model = GNNLinkPredictor(in_channels=8, hidden_channels=32)
    z = model.encode(data)
    assert z.shape == (5, 32), f"Expected shape (5, 32), got {z.shape}"
