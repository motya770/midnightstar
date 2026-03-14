# tests/models/test_graph_transformer.py
import torch
from torch_geometric.data import Data
from src.models.graph_transformer import GraphTransformerLinkPredictor


def _make_data(num_nodes=5, in_channels=8):
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def test_transformer_forward_shape():
    data = _make_data()
    model = GraphTransformerLinkPredictor(in_channels=8, hidden_channels=32, num_heads=4)
    src = torch.tensor([0, 2])
    dst = torch.tensor([1, 3])
    out = model(data, src, dst)
    assert out.shape == (2,), f"Expected shape (2,), got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "Output values should be in [0, 1]"


def test_transformer_encode():
    data = _make_data(num_nodes=5, in_channels=8)
    model = GraphTransformerLinkPredictor(in_channels=8, hidden_channels=32, num_heads=4)
    z = model.encode(data)
    assert z.shape == (5, 32), f"Expected shape (5, 32), got {z.shape}"


def test_transformer_different_configs():
    data = _make_data(num_nodes=6, in_channels=4)
    src = torch.tensor([0, 1])
    dst = torch.tensor([3, 4])
    for num_layers, hidden, heads in [(1, 8, 2), (2, 16, 4)]:
        model = GraphTransformerLinkPredictor(
            in_channels=4, hidden_channels=hidden, num_layers=num_layers, num_heads=heads
        )
        out = model(data, src, dst)
        assert out.shape == (2,), f"Config ({num_layers}, {hidden}, {heads}): expected (2,), got {out.shape}"
