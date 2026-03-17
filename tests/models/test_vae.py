# tests/models/test_vae.py
import torch
from torch_geometric.data import Data
from src.models.vae import GraphVAE


def _make_data(num_nodes=5, in_channels=8):
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def test_vae_encode_shape():
    data = _make_data(num_nodes=5, in_channels=8)
    model = GraphVAE(in_channels=8, hidden_channels=32, latent_dim=16)
    z = model.encode(data)
    assert z.shape == (5, 16), f"Expected z shape (5, 16), got {z.shape}"
    assert model._mu.shape == (5, 16)
    assert model._logvar.shape == (5, 16)


def test_vae_decode_edge_level():
    data = _make_data(num_nodes=5, in_channels=8)
    model = GraphVAE(in_channels=8, hidden_channels=32, latent_dim=16)
    z = model.encode(data)
    src = torch.tensor([0, 2])
    dst = torch.tensor([1, 3])
    scores = model.decode(z, src, dst)
    assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
    assert (scores >= 0).all() and (scores <= 1).all(), "Scores should be in [0, 1]"


def test_vae_link_scores():
    data = _make_data(num_nodes=5, in_channels=8)
    model = GraphVAE(in_channels=8, hidden_channels=32, latent_dim=16)
    src = torch.tensor([0, 2])
    dst = torch.tensor([1, 3])
    scores = model.predict_links(data.x, data.edge_index, src, dst)
    assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
    assert (scores >= 0).all() and (scores <= 1).all(), "Scores should be in [0, 1]"


def test_vae_kl_loss():
    data = _make_data(num_nodes=5, in_channels=8)
    model = GraphVAE(in_channels=8, hidden_channels=32, latent_dim=16)
    model.encode(data)
    kl = model.kl_loss()
    assert kl.item() >= 0, "KL loss should be non-negative"


def test_vae_get_embeddings():
    data = _make_data(num_nodes=5, in_channels=8)
    model = GraphVAE(in_channels=8, hidden_channels=32, latent_dim=16)
    embeddings = model.get_embeddings(data.x, data.edge_index)
    assert embeddings.shape == (5, 16), f"Expected shape (5, 16), got {embeddings.shape}"
