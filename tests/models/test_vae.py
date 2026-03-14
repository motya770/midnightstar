# tests/models/test_vae.py
import torch
from src.models.vae import GraphVAE


def _make_inputs(num_nodes=5, in_channels=8):
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return x, edge_index


def test_vae_forward_shape():
    x, edge_index = _make_inputs(num_nodes=5, in_channels=8)
    model = GraphVAE(in_channels=8, hidden_channels=32, latent_dim=16)
    adj_pred, mu, logvar = model(x, edge_index)
    assert adj_pred.shape == (5, 5), f"Expected adj_pred shape (5, 5), got {adj_pred.shape}"
    assert mu.shape == (5, 16), f"Expected mu shape (5, 16), got {mu.shape}"
    assert logvar.shape == (5, 16), f"Expected logvar shape (5, 16), got {logvar.shape}"


def test_vae_link_scores():
    x, edge_index = _make_inputs(num_nodes=5, in_channels=8)
    model = GraphVAE(in_channels=8, hidden_channels=32, latent_dim=16)
    src = torch.tensor([0, 2])
    dst = torch.tensor([1, 3])
    scores = model.predict_links(x, edge_index, src, dst)
    assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
    assert (scores >= 0).all() and (scores <= 1).all(), "Scores should be in [0, 1]"


def test_vae_loss():
    x, edge_index = _make_inputs(num_nodes=5, in_channels=8)
    model = GraphVAE(in_channels=8, hidden_channels=32, latent_dim=16)
    loss = model.loss(x, edge_index)
    assert loss.item() > 0, "Loss should be positive"


def test_vae_get_embeddings():
    x, edge_index = _make_inputs(num_nodes=5, in_channels=8)
    model = GraphVAE(in_channels=8, hidden_channels=32, latent_dim=16)
    embeddings = model.get_embeddings(x, edge_index)
    assert embeddings.shape == (5, 16), f"Expected shape (5, 16), got {embeddings.shape}"
