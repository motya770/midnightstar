# tests/models/test_trainer.py
import torch
from torch_geometric.data import Data
from src.models.gnn import GNNLinkPredictor
from src.models.vae import GraphVAE
from src.models.trainer import TrainConfig, Trainer


def _make_data(num_nodes=20, in_channels=8):
    x = torch.randn(num_nodes, in_channels)
    # Create a denser graph so we have enough edges for splitting
    src = torch.arange(num_nodes - 1)
    dst = torch.arange(1, num_nodes)
    edge_index = torch.stack([src, dst], dim=0)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def test_trainer_runs_epochs():
    data = _make_data()
    model = GNNLinkPredictor(in_channels=8, hidden_channels=16, num_layers=2)
    config = TrainConfig(epochs=3, lr=0.01, patience=10)
    trainer = Trainer(model, config)
    history = trainer.train(data)
    assert len(history) == 3, f"Expected 3 history entries, got {len(history)}"


def test_trainer_early_stopping():
    data = _make_data()
    model = GNNLinkPredictor(in_channels=8, hidden_channels=16, num_layers=2)
    config = TrainConfig(epochs=100, lr=0.01, patience=5)
    trainer = Trainer(model, config)
    history = trainer.train(data)
    assert len(history) < 100, f"Expected early stopping before 100 epochs, got {len(history)}"


def test_trainer_returns_metrics():
    data = _make_data()
    model = GNNLinkPredictor(in_channels=8, hidden_channels=16, num_layers=2)
    config = TrainConfig(epochs=3, lr=0.01, patience=10)
    trainer = Trainer(model, config)
    trainer.train(data)
    metrics = trainer.evaluate(data)
    assert "auc_roc" in metrics, "metrics must contain auc_roc"
    assert "avg_precision" in metrics, "metrics must contain avg_precision"
    assert 0.0 <= metrics["auc_roc"] <= 1.0, f"auc_roc out of range: {metrics['auc_roc']}"
    assert 0.0 <= metrics["avg_precision"] <= 1.0, f"avg_precision out of range: {metrics['avg_precision']}"
