# tests/models/test_trainer.py
import torch
from torch_geometric.data import Data
from src.models.gnn import GNNLinkPredictor
from src.models.trainer import TrainConfig, Trainer


def _make_data(num_nodes=20, in_channels=8):
    x = torch.randn(num_nodes, in_channels)
    src = torch.arange(num_nodes - 1)
    dst = torch.arange(1, num_nodes)
    edge_index = torch.stack([src, dst], dim=0)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def test_trainer_runs_epochs():
    data = _make_data()
    model = GNNLinkPredictor(in_channels=8, hidden_channels=16, num_layers=2)
    config = TrainConfig(epochs=3, lr=0.01, train_ratio=0.6, val_ratio=0.2)
    trainer = Trainer(model, config)
    history = trainer.train(data)
    assert len(history["train_loss"]) == 3
    assert "val_auc" in history


def test_trainer_early_stopping():
    data = _make_data()
    model = GNNLinkPredictor(in_channels=8, hidden_channels=16, num_layers=2)
    config = TrainConfig(epochs=100, lr=0.01, early_stopping=True, patience=3)
    trainer = Trainer(model, config)
    history = trainer.train(data)
    assert len(history["train_loss"]) <= 100


def test_trainer_returns_metrics():
    data = _make_data()
    model = GNNLinkPredictor(in_channels=8, hidden_channels=16, num_layers=2)
    config = TrainConfig(epochs=5, lr=0.01)
    trainer = Trainer(model, config)
    trainer.train(data)
    metrics = trainer.evaluate(data)
    assert "auc_roc" in metrics
    assert "avg_precision" in metrics
    assert 0.0 <= metrics["auc_roc"] <= 1.0


def test_trainer_on_epoch_callback():
    data = _make_data()
    model = GNNLinkPredictor(in_channels=8, hidden_channels=16, num_layers=2)
    config = TrainConfig(epochs=3, lr=0.01)
    trainer = Trainer(model, config)
    callbacks = []
    trainer.train(data, on_epoch=lambda epoch, loss, auc: callbacks.append((epoch, loss, auc)))
    assert len(callbacks) == 3
    assert all(isinstance(c[1], float) for c in callbacks)
