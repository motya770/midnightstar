# tests/models/test_training_regressions.py
"""Regression tests for the training-logic bugs found in the 2026-10 review."""
import copy

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import degree

import src.models.trainer as trainer_mod
from src.models.gnn import GNNLinkPredictor
from src.models.graph_transformer import GraphTransformerLinkPredictor
from src.models.trainer import TrainConfig, Trainer
from src.models.vae import GraphVAE


def _graph(n=40, extra=30, feats=4, seed=0):
    """Undirected graph whose edge_attr identifies its edge: (min, max, min+max, 0, 0).

    The last feature column is the full-graph degree / max degree, and
    ``degree_col`` points at it, mirroring ``nx_to_pyg_data``.
    """
    g = torch.Generator().manual_seed(seed)
    s = torch.cat([torch.arange(n - 1), torch.randint(0, n, (extra,), generator=g)])
    d = torch.cat([torch.arange(1, n), torch.randint(0, n, (extra,), generator=g)])
    keep = s != d
    lo, hi = torch.minimum(s[keep], d[keep]), torch.maximum(s[keep], d[keep])
    keys = torch.unique(lo * n + hi)
    lo, hi = keys // n, keys % n
    ei = torch.stack([torch.cat([lo, hi]), torch.cat([hi, lo])])
    mn, mx = ei.min(0).values, ei.max(0).values
    attr = torch.stack([mn, mx, mn + mx, torch.zeros_like(mn), torch.zeros_like(mn)], 1).float()
    x = torch.randn(n, feats, generator=g)
    deg = degree(ei[0], n)
    x[:, -1] = deg / deg.max()
    data = Data(x=x, edge_index=ei, edge_attr=attr, num_nodes=n)
    data.degree_col = feats - 1
    return data


def _undirected_keys(ei):
    return set(map(tuple, torch.stack([ei.min(0).values, ei.max(0).values], 1).tolist()))


def _positive_keys(split):
    mask = split.edge_label == 1
    return _undirected_keys(split.edge_label_index[:, mask])


# --------------------------------------------------------------------------- split

def test_split_edge_attr_rows_match_their_edges():
    data = _graph()
    tr = Trainer(GNNLinkPredictor(4, 8, 1), TrainConfig(epochs=1))
    tr._split_data(data)
    for split in (tr._train_data, tr._val_data, tr._test_data):
        ei, attr = split.edge_index, split.edge_attr
        expected = torch.stack([ei.min(0).values, ei.max(0).values], 1).float()
        assert torch.equal(attr[:, :2], expected), "edge_attr rows do not belong to their edge_index columns"


def test_split_supervision_edges_are_disjoint_from_message_graph():
    data = _graph()
    tr = Trainer(GNNLinkPredictor(4, 8, 1), TrainConfig(epochs=1))
    tr._split_data(data)
    msg = _undirected_keys(tr._train_data.edge_index)
    train_pos, val_pos, test_pos = (_positive_keys(s) for s in (tr._train_data, tr._val_data, tr._test_data))
    assert train_pos.isdisjoint(msg) and val_pos.isdisjoint(msg) and test_pos.isdisjoint(msg)
    assert train_pos.isdisjoint(val_pos) and train_pos.isdisjoint(test_pos) and val_pos.isdisjoint(test_pos)
    assert msg | train_pos | val_pos | test_pos == _undirected_keys(data.edge_index)
    all_edges = _undirected_keys(data.edge_index)
    for split in (tr._train_data, tr._val_data, tr._test_data):
        neg = split.edge_label_index[:, split.edge_label == 0]
        assert _undirected_keys(neg).isdisjoint(all_edges), "a real edge was sampled as a negative"


def test_split_degree_feature_uses_message_passing_graph():
    data = _graph()
    original_x = data.x.clone()
    tr = Trainer(GNNLinkPredictor(4, 8, 1), TrainConfig(epochs=1))
    tr._split_data(data)
    for split in (tr._train_data, tr._val_data, tr._test_data):
        msg_deg = degree(split.edge_index[0], split.num_nodes)
        expected = msg_deg / msg_deg.max().clamp(min=1)
        assert torch.allclose(split.x[:, data.degree_col], expected)
    assert not torch.allclose(tr._train_data.x[:, data.degree_col], original_x[:, data.degree_col])
    assert torch.equal(data.x, original_x), "input features must not be mutated"


def test_split_ratios_summing_past_one_raise():
    data = _graph()
    with pytest.raises(ValueError):
        Trainer(GNNLinkPredictor(4, 8, 1), TrainConfig(epochs=1, train_ratio=0.9, val_ratio=0.3)).train(data)


def test_split_leaving_no_test_edges_raises():
    n = 101  # 100 unique edges: int(90) + int(10) == 100 leaves nothing for test
    s, d = torch.arange(n - 1), torch.arange(1, n)
    data = Data(x=torch.randn(n, 4), edge_index=torch.stack([torch.cat([s, d]), torch.cat([d, s])]), num_nodes=n)
    with pytest.raises(ValueError, match="test"):
        Trainer(GNNLinkPredictor(4, 8, 1), TrainConfig(epochs=1, train_ratio=0.9, val_ratio=0.1)).train(data)


def test_same_seed_gives_same_split_and_different_seed_differs():
    data = _graph()

    def split(seed):
        tr = Trainer(GNNLinkPredictor(4, 8, 1), TrainConfig(epochs=1, seed=seed))
        tr._split_data(data)
        return tr._train_data.edge_label_index.clone(), tr._val_data.edge_label_index.clone()

    a, b, c = split(1), split(1), split(2)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


# --------------------------------------------------------------- early stopping

def test_early_stopping_restores_best_validation_weights(monkeypatch):
    data = _graph()
    torch.manual_seed(0)
    model = GNNLinkPredictor(4, 16, 2)
    tr = Trainer(model, TrainConfig(epochs=20, lr=0.05, early_stopping=True, patience=2))
    fake_val = iter([0.9, 0.8, 0.7, 0.6, 0.5])
    monkeypatch.setattr(tr, "_eval_auc_full", lambda split: next(fake_val))
    snapshots = {}
    history = tr.train(data, on_epoch=lambda epoch, loss, auc: snapshots.__setitem__(epoch, copy.deepcopy(model.state_dict())))
    assert len(history["train_loss"]) == 3
    assert history["best_epoch"] == 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, snapshots[0][key]), f"{key} is not the best-epoch weight"


# ------------------------------------------------------------------ mini-batch

@pytest.mark.parametrize("make_model", [
    lambda: GraphTransformerLinkPredictor(4, 16, 1, 2, rwse_dim=0, rwse_walk_length=0, dropout=0.3),
    lambda: GraphVAE(4, 16, 8, 1),
], ids=["transformer-dropout", "vae-noise"])
def test_mini_batch_backward_replays_the_same_stochastic_encode(make_model):
    data = _graph()
    torch.manual_seed(0)
    model = make_model()
    tr = Trainer(model, TrainConfig(epochs=1, mini_batch=True, batch_size=16, edge_dropout=0.3))
    tr._split_data(data)
    seen = []
    original = model.encode
    model.encode = lambda d: (lambda z: (seen.append(z.detach().clone()), z)[1])(original(d))
    tr._train_step_mini()
    assert len(seen) == 2
    assert torch.equal(seen[0], seen[1]), "encoder backward ran on a different stochastic forward pass"


def test_mini_batch_gradient_equals_full_batch_mean_loss_gradient(monkeypatch):
    data = _graph(n=30, extra=40)
    torch.manual_seed(0)
    model = GraphTransformerLinkPredictor(4, 16, 1, 2, rwse_dim=0, rwse_walk_length=0, dropout=0.0)
    tr = Trainer(model, TrainConfig(epochs=1, mini_batch=True, batch_size=8, edge_dropout=0.0))
    tr._split_data(data)
    train = tr._train_data
    pos_mask = train.edge_label == 1
    n_pos = int(pos_mask.sum())
    fixed_neg = torch.stack([torch.arange(n_pos) % 30, (torch.arange(n_pos) * 7 + 3) % 30])
    monkeypatch.setattr(trainer_mod, "negative_sampling",
                        lambda edge_index, num_nodes, num_neg_samples: fixed_neg[:, :num_neg_samples])
    monkeypatch.setattr(tr.optimizer, "step", lambda: None)

    src = torch.cat([train.edge_label_index[0][pos_mask], fixed_neg[0]])
    dst = torch.cat([train.edge_label_index[1][pos_mask], fixed_neg[1]])
    labels = torch.cat([torch.ones(n_pos), torch.zeros(n_pos)])
    model.train()
    model.zero_grad()
    z = model.encode(train)
    nn.functional.binary_cross_entropy_with_logits(model.decode_logits(z, src, dst), labels).backward()
    expected = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    tr._train_step_mini()
    for p, e in zip(model.parameters(), expected):
        assert torch.allclose(p.grad, e, rtol=1e-4, atol=1e-6), "mini-batch gradient is not the mean-loss gradient"


# ----------------------------------------------------------------------- models

def test_gnn_embeddings_are_not_clamped_non_negative():
    """A ReLU on the output layer forces z >= 0, hence z_i . z_j >= 0 and every link
    probability >= 0.5. Without it, embeddings (and logits) can take either sign."""
    torch.manual_seed(0)
    data = _graph()
    model = GNNLinkPredictor(4, 16, 2)
    model.eval()
    s, d = torch.randint(0, data.num_nodes, (2000,)), torch.randint(0, data.num_nodes, (2000,))
    with torch.no_grad():
        z = model.encode(data)
        logits = model.decode_logits(z, s, d)
    assert (z < 0).any(), "final-layer ReLU clamps every embedding to be non-negative"
    assert torch.allclose(torch.sigmoid(logits), model.decode(z, s, d))


def test_transformer_without_edge_dim_ignores_edge_attr():
    data = _graph()
    model = GraphTransformerLinkPredictor(4, 16, 1, 4, rwse_dim=0, rwse_walk_length=0, edge_dim=0)
    assert model.encode(data).shape == (data.num_nodes, 16)


def test_transformer_edge_attr_width_mismatch_raises_value_error():
    data = _graph()
    data.edge_attr = data.edge_attr[:, :3]
    model = GraphTransformerLinkPredictor(4, 16, 1, 4, rwse_dim=0, rwse_walk_length=0, edge_dim=5)
    with pytest.raises(ValueError, match="edge_attr"):
        model.encode(data)


def test_rwse_encode_is_deterministic_and_reuses_cache():
    data = _graph()
    torch.manual_seed(0)
    model = GraphTransformerLinkPredictor(4, 16, 1, 2, rwse_dim=4, rwse_walk_length=4, rwse_probes=16)
    calls = []
    original = model.rwse.compute_rw_diag
    model.rwse.compute_rw_diag = lambda ei, n: (calls.append(1), original(ei, n))[1]
    model.eval()
    with torch.no_grad():
        z1 = model.encode(data)
        z2 = model.encode(data)
    assert torch.equal(z1, z2), "RWSE probes are re-drawn on every encode"
    assert len(calls) == 1, "RWSE was recomputed instead of cached"


def test_rwse_cache_key_distinguishes_graphs_with_equal_sums():
    two_triangles = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]])
    six_cycle = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    assert GraphTransformerLinkPredictor._edge_hash(two_triangles) != GraphTransformerLinkPredictor._edge_hash(six_cycle)
