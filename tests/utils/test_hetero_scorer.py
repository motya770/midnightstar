# tests/utils/test_hetero_scorer.py
import networkx as nx
import numpy as np
import pytest
import torch

from src.models.hetero_gnn import HeteroGNNLinkPredictor
from src.models.hetero_trainer import HeteroTrainConfig, HeteroTrainer
from src.utils.benchmark import make_holdout
from src.utils.discovery import index_disease_genes
from src.utils.hetero_graph_features import nx_to_hetero_data
from src.utils.hetero_scorer import (
    HETERO_MODEL_CLASS,
    HETERO_MODEL_TYPE,
    find_hetero_checkpoints,
    hetero_encode,
    hetero_score_matrix,
    is_hetero_checkpoint_path,
    load_hetero_checkpoint,
    save_hetero_checkpoint,
    score_from_encoding,
)

GENES = [f"G{i:02d}" for i in range(24)]


def make_graph(extra_trait=False, extra_tissue=False):
    G = nx.Graph()
    expression = {"liver": 1.0, "brain": 0.5} if extra_tissue else {"liver": 1.0}
    for g in GENES:
        G.add_node(g, node_type="gene", expression=dict(expression))
    G.add_edges_from([(GENES[i], GENES[(i + 1) % 24]) for i in range(24)])
    for g in GENES[:12]:
        G.nodes[g].setdefault("gwas", {}).setdefault("disease", []).append({"trait": "Asthma", "score": 0.9})
    for g in GENES[8:16]:
        G.nodes[g].setdefault("gwas", {}).setdefault("trait", []).append({"trait": "BMI", "score": 0.5})
    if extra_trait:
        G.nodes["G20"]["gwas"] = {"trait": [{"trait": "Extra", "score": 0.4}]}
    return G


def train_and_save(tmp_path, **kwargs):
    G = make_graph()
    node_list = sorted(G.nodes())
    index = index_disease_genes(G, node_list)
    split = make_holdout(index)
    hidden = {(g, d) for d, genes in split.hidden.items() for g in genes}
    bundle = nx_to_hetero_data(G, exclude_pairs=hidden)
    torch.manual_seed(0)
    model_kwargs = dict(gene_in=bundle.data["gene"].x.size(1), num_traits=len(bundle.trait_list),
                        hidden_channels=8, dropout=0.0, **kwargs)
    model = HeteroGNNLinkPredictor(**model_kwargs)
    config = HeteroTrainConfig(epochs=2, early_stopping=False)
    trainer = HeteroTrainer(model, config)
    trainer.train(bundle, index, split, G)
    path = tmp_path / "hetero_gnn_20261108_000000_checkpoint.pt"
    save_hetero_checkpoint(path, model=model, model_kwargs=model_kwargs, bundle=bundle, split=split,
                           config=config, metrics={"val_mrr": 0.5}, graph_name="toy")
    return G, index, split, bundle, trainer, path


def test_checkpoint_round_trip_reproduces_score_matrix(tmp_path):
    G, index, split, bundle, trainer, path = train_and_save(tmp_path)
    checkpoint = load_hetero_checkpoint(path)
    assert checkpoint["model_class"] == HETERO_MODEL_CLASS
    assert checkpoint["model_type"] == HETERO_MODEL_TYPE
    assert checkpoint["trait_list"] == bundle.trait_list
    assert checkpoint["split"]["hidden"] == split.hidden
    assert checkpoint["graph_name"] == "toy" and checkpoint["metrics"] == {"val_mrr": 0.5}
    assert checkpoint["train_config"]["epochs"] == 2
    matrix = hetero_score_matrix(checkpoint, G, split.diseases, honest=True)
    assert matrix.shape == (len(bundle.node_list), 1)
    assert np.allclose(matrix, trainer.score_matrix(split.diseases), atol=1e-5)


def test_honest_mode_excludes_hidden_pairs_from_messages(tmp_path):
    G, index, split, bundle, trainer, path = train_and_save(tmp_path)
    checkpoint = load_hetero_checkpoint(path)
    honest = hetero_score_matrix(checkpoint, G, ["Asthma"], honest=True)
    production = hetero_score_matrix(checkpoint, G, ["Asthma"], honest=False)
    assert honest.shape == production.shape
    assert not np.allclose(honest, production)


def test_mismatched_trait_list_raises(tmp_path):
    _, _, _, _, _, path = train_and_save(tmp_path)
    checkpoint = load_hetero_checkpoint(path)
    with pytest.raises(ValueError, match="trait"):
        hetero_score_matrix(checkpoint, make_graph(extra_trait=True), ["Asthma"])


def test_unknown_disease_gets_zero_column_and_warning(tmp_path):
    G, _, _, bundle, _, path = train_and_save(tmp_path)
    checkpoint = load_hetero_checkpoint(path)
    with pytest.warns(UserWarning, match="Nonexistent"):
        matrix = hetero_score_matrix(checkpoint, G, ["Asthma", "Nonexistent"])
    assert matrix.shape == (len(bundle.node_list), 2)
    assert np.all(matrix[:, 1] == 0)
    assert np.any(matrix[:, 0] != 0)


def test_load_rejects_non_hetero_checkpoint(tmp_path):
    path = tmp_path / "gnn_20261108_000000_checkpoint.pt"
    torch.save({"model_class": "GNNLinkPredictor", "model_state": {}}, path)
    with pytest.raises(ValueError, match="hetero"):
        load_hetero_checkpoint(path)


def test_find_hetero_checkpoints_lists_only_hetero_files(tmp_path):
    for name in ["hetero_gnn_20261108_000001_checkpoint.pt", "gnn_20261030_012253_checkpoint.pt",
                 "hetero_gnn_20261108_000000_checkpoint.pt", "hetero_gnn_notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    found = find_hetero_checkpoints(tmp_path)
    assert [p.split("/")[-1] for p in found] == [
        "hetero_gnn_20261108_000000_checkpoint.pt", "hetero_gnn_20261108_000001_checkpoint.pt",
    ]
    assert find_hetero_checkpoints(tmp_path / "missing") == []


def test_feature_width_mismatch_raises_value_error(tmp_path):
    """A graph with a different gene-feature layout must be refused, not crash inside encode (RuntimeError)."""
    _, _, _, _, _, path = train_and_save(tmp_path)
    checkpoint = load_hetero_checkpoint(path)
    with pytest.raises(ValueError, match="feature"):
        hetero_score_matrix(checkpoint, make_graph(extra_tissue=True), ["Asthma"])


def test_encoding_is_reusable_across_disease_lists(tmp_path):
    """Encode once per graph, score any disease list from it: same numbers as hetero_score_matrix."""
    G, _, split, bundle, _, path = train_and_save(tmp_path)
    checkpoint = load_hetero_checkpoint(path)
    encoding = hetero_encode(checkpoint, G, honest=True)
    assert np.allclose(score_from_encoding(encoding, split.diseases),
                       hetero_score_matrix(checkpoint, G, split.diseases, honest=True))
    with pytest.warns(UserWarning, match="Nonexistent"):
        matrix = score_from_encoding(encoding, ["Asthma", "Nonexistent"])
    assert matrix.shape == (len(bundle.node_list), 2)
    assert np.all(matrix[:, 1] == 0) and np.any(matrix[:, 0] != 0)


def test_is_hetero_checkpoint_path_matches_only_the_naming_pattern():
    assert is_hetero_checkpoint_path("/models/hetero_gnn_20261108_013654_checkpoint.pt")
    assert not is_hetero_checkpoint_path("/models/gnn_20261030_012253_checkpoint.pt")
    assert not is_hetero_checkpoint_path("/models/hetero_gnn_notes.txt")
