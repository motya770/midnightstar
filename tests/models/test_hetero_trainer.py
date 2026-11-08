# tests/models/test_hetero_trainer.py
import copy
import math

import networkx as nx
import pytest
import torch

from src.models.hetero_gnn import HeteroGNNLinkPredictor
from src.models.hetero_trainer import HeteroTrainConfig, HeteroTrainer
from src.utils.benchmark import BenchmarkResult, make_holdout
from src.utils.discovery import index_disease_genes
from src.utils.hetero_graph_features import nx_to_hetero_data

GENES = [f"G{i:02d}" for i in range(30)]
ASTHMA = GENES[:12]                 # disease with 12 known genes: benchmark hides 2, validation hides 1
BMI = GENES[6:20]                   # a trait sharing genes with asthma
RARE = ["G25"]                      # a one-gene trait


def make_graph():
    G = nx.Graph()
    for g in GENES:
        G.add_node(g, node_type="gene", expression={"liver": 1.0})
    G.add_edges_from([(GENES[i], GENES[(i + 1) % 30]) for i in range(30)])   # ring
    G.add_edges_from([("G29", g) for g in GENES[:10]])                         # hub
    for g in ASTHMA:
        G.nodes[g].setdefault("gwas", {}).setdefault("disease", []).append({"trait": "Asthma", "score": 0.9})
    for g in BMI:
        G.nodes[g].setdefault("gwas", {}).setdefault("trait", []).append({"trait": "BMI", "score": 0.5})
    for g in RARE:
        G.nodes[g].setdefault("gwas", {}).setdefault("trait", []).append({"trait": "Rare", "score": 0.3})
    return G


def setup(**config_kwargs):
    G = make_graph()
    node_list = sorted(G.nodes())
    index = index_disease_genes(G, node_list)
    split = make_holdout(index)
    hidden_pairs = {(g, d) for d, genes in split.hidden.items() for g in genes}
    bundle = nx_to_hetero_data(G, exclude_pairs=hidden_pairs)
    torch.manual_seed(0)
    model = HeteroGNNLinkPredictor(gene_in=bundle.data["gene"].x.size(1), num_traits=len(bundle.trait_list),
                                   hidden_channels=16, dropout=0.0)
    config = HeteroTrainConfig(**{"epochs": 3, "patience": 2, **config_kwargs})
    trainer = HeteroTrainer(model, config)
    return G, index, split, bundle, trainer


def pairs(bundle, edges):
    return {(bundle.node_list[g], bundle.trait_list[t]) for g, t in edges.t().tolist()}


def test_trainer_runs_requested_epochs_and_records_history():
    G, index, split, bundle, trainer = setup(epochs=3, early_stopping=False)
    history = trainer.train(bundle, index, split, G)
    assert len(history["train_loss"]) == 3
    assert set(history) >= {"train_loss", "val_mrr", "val_recall50", "val_auc", "best_epoch"}
    assert all(0.0 <= v <= 1.0 for v in history["val_mrr"])


def test_hidden_and_validation_pairs_are_absent_from_training_edges():
    G, index, split, bundle, trainer = setup(epochs=1)
    trainer.train(bundle, index, split, G)
    train_pairs = pairs(bundle, trainer._train_edges)
    hidden = {(g, d) for d, genes in split.hidden.items() for g in genes}
    val = {(g, d) for d, genes in trainer.val_split.hidden.items() for g in genes}
    assert len(hidden) == 2 and len(val) == 1
    assert not train_pairs & hidden
    assert not train_pairs & val
    # every other association survives
    assert train_pairs == pairs(bundle, bundle.data["gene", "has", "trait"].edge_index) - val
    sup, msg = trainer._epoch_edges()
    assert pairs(bundle, sup) | pairs(bundle, msg) == train_pairs
    assert not pairs(bundle, sup) & pairs(bundle, msg)


def test_negatives_never_coincide_with_known_edges():
    G, index, split, bundle, trainer = setup(epochs=1)
    trainer.train(bundle, index, split, G)
    known = set(map(tuple, bundle.data["gene", "has", "trait"].edge_index.t().tolist()))
    pos = trainer._train_edges
    for _ in range(5):
        neg_gene, neg_trait = trainer._sample_negatives(pos)
        assert neg_gene.shape == neg_trait.shape
        assert neg_gene.numel() > 0
        assert not set(zip(neg_gene.tolist(), neg_trait.tolist())) & known


def test_negative_sampling_survives_a_dense_graph():
    """Every gene is a known gene of the only trait, so every gene-corrupted negative is filtered out."""
    G = nx.Graph()
    for i in range(12):
        G.add_node(f"G{i}", node_type="gene", expression={"liver": 1.0},
                   gwas={"disease": [{"trait": "Dense", "score": 0.9}]})
    G.add_edges_from([(f"G{i}", f"G{i+1}") for i in range(11)])
    node_list = sorted(G.nodes())
    index = index_disease_genes(G, node_list)
    bundle = nx_to_hetero_data(G)          # no benchmark hold-out: hidden genes would survive as negatives
    model = HeteroGNNLinkPredictor(gene_in=bundle.data["gene"].x.size(1), num_traits=1, hidden_channels=8)
    trainer = HeteroTrainer(model, HeteroTrainConfig(epochs=2, neg_trait=0))
    trainer._setup(bundle, index, None, G)
    neg_gene, neg_trait = trainer._sample_negatives(trainer._train_edges)
    assert neg_gene.numel() == 0 and neg_trait.numel() == 0
    history = trainer.train(bundle, index, None, G)
    assert len(history["train_loss"]) == 2
    assert all(math.isfinite(loss) for loss in history["train_loss"])


def test_early_stopping_restores_best_weights():
    G, index, split, bundle, trainer = setup(epochs=50, patience=1, early_stopping=True, lr=0.05)
    snapshots = []
    history = trainer.train(bundle, index, split, G,
                            on_epoch=lambda *_: snapshots.append(copy.deepcopy(trainer.model.state_dict())))
    assert len(history["train_loss"]) < 50
    best = snapshots[history["best_epoch"]]
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v, best[k])


def test_evaluate_returns_benchmark_result_with_hidden_ranks():
    G, index, split, bundle, trainer = setup(epochs=2)
    trainer.train(bundle, index, split, G)
    result = trainer.evaluate(split)
    assert isinstance(result, BenchmarkResult)
    assert list(result.hidden_ranks) == split.diseases
    assert set(result.hidden_ranks["Asthma"]) == set(split.hidden["Asthma"])
    matrix = trainer.score_matrix(split.diseases)
    assert matrix.shape == (len(bundle.node_list), 1)


def test_degree_aware_negatives_prefer_high_degree_genes():
    G, index, split, bundle, trainer = setup(epochs=1, neg_sampling="degree", neg_trait=0)
    trainer.train(bundle, index, split, G)
    pos = trainer._pairs_to_edges({("G25", "Rare")}).repeat(1, 2000)   # negatives (g, Rare) are never known pairs for g != G25
    neg_gene, _ = trainer._sample_negatives(pos)
    counts = torch.bincount(neg_gene, minlength=len(bundle.node_list))
    isolated = [bundle.node_to_idx[g] for g in GENES[20:25]]   # no gene–trait edges at all
    busy = bundle.node_to_idx["G10"]                           # in both Asthma and BMI
    assert counts[isolated].sum() == 0
    assert counts[busy] > 0


def test_eval_every_validates_on_a_schedule_and_counts_patience_in_evaluations():
    G, index, split, bundle, trainer = setup(epochs=7, eval_every=3, early_stopping=False)
    calls = []
    original = trainer._validate

    def counting_validate():
        calls.append(len(trainer_history["train_loss"]) if trainer_history else None)
        return original()

    trainer_history = None
    trainer._validate = counting_validate
    history = trainer.train(bundle, index, split, G)
    # epochs 0..6 with eval_every=3: evaluated after epochs 2, 5 and the final epoch 6
    assert history["eval_epochs"] == [2, 5, 6]
    assert len(history["val_mrr"]) == 7                      # one entry per epoch, carried forward between evaluations
    assert history["val_mrr"][2] == history["val_mrr"][3] == history["val_mrr"][4]
    assert history["val_mrr"][0] == history["val_mrr"][1] == 0.0  # nothing evaluated yet
    assert history["best_epoch"] in history["eval_epochs"]
    assert len(calls) == 3


def test_trainer_scores_align_to_graphs_with_non_gene_nodes():
    """The training CLI benchmarks trainer.score_matrix against sorted(graph.nodes()), which may hold
    disease nodes; the gene-only matrix must be aligned to that list instead of failing the shape check."""
    from src.utils.benchmark import run_benchmark
    from src.utils.hetero_scorer import align_gene_rows
    G = make_graph()
    G.add_node("DIS:1", node_type="disease")
    G.add_edge("G00", "DIS:1")
    node_list = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    index = index_disease_genes(G, node_list)
    split = make_holdout(index)
    bundle = nx_to_hetero_data(G, exclude_pairs={(g, d) for d, genes in split.hidden.items() for g in genes})
    model = HeteroGNNLinkPredictor(gene_in=bundle.data["gene"].x.size(1), num_traits=len(bundle.trait_list),
                                   hidden_channels=8)
    trainer = HeteroTrainer(model, HeteroTrainConfig(epochs=1, early_stopping=False))
    trainer.train(bundle, index, split, G)
    gene_scores = trainer.score_matrix(split.diseases)
    assert gene_scores.shape == (len(bundle.node_list), len(split.diseases))
    aligned = align_gene_rows(gene_scores, bundle.node_list, node_list)
    assert aligned.shape == (len(node_list), len(split.diseases))
    result = run_benchmark(aligned, G, node_list, node_to_idx, index, split)
    assert result.n_hidden == sum(len(g) for g in split.hidden.values())


def test_empty_validation_set_is_refused_when_early_stopping():
    """A disease with exactly 10 genes has 8 visible after benchmark masking, so nothing is left to validate on;
    early stopping on an always-zero MRR would silently keep the epoch-1 weights."""
    G = nx.Graph()
    for i in range(10):
        G.add_node(f"G{i}", node_type="gene", expression={"liver": 1.0},
                   gwas={"disease": [{"trait": "Ten", "score": 0.9}]})
    G.add_edges_from([(f"G{i}", f"G{i+1}") for i in range(9)])
    node_list = sorted(G.nodes())
    index = index_disease_genes(G, node_list)
    split = make_holdout(index)
    bundle = nx_to_hetero_data(G, exclude_pairs={(g, d) for d, genes in split.hidden.items() for g in genes})
    gene_in = bundle.data["gene"].x.size(1)
    with pytest.raises(ValueError, match="validation"):
        HeteroTrainer(HeteroGNNLinkPredictor(gene_in=gene_in, num_traits=1, hidden_channels=8),
                      HeteroTrainConfig(epochs=1)).train(bundle, index, split, G)
    with pytest.warns(UserWarning, match="validation"):
        history = HeteroTrainer(HeteroGNNLinkPredictor(gene_in=gene_in, num_traits=1, hidden_channels=8),
                                HeteroTrainConfig(epochs=2, early_stopping=False)).train(bundle, index, split, G)
    assert len(history["train_loss"]) == 2
