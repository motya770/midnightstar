"""The hetero gene–disease training pipeline, shared by the CLI, the Model Training page and Modal.

``prepare_hetero_training`` makes the benchmark hold-out and converts the graph with the hidden
pairs removed; ``train_hetero`` trains a model on that bundle; ``benchmark_hetero`` scores it
against enrichment on the hidden pairs; ``save_hetero_run`` writes the checkpoint and records the
training run. ``hetero_training_payload`` / ``train_hetero_from_payload`` carry the prepared data to
another machine (a Modal GPU) and bring weights, history and metrics back, so a remote run leaves
exactly what a local run leaves once it is benchmarked and saved here.
"""
from __future__ import annotations

import datetime
import os
import pickle
import time
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import torch

from src.models.hetero_gnn import HeteroGNNLinkPredictor
from src.models.hetero_trainer import HeteroTrainConfig, HeteroTrainer
from src.utils.benchmark import HETERO_SCORERS, BenchmarkResult, HoldoutSplit, make_holdout, reference_scorers
from src.utils.discovery import index_disease_genes
from src.utils.hetero_graph_features import GENE_GENE, GENE_TRAIT, TRAIT_CATEGORIES, HeteroBundle, nx_to_hetero_data
from src.utils.hetero_scorer import (
    HETERO_MODEL_TYPE,
    HeteroEncoding,
    align_gene_rows,
    save_hetero_checkpoint,
    score_from_encoding,
)

BENCHMARK_MIN_KNOWN = 10   # make_holdout's default: disease traits with at least this many known genes


@dataclass
class HeteroPrepared:
    """A graph ready for training: the hetero bundle (hidden pairs removed) and the benchmark split."""
    bundle: HeteroBundle
    index: dict                      # index_disease_genes(graph, node_list), unmasked
    split: HoldoutSplit | None
    node_list: list[str]             # sorted(graph.nodes()): the benchmark's rows
    node_to_idx: dict[str, int]
    n_hidden: int                    # (gene, disease) pairs removed from the bundle

    @property
    def n_string(self) -> int:
        return self.bundle.data[GENE_GENE].edge_index.size(1) // 2

    @property
    def n_assoc(self) -> int:
        return self.bundle.data[GENE_TRAIT].edge_index.size(1)


def prepare_hetero_training(graph, *, holdout: bool = True, min_trait_genes: int = 1) -> HeteroPrepared:
    """Index the GWAS associations, hold out the benchmark pairs (unless ``holdout`` is off) and convert.

    Raises ``ValueError`` with the fix in the message when the network has no GWAS associations, when
    no disease has enough known genes for the benchmark protocol, or when ``min_trait_genes`` drops
    every association.
    """
    node_list = sorted(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    index = index_disease_genes(graph, node_list)
    if not index:
        raise ValueError("this network has no GWAS associations to learn from; rebuild it with "
                         "'Include disease associations (from GWAS)' ticked")
    split = make_holdout(index) if holdout else None
    if split is not None and not split.diseases:
        raise ValueError(f"no disease has {BENCHMARK_MIN_KNOWN}+ known genes in this network, so there is nothing to "
                         "hold out for the benchmark; untick the benchmark hold-out or build a larger network")
    hidden = {(g, d) for d, genes in split.hidden.items() for g in genes} if split is not None else set()
    bundle = nx_to_hetero_data(graph, exclude_pairs=hidden, min_trait_genes=min_trait_genes)
    if bundle.data[GENE_TRAIT].edge_index.size(1) == 0:
        raise ValueError(f"no gene–trait associations remain after dropping traits with fewer than "
                         f"{min_trait_genes} genes; lower that setting")
    return HeteroPrepared(bundle, index, split, node_list, node_to_idx, len(hidden))


def hetero_model_kwargs(bundle: HeteroBundle, *, hidden: int = 128, layers: int = 2, dropout: float = 0.2,
                        decoder: str = "dot") -> dict:
    return dict(gene_in=bundle.data["gene"].x.size(1), num_traits=len(bundle.trait_list),
                trait_in=len(TRAIT_CATEGORIES), hidden_channels=hidden, num_layers=layers,
                dropout=dropout, decoder=decoder)


def gene_only_graph(node_list) -> nx.Graph:
    """A stand-in for the training graph with just gene nodes: the trainer's validation only asks
    which nodes are genes, and the bundle's nodes all are, so this travels to a remote machine instead
    of the full network."""
    G = nx.Graph()
    G.add_nodes_from((n, {"node_type": "gene"}) for n in node_list)
    return G


@dataclass
class HeteroTrained:
    model: HeteroGNNLinkPredictor    # on CPU
    history: dict
    metrics: dict                    # epochs_run, best_epoch, val_mrr, val_recall50, val_auc, train_minutes


def train_hetero(bundle: HeteroBundle, index: dict, split: HoldoutSplit | None, model_kwargs: dict,
                 config: HeteroTrainConfig, on_epoch=None) -> HeteroTrained:
    """Build a seeded model and train it; the returned model is on the CPU with its best weights."""
    torch.manual_seed(config.seed)
    model = HeteroGNNLinkPredictor(**model_kwargs)
    trainer = HeteroTrainer(model, config)
    started = time.time()
    history = trainer.train(bundle, index, split, gene_only_graph(bundle.node_list), on_epoch=on_epoch)
    best = history["best_epoch"]
    metrics = {
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best,
        "val_mrr": history["val_mrr"][best],
        "val_recall50": history["val_recall50"][best],
        "val_auc": history["val_auc"][best],
        "train_minutes": round((time.time() - started) / 60, 2),
    }
    return HeteroTrained(trainer.model.cpu(), history, metrics)


def _gene_scores(model: HeteroGNNLinkPredictor, bundle: HeteroBundle, diseases) -> np.ndarray:
    """(n_genes × len(diseases)) logits with every bundle edge as message context."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        z_gene, z_trait = model.encode(bundle.data.to(device))
    return score_from_encoding(HeteroEncoding(bundle, model, z_gene, z_trait, list(bundle.node_list)), diseases)


def benchmark_hetero(model: HeteroGNNLinkPredictor, prepared: HeteroPrepared, graph) -> dict[str, BenchmarkResult]:
    """Enrichment, the model and their combinations on the hidden pairs; ``{}`` without a hold-out."""
    if prepared.split is None:
        return {}
    hetero_scores = align_gene_rows(_gene_scores(model, prepared.bundle, prepared.split.diseases),
                                    prepared.bundle.node_list, prepared.node_list)
    return reference_scorers(["enrichment", *HETERO_SCORERS], graph, prepared.node_list, prepared.node_to_idx,
                             prepared.index, prepared.split, hetero_scores=hetero_scores)


def describe_hetero_params(model_kwargs: dict, config: HeteroTrainConfig, *, holdout: bool,
                           gpu: str | None = None) -> str:
    """The settings string shown in the training-runs table."""
    text = (f"hidden={model_kwargs['hidden_channels']}, layers={model_kwargs['num_layers']}, "
            f"decoder={model_kwargs['decoder']}, dropout={model_kwargs['dropout']}, "
            f"neg={config.neg_gene}g+{config.neg_trait}t/{config.neg_sampling}, lr={config.lr}, "
            f"wd={config.weight_decay}, epochs={config.epochs}, seed={config.seed}, "
            f"holdout={'benchmark' if holdout else 'no'}")
    if gpu:
        text += f", gpu={gpu}"
    return text


@dataclass
class HeteroSaved:
    checkpoint_path: str
    run_id: int | None
    model_name: str


def save_hetero_run(manager, *, model: HeteroGNNLinkPredictor, model_kwargs: dict, prepared: HeteroPrepared,
                    config: HeteroTrainConfig, metrics: dict, benchmark: dict | None, graph_name: str,
                    min_trait_genes: int, params: str, out: str | None = None, save_run: bool = True) -> HeteroSaved:
    """Write ``hetero_gnn_<stamp>_checkpoint.pt`` under the manager's ``trained_models`` (or ``out``) and,
    unless ``save_run`` is off, record the run (AUC column = validation AUC, precision column = benchmark
    recall@50, or validation recall@50 without a hold-out)."""
    metrics = dict(metrics)
    if benchmark:
        metrics["benchmark"] = {name: result.to_dict() for name, result in benchmark.items()}
    model_name = f"hetero_gnn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = out or os.path.join(manager.data_dir, "trained_models", f"{model_name}_checkpoint.pt")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    model = model.cpu()
    save_hetero_checkpoint(path, model=model, model_kwargs=model_kwargs, bundle=prepared.bundle, split=prepared.split,
                           config=config, metrics=metrics, graph_name=graph_name, min_trait_genes=min_trait_genes)
    run_id = None
    if save_run:
        bench = metrics.get("benchmark", {}).get("hetero", {})
        run_id = manager.save_training_run(
            model_type=HETERO_MODEL_TYPE, parameters=params, graph_name=graph_name,
            nodes=len(prepared.bundle.node_list), edges=prepared.n_string, features=model_kwargs["gene_in"],
            epochs_run=metrics["epochs_run"], auc_roc=metrics["val_auc"],
            avg_precision=bench.get("recall@50", metrics["val_recall50"]),
            model=model, model_name=model_name,
        )
    return HeteroSaved(path, run_id, model_name)


# ---- remote training --------------------------------------------------------------------------------

def hetero_training_payload(prepared: HeteroPrepared, model_kwargs: dict, config: HeteroTrainConfig) -> dict:
    """Everything ``train_hetero_from_payload`` needs, as plain picklable objects (the device is chosen there)."""
    payload_config = {k: v for k, v in asdict(config).items() if k != "device"}
    return {
        "bundle": prepared.bundle,
        "index": prepared.index,
        "split": prepared.split.to_dict() if prepared.split is not None else None,
        "model_kwargs": dict(model_kwargs),
        "config": payload_config,
    }


def train_hetero_from_payload(payload: dict, device: str = "cpu", on_epoch=None) -> dict:
    """``train_hetero`` on a payload; returns the CPU state dict (pickled), history and metrics."""
    config = HeteroTrainConfig(**payload["config"], device=device)
    split = HoldoutSplit.from_dict(payload["split"]) if payload["split"] is not None else None
    trained = train_hetero(payload["bundle"], payload["index"], split, payload["model_kwargs"], config,
                           on_epoch=on_epoch)
    return {
        "model_state_dict": pickle.dumps(trained.model.state_dict()),
        "history": trained.history,
        "metrics": trained.metrics,
        "epochs_run": trained.metrics["epochs_run"],
        "device_used": device,
    }
