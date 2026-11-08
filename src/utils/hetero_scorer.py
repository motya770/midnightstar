"""Save, load and score with heterogeneous gene–trait checkpoints.

``hetero_score_matrix`` is the one way to turn a checkpoint into a
(n_nodes × n_diseases) score matrix in ``node_list`` order, used by the
benchmark CLI and the Results page. ``honest=True`` rebuilds the graph without
the checkpoint's hidden gene–disease pairs, which is what the benchmark needs;
production scoring uses every known association as message-passing context.
"""
from __future__ import annotations

import glob
import os
import warnings
from dataclasses import asdict

import numpy as np
import torch

from src.models.hetero_gnn import HeteroGNNLinkPredictor
from src.utils.benchmark import HoldoutSplit
from src.utils.hetero_graph_features import HeteroBundle, nx_to_hetero_data

HETERO_MODEL_CLASS = "HeteroGNNLinkPredictor"
HETERO_MODEL_TYPE = "Hetero GNN"
CHECKPOINT_PATTERN = "hetero_gnn_*_checkpoint.pt"


def save_hetero_checkpoint(path, *, model, model_kwargs: dict, bundle: HeteroBundle, split: HoldoutSplit | None,
                           config, metrics: dict, graph_name: str, min_trait_genes: int = 1) -> dict:
    """Write the checkpoint and return its payload."""
    payload = {
        "model_class": HETERO_MODEL_CLASS,
        "model_type": HETERO_MODEL_TYPE,
        "model_kwargs": dict(model_kwargs),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": dict(metrics),
        "graph_name": graph_name,
        "train_config": asdict(config) if not isinstance(config, dict) else dict(config),
        "split": split.to_dict() if split is not None else None,
        "trait_list": list(bundle.trait_list),
        "min_trait_genes": min_trait_genes,
    }
    torch.save(payload, str(path))
    return payload


def load_hetero_checkpoint(path) -> dict:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if checkpoint.get("model_class") != HETERO_MODEL_CLASS:
        raise ValueError(f"{path} is not a hetero gene–disease checkpoint "
                         f"(model_class={checkpoint.get('model_class')!r})")
    return checkpoint


def build_hetero_model(checkpoint: dict) -> HeteroGNNLinkPredictor:
    model = HeteroGNNLinkPredictor(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def hidden_pairs(checkpoint: dict) -> set[tuple[str, str]]:
    split = checkpoint.get("split")
    if not split:
        return set()
    return {(g, d) for d, genes in split["hidden"].items() for g in genes}


def hetero_score_matrix(checkpoint: dict, graph, diseases, *, honest: bool = False, device: str = "cpu") -> np.ndarray:
    """Logit of every node of ``graph`` for each disease, shape (n_nodes, len(diseases)).

    Rows follow ``sorted(graph.nodes())``; non-gene nodes get zeros. A disease the
    model has no trait node for gets a zero column and a warning. Raises
    ``ValueError`` when the graph's traits differ from the checkpoint's.
    """
    diseases = list(diseases)
    exclude = hidden_pairs(checkpoint) if honest else None
    bundle = nx_to_hetero_data(graph, exclude_pairs=exclude, min_trait_genes=checkpoint.get("min_trait_genes", 1),
                               trait_list=checkpoint["trait_list"])
    model = build_hetero_model(checkpoint).to(device)
    data = bundle.data.to(device)
    with torch.no_grad():
        z_gene, z_trait = model.encode(data)
        known = [d for d in diseases if d in bundle.trait_to_idx]
        gene_scores = None
        if known:
            idx = torch.tensor([bundle.trait_to_idx[d] for d in known], device=z_gene.device)
            gene_scores = model.score_matrix(z_gene, z_trait, idx).cpu().numpy()
    missing = [d for d in diseases if d not in bundle.trait_to_idx]
    if missing:
        warnings.warn(f"{len(missing)} disease(s) have no trait node in this checkpoint and score 0: "
                      f"{missing[:5]}", UserWarning, stacklevel=2)

    node_list = sorted(graph.nodes())
    matrix = np.zeros((len(node_list), len(diseases)))
    if gene_scores is not None:
        rows = np.array([node_list.index(g) for g in bundle.node_list]) if len(node_list) != len(bundle.node_list) \
            else np.arange(len(node_list))
        cols = [j for j, d in enumerate(diseases) if d in bundle.trait_to_idx]
        matrix[np.ix_(rows, cols)] = gene_scores
    return matrix


def find_hetero_checkpoints(checkpoint_dir) -> list[str]:
    """Sorted paths of hetero checkpoints in ``checkpoint_dir`` (empty when it does not exist)."""
    if not os.path.isdir(str(checkpoint_dir)):
        return []
    return sorted(glob.glob(os.path.join(str(checkpoint_dir), CHECKPOINT_PATTERN)))
