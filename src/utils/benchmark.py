"""Held-out disease-gene benchmark.

Hide a fifth of each disease's known GWAS genes, score every gene from the rest,
and ask where the hidden genes land. The protocol is fixed so that numbers stay
comparable across scorers and sessions:

* traits of the chosen categories (default "disease") with at least
  ``min_known_genes`` known genes, most genes first (``select_diseases`` order);
* ``numpy.random.default_rng(seed)`` permutes each trait's sorted gene list in
  that order and hides the first ``max(1, n // 5)`` genes (for the default 20%);
* candidates are gene nodes that are not among the visible known genes;
* metrics are pooled over hidden genes (recall@k, median rank), except MRR,
  which is the mean over diseases of the reciprocal rank of the best hidden gene.

A scorer is either a callable with the signature of
``discovery.predict_gene_disease_links`` (it receives the masked index, so it
cannot see the hidden genes) or a precomputed score matrix of shape
(n_nodes, n_diseases) in ``split.diseases`` order. Ties in a matrix get their
average rank; a callable scorer supplies its own ranks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

from src.utils.discovery import select_diseases

RECALL_KS = (50, 200)


@dataclass(frozen=True)
class HoldoutSplit:
    """Which known genes a scorer may see per disease, and which it must recover."""

    visible: dict[str, list[str]]
    hidden: dict[str, list[str]]
    categories: tuple[str, ...] = ("disease",)
    min_known_genes: int = 10
    hidden_fraction: float = 0.2
    seed: int = 0

    @property
    def diseases(self) -> list[str]:
        return list(self.hidden)

    def to_dict(self) -> dict:
        return {
            "categories": list(self.categories),
            "min_known_genes": self.min_known_genes,
            "hidden_fraction": self.hidden_fraction,
            "seed": self.seed,
            "visible": {d: list(g) for d, g in self.visible.items()},
            "hidden": {d: list(g) for d, g in self.hidden.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "HoldoutSplit":
        return cls(
            visible={d: list(g) for d, g in payload["visible"].items()},
            hidden={d: list(g) for d, g in payload["hidden"].items()},
            categories=tuple(payload.get("categories", ("disease",))),
            min_known_genes=int(payload.get("min_known_genes", 10)),
            hidden_fraction=float(payload.get("hidden_fraction", 0.2)),
            seed=int(payload.get("seed", 0)),
        )


def make_holdout(index: dict[str, dict], categories=("disease",), min_known_genes: int = 10,
                 hidden_fraction: float = 0.2, seed: int = 0) -> HoldoutSplit:
    """Hide ``hidden_fraction`` of each eligible trait's known genes, reproducibly."""
    rng = np.random.default_rng(seed)
    visible, hidden = {}, {}
    for disease in select_diseases(index, categories, min_known_genes):
        genes = sorted(index[disease]["genes"])
        perm = rng.permutation(len(genes))
        n_hidden = max(1, int(len(genes) * hidden_fraction))
        hidden[disease] = [genes[i] for i in perm[:n_hidden]]
        visible[disease] = [genes[i] for i in perm[n_hidden:]]
    return HoldoutSplit(visible=visible, hidden=hidden, categories=tuple(categories),
                        min_known_genes=min_known_genes, hidden_fraction=hidden_fraction, seed=seed)


def mask_index(index: dict[str, dict], split: HoldoutSplit) -> dict[str, dict]:
    """A copy of ``index`` in which split diseases keep only their visible genes."""
    masked = {}
    for trait, record in index.items():
        if trait in split.visible:
            keep = set(split.visible[trait])
            genes = {g: s for g, s in record["genes"].items() if g in keep}
        else:
            genes = dict(record["genes"])
        masked[trait] = {"category": record["category"], "genes": genes}
    return masked


@dataclass
class BenchmarkResult:
    hidden_ranks: dict[str, dict[str, float]]
    recall: dict[int, float]
    median_rank: float
    mrr: float
    n_diseases: int
    n_hidden: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {f"recall@{k}": v for k, v in self.recall.items()}
        out.update({"median_rank": self.median_rank, "mrr": self.mrr,
                    "n_diseases": self.n_diseases, "n_hidden": self.n_hidden})
        out.update(self.extra)
        return out

    def summary(self) -> str:
        recalls = "  ".join(f"recall@{k} {v:.4f}" for k, v in self.recall.items())
        return (f"{recalls}  median rank {self.median_rank:.0f}  MRR {self.mrr:.4f}  "
                f"({self.n_hidden} hidden genes, {self.n_diseases} diseases)")


def summarize_ranks(hidden_ranks: dict[str, dict[str, float]], ks=RECALL_KS) -> BenchmarkResult:
    """Pool hidden-gene ranks into recall@k, median rank and per-disease MRR."""
    all_ranks = np.array([r for ranks in hidden_ranks.values() for r in ranks.values()], dtype=float)
    best = [min(ranks.values()) for ranks in hidden_ranks.values() if ranks]
    if all_ranks.size == 0:
        raise ValueError("No hidden genes to evaluate")
    return BenchmarkResult(
        hidden_ranks=hidden_ranks,
        recall={k: float((all_ranks <= k).mean()) for k in ks},
        median_rank=float(np.median(all_ranks)),
        mrr=float(np.mean([1.0 / r for r in best])),
        n_diseases=len(best),
        n_hidden=int(all_ranks.size),
    )


def _gene_mask(graph, node_list) -> np.ndarray:
    return np.array([graph.nodes[v].get("node_type") == "gene" for v in node_list], dtype=bool)


def _ranks_from_matrix(scores: np.ndarray, graph, node_list, node_to_idx, split: HoldoutSplit) -> dict:
    gene_mask = _gene_mask(graph, node_list)
    hidden_ranks: dict[str, dict[str, float]] = {}
    for j, disease in enumerate(split.diseases):
        allowed = gene_mask.copy()
        allowed[[node_to_idx[g] for g in split.visible[disease] if g in node_to_idx]] = False
        candidates = np.flatnonzero(allowed)
        ranks = rankdata(-scores[candidates, j], method="average")
        rank_of = dict(zip(candidates.tolist(), ranks.tolist()))
        hidden_ranks[disease] = {g: rank_of[node_to_idx[g]] for g in split.hidden[disease] if g in node_to_idx}
    return hidden_ranks


def _ranks_from_links(rows: list[dict], split: HoldoutSplit) -> dict:
    rank_of = {(row["disease"], row["gene"]): row["rank"] for row in rows}
    hidden_ranks: dict[str, dict[str, float]] = {}
    for disease in split.diseases:
        ranks = {}
        for gene in split.hidden[disease]:
            if (disease, gene) not in rank_of:
                raise ValueError(f"Scorer returned no rank for hidden gene {gene!r} of {disease!r}; "
                                 f"it must rank every candidate (top_k = number of nodes)")
            ranks[gene] = rank_of[(disease, gene)]
        hidden_ranks[disease] = ranks
    return hidden_ranks


def run_benchmark(scorer, graph, node_list, node_to_idx, index: dict[str, dict],
                  split: HoldoutSplit | None = None, *, z_norm=None) -> BenchmarkResult:
    """Evaluate a scorer on the hold-out split (made with defaults when not given).

    ``scorer`` is a callable ``(z_norm, node_list, node_to_idx, graph, index, diseases,
    top_k) -> rows`` like ``predict_gene_disease_links``, or a (n_nodes, n_diseases)
    array/tensor of scores in ``split.diseases`` order.
    """
    if split is None:
        split = make_holdout(index)
    if callable(scorer):
        masked = mask_index(index, split)
        rows = scorer(z_norm, node_list, node_to_idx, graph, masked, split.diseases, len(node_list))
        hidden_ranks = _ranks_from_links(rows, split)
    else:
        scores = scorer.detach().cpu().numpy() if isinstance(scorer, torch.Tensor) else np.asarray(scorer)
        if scores.shape != (len(node_list), len(split.diseases)):
            raise ValueError(f"Score matrix must be shaped (n_nodes={len(node_list)}, "
                             f"n_diseases={len(split.diseases)}), got {scores.shape}")
        hidden_ranks = _ranks_from_matrix(scores, graph, node_list, node_to_idx, split)
    return summarize_ranks(hidden_ranks)


# ---- reference scorers ---------------------------------------------------------

def centroid_cosine_matrix(z, node_list, node_to_idx, index: dict[str, dict], diseases,
                           centre: bool = False) -> np.ndarray:
    """Cosine similarity of every node to each disease signature (mean of its known genes).

    With ``centre=True`` the mean embedding over all nodes is subtracted first,
    which removes the shared direction that otherwise makes hubs look like
    everything.
    """
    z = z.detach().cpu().float()
    if centre:
        z = z - z.mean(dim=0, keepdim=True)
    z_norm = F.normalize(z, dim=-1)
    signatures = []
    for disease in diseases:
        idx = [node_to_idx[g] for g in index[disease]["genes"] if g in node_to_idx]
        signatures.append(F.normalize(z_norm[idx].mean(dim=0), dim=0))
    return (z_norm @ torch.stack(signatures, dim=1)).numpy()


REFERENCE_SCORERS = ("enrichment", "cosine", "centred-cosine")


def reference_scorers(names, graph, node_list, node_to_idx, index: dict[str, dict], split: HoldoutSplit,
                      z=None) -> dict[str, BenchmarkResult]:
    """Benchmark the named baselines: the production enrichment ranking, centroid
    cosine on raw embeddings and centroid cosine on mean-centred embeddings."""
    from src.utils.discovery import predict_gene_disease_links

    unknown = [n for n in names if n not in REFERENCE_SCORERS]
    if unknown:
        raise ValueError(f"unknown scorer(s) {unknown}; choose from {list(REFERENCE_SCORERS)}")
    needs_z = [n for n in names if n != "enrichment"]
    if needs_z and z is None:
        raise ValueError(f"{needs_z} need model embeddings (z); pass a checkpoint")

    masked = mask_index(index, split)
    results = {}
    for name in names:
        if name == "enrichment":
            z_norm = F.normalize(z.detach().cpu().float(), dim=-1) if z is not None else torch.zeros(len(node_list), 1)
            results[name] = run_benchmark(predict_gene_disease_links, graph, node_list, node_to_idx, index, split,
                                          z_norm=z_norm)
        else:
            matrix = centroid_cosine_matrix(z, node_list, node_to_idx, masked, split.diseases,
                                            centre=(name == "centred-cosine"))
            results[name] = run_benchmark(matrix, graph, node_list, node_to_idx, index, split)
    return results
