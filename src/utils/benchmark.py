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
        text = (f"{recalls}  median rank {self.median_rank:.0f}  MRR {self.mrr:.4f}  "
                f"({self.n_hidden} hidden genes, {self.n_diseases} diseases)")
        if "distinct_top10" in self.extra:
            text += f"  distinct top-10 {self.extra['distinct_top10']}, max rank-1 {self.extra['max_rank1']}"
        return text


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
        hubs = _hub_from_rows(rows, split)
    else:
        scores = scorer.detach().cpu().numpy() if isinstance(scorer, torch.Tensor) else np.asarray(scorer)
        if scores.shape != (len(node_list), len(split.diseases)):
            raise ValueError(f"Score matrix must be shaped (n_nodes={len(node_list)}, "
                             f"n_diseases={len(split.diseases)}), got {scores.shape}")
        hidden_ranks = _ranks_from_matrix(scores, graph, node_list, node_to_idx, split)
        hubs = hub_concentration(scores, graph, node_list, node_to_idx, split)
    result = summarize_ranks(hidden_ranks)
    result.extra.update(hubs)
    return result


# ---- hub concentration -----------------------------------------------------------

HUB_TOP_K = 10


def _hub_stats(top_lists: list[list[str]]) -> dict:
    """distinct genes across the top lists, and the most diseases any gene ranks first for."""
    firsts: dict[str, int] = {}
    for genes in top_lists:
        if genes:
            firsts[genes[0]] = firsts.get(genes[0], 0) + 1
    return {
        "distinct_top10": len({g for genes in top_lists for g in genes}),
        "max_rank1": max(firsts.values(), default=0),
    }


def hub_concentration(scores: np.ndarray, graph, node_list, node_to_idx, split: HoldoutSplit, k: int = HUB_TOP_K) -> dict:
    """How concentrated the top-k lists are across diseases: a scorer that puts the same
    hub genes first for every disease shows few distinct genes and a large max rank-1 count."""
    gene_mask = _gene_mask(graph, node_list)
    top_lists = []
    for j, disease in enumerate(split.diseases):
        allowed = gene_mask.copy()
        allowed[[node_to_idx[g] for g in split.visible[disease] if g in node_to_idx]] = False
        candidates = np.flatnonzero(allowed)
        order = candidates[np.argsort(-scores[candidates, j], kind="stable")[:k]]
        top_lists.append([node_list[i] for i in order])
    return _hub_stats(top_lists)


def _hub_from_rows(rows: list[dict], split: HoldoutSplit, k: int = HUB_TOP_K) -> dict:
    per_disease: dict[str, list[tuple[int, str]]] = {d: [] for d in split.diseases}
    for row in rows:
        if row["disease"] in per_disease and row["rank"] <= k:
            per_disease[row["disease"]].append((row["rank"], row["gene"]))
    return _hub_stats([[g for _, g in sorted(v)] for v in per_disease.values()])


# ---- combining two score matrices ------------------------------------------------

def _column_ranks(matrix: np.ndarray) -> np.ndarray:
    return np.column_stack([rankdata(-matrix[:, j], method="average") for j in range(matrix.shape[1])]) \
        if matrix.shape[1] else matrix


def rank_average(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Negative mean of the two per-column descending ranks (higher is better)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return -(_column_ranks(a) + _column_ranks(b)) / 2.0


def zscore_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sum of per-column z-scores; a constant column contributes nothing."""
    def z(m):
        m = np.asarray(m, dtype=float)
        std = m.std(axis=0, keepdims=True)
        return np.where(std > 1e-9, (m - m.mean(axis=0, keepdims=True)) / np.maximum(std, 1e-9), 0.0)
    return z(a) + z(b)


def fallback(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """``primary`` where it is positive; elsewhere ``secondary``, rescaled per column to sit
    strictly below the smallest positive primary score while keeping its own order."""
    primary, secondary = np.asarray(primary, dtype=float), np.asarray(secondary, dtype=float)
    out = primary.copy()
    for j in range(primary.shape[1]):
        hit = primary[:, j] > 0
        floor = primary[hit, j].min() if hit.any() else 0.0
        sec = secondary[~hit, j]
        if sec.size:
            span = sec.max() - sec.min()
            scaled = (sec - sec.min()) / span if span > 0 else np.zeros_like(sec)
            out[~hit, j] = floor - 1.0 + 0.999 * scaled
    return out


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


EMBEDDING_SCORERS = ("cosine", "centred-cosine")
HETERO_SCORERS = ("hetero", "hetero+enrichment", "hetero+enrichment-z", "enrichment>hetero")
REFERENCE_SCORERS = ("enrichment",) + EMBEDDING_SCORERS + HETERO_SCORERS


def reference_scorers(names, graph, node_list, node_to_idx, index: dict[str, dict], split: HoldoutSplit,
                      z=None, hetero_scores=None) -> dict[str, BenchmarkResult]:
    """Benchmark the named scorers.

    ``enrichment`` is the production ranking; ``cosine`` / ``centred-cosine`` need the
    protein–protein embeddings ``z``; ``hetero`` and its combinations with enrichment
    (rank average, z-score sum, enrichment with the model as fallback for genes with
    no known partner) need the hetero model's ``hetero_scores`` matrix.
    """
    from src.utils.discovery import enrichment_matrix, predict_gene_disease_links

    unknown = [n for n in names if n not in REFERENCE_SCORERS]
    if unknown:
        raise ValueError(f"unknown scorer(s) {unknown}; choose from {list(REFERENCE_SCORERS)}")
    needs_z = [n for n in names if n in EMBEDDING_SCORERS]
    if needs_z and z is None:
        raise ValueError(f"{needs_z} need model embeddings (z); pass a checkpoint")
    needs_hetero = [n for n in names if n in HETERO_SCORERS]
    if needs_hetero and hetero_scores is None:
        raise ValueError(f"{needs_hetero} need the hetero model's score matrix; pass a hetero checkpoint")

    masked = mask_index(index, split)
    enrichment = None
    if any(n in HETERO_SCORERS and n != "hetero" for n in names):
        enrichment = enrichment_matrix(graph, node_list, node_to_idx, masked, split.diseases)
    if hetero_scores is not None:
        hetero_scores = np.asarray(hetero_scores, dtype=float)

    results = {}
    for name in names:
        if name == "enrichment":
            z_norm = F.normalize(z.detach().cpu().float(), dim=-1) if z is not None else torch.zeros(len(node_list), 1)
            results[name] = run_benchmark(predict_gene_disease_links, graph, node_list, node_to_idx, index, split,
                                          z_norm=z_norm)
            continue
        if name in EMBEDDING_SCORERS:
            matrix = centroid_cosine_matrix(z, node_list, node_to_idx, masked, split.diseases,
                                            centre=(name == "centred-cosine"))
        elif name == "hetero":
            matrix = hetero_scores
        elif name == "hetero+enrichment":
            matrix = rank_average(hetero_scores, enrichment)
        elif name == "hetero+enrichment-z":
            matrix = zscore_sum(hetero_scores, enrichment)
        else:  # enrichment>hetero
            matrix = fallback(enrichment, hetero_scores)
        results[name] = run_benchmark(matrix, graph, node_list, node_to_idx, index, split)
    return results
