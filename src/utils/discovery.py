"""Gene–disease discovery helpers.

Pure functions over the NetworkX graph and, when a protein–protein model is
loaded, its node embeddings. Used by the Results page and its export so the
logic can be tested without Streamlit.

Without ``scores``, ``predict_gene_disease_links`` ranks candidates by
guilt-by-association on the STRING network: the hypergeometric enrichment of
known disease genes among a gene's interaction partners. Embedding similarity
to the disease signature only breaks ties, which matters mostly for genes with
no known interactor. The Predictions page ranks by trait co-association by
default (``src/utils/coassociation.py``) and passes its matrix as ``scores``;
enrichment and then similarity break ties.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom

MAX_ENRICHMENT = 300.0  # cap for -log10 p when the p-value underflows


def index_disease_genes(graph, node_list) -> dict[str, dict]:
    """Map each GWAS trait to its category and the known genes with their scores.

    Only gene nodes contribute. A gene listed under the same trait more than once
    keeps its highest score. Trait names are matched exactly.
    """
    index: dict[str, dict] = {}
    for node in node_list:
        data = graph.nodes[node]
        if data.get("node_type") != "gene":
            continue
        for category, entries in data.get("gwas", {}).items():
            for entry in entries:
                record = index.setdefault(entry["trait"], {"category": category, "genes": {}})
                score = entry["score"]
                previous = record["genes"].get(node)
                if previous is None or score > previous:
                    record["genes"][node] = score
    return index


def select_diseases(index: dict[str, dict], categories, min_known_genes: int) -> list[str]:
    """Traits in the chosen categories with enough known genes, most genes first."""
    wanted = set(categories)
    chosen = [
        trait for trait, record in index.items()
        if record["category"] in wanted and len(record["genes"]) >= min_known_genes
    ]
    return sorted(chosen, key=lambda trait: (-len(index[trait]["genes"]), trait))


def _gene_adjacency(graph, node_list, node_to_idx, gene_mask: np.ndarray) -> csr_matrix:
    """Symmetric adjacency over gene–gene edges only, indexed like node_list."""
    rows, cols = [], []
    for u, v in graph.edges():
        iu, iv = node_to_idx.get(u), node_to_idx.get(v)
        if iu is None or iv is None or iu == iv or not (gene_mask[iu] and gene_mask[iv]):
            continue
        rows += [iu, iv]
        cols += [iv, iu]
    n = len(node_list)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def enrichment_matrix(graph, node_list, node_to_idx, index, diseases) -> np.ndarray:
    """Guilt-by-association score of every node for every disease, as (n_nodes, n_diseases).

    Entry (i, j) is -log10 of the hypergeometric probability that a gene with node
    i's degree has at least its observed number of known disease-j genes among its
    STRING partners (population = gene nodes). Nodes with no known partner score 0.
    Known genes are scored like any other node; callers mask them out.
    """
    n = len(node_list)
    gene_mask = np.array([graph.nodes[v].get("node_type") == "gene" for v in node_list], dtype=bool)
    n_genes = int(gene_mask.sum())
    adjacency = _gene_adjacency(graph, node_list, node_to_idx, gene_mask)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()

    matrix = np.zeros((n, len(diseases)))
    for j, disease in enumerate(diseases):
        known_idx = np.array([node_to_idx[g] for g in index[disease]["genes"] if g in node_to_idx], dtype=int)
        if known_idx.size == 0:
            continue
        indicator = np.zeros(n)
        indicator[known_idx] = 1.0
        known_neighbours = adjacency @ indicator
        hit = known_neighbours > 0
        if hit.any():
            log_p = hypergeom.logsf(known_neighbours[hit] - 1, n_genes, len(known_idx), degree[hit])
            matrix[hit, j] = -log_p / math.log(10)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=MAX_ENRICHMENT)
    return np.minimum(matrix, MAX_ENRICHMENT)


def predict_gene_disease_links(z_norm, node_list, node_to_idx, graph, index, diseases,
                               top_k: int, max_listed_interactors: int = 10, *, scores=None) -> list[dict]:
    """Rank candidate genes for each disease. One row per (disease, gene).

    Primary score: -log10 of the hypergeometric probability that a gene with its
    degree has at least its observed number of known disease genes among its
    STRING partners (population = gene nodes). Tie-break: cosine similarity to
    the disease signature, the normalised mean embedding of the known genes.
    Known genes and non-gene nodes are never returned.

    With ``scores``, a (len(node_list), len(diseases)) matrix from a trained
    gene–disease model, that score ranks first and enrichment then similarity
    break ties; each row then carries it as ``model_score`` (None otherwise).

    ``z_norm`` may be None (no protein–protein model loaded): ``similarity`` is
    then None in every row and ties keep ``node_list`` order.
    """
    if z_norm is not None:
        z_norm = z_norm.detach().cpu()
    diseases = list(diseases)
    if scores is not None:
        scores = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores, dtype=float)
        if scores.shape != (len(node_list), len(diseases)):
            raise ValueError(f"scores must be shaped (n_nodes={len(node_list)}, n_diseases={len(diseases)}), "
                             f"got {scores.shape}")
    gene_mask = np.array([graph.nodes[v].get("node_type") == "gene" for v in node_list], dtype=bool)
    adjacency = _gene_adjacency(graph, node_list, node_to_idx, gene_mask)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    enrichment_all = enrichment_matrix(graph, node_list, node_to_idx, index, diseases)

    rows: list[dict] = []
    for j, disease in enumerate(diseases):
        record = index[disease]
        known = [g for g in record["genes"] if g in node_to_idx]
        if not known:
            continue
        known_idx = np.array([node_to_idx[g] for g in known])
        enrichment = enrichment_all[:, j]

        similarity = None
        if z_norm is not None:
            signature = F.normalize(z_norm[torch.from_numpy(known_idx)].mean(dim=0, keepdim=True), dim=-1)
            similarity = (z_norm @ signature.T).squeeze(1).numpy()

        allowed = gene_mask.copy()
        allowed[known_idx] = False
        candidates = np.flatnonzero(allowed)
        if candidates.size == 0:
            continue
        keys = ([-similarity[candidates]] if similarity is not None else []) + [-enrichment[candidates]]
        if scores is not None:
            keys.append(-scores[candidates, j])
        order = candidates[np.lexsort(keys)][:top_k]

        known_set = set(known)
        for rank, idx in enumerate(order.tolist(), start=1):
            gene = node_list[idx]
            interactors = sorted(set(graph.neighbors(gene)) & known_set) if gene in graph else []
            rows.append({
                "disease": disease,
                "category": record["category"],
                "gene": gene,
                "gene_description": graph.nodes[gene].get("name", ""),
                "enrichment_score": round(float(enrichment[idx]), 4),
                "similarity": round(float(similarity[idx]), 4) if similarity is not None else None,
                "rank": rank,
                "n_known_genes": len(known),
                "known_interactors": len(interactors),
                "degree": int(degree[idx]),
                "known_interactor_genes": ";".join(interactors[:max_listed_interactors]),
                "model_score": round(float(scores[idx, j]), 4) if scores is not None else None,
            })
    return rows


def known_links_rows(index: dict[str, dict], diseases) -> list[dict]:
    """The known gene–disease associations that seeded each disease."""
    rows: list[dict] = []
    for disease in diseases:
        record = index[disease]
        for gene in sorted(record["genes"]):
            rows.append({
                "disease": disease,
                "category": record["category"],
                "gene": gene,
                "gwas_score": record["genes"][gene],
            })
    return rows
