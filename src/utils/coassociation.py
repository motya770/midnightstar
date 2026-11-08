"""Trait co-association: rank disease genes by their GWAS associations with similar traits.

A gene's associations with *other* GWAS traits that share genes with the disease
(pleiotropy) predict hidden disease genes far better than anything the STRING
network offers. The scorer builds a gene × trait matrix of ``log1p(GWAS score)``
over every association in the index, finds the traits most similar to the
disease by cosine similarity of their columns (the disease's own column
excluded), keeps the ``top_traits`` most similar with positive similarity, and
scores each gene by the similarity-weighted sum of its associations with those
traits. No network, no training. On the held-out benchmark
(``src/utils/benchmark.py``, 219 diseases, 2,648 hidden genes) it recovers 26 %
of the hidden genes in the top 50 against 4 % for network enrichment, with no
hub genes; part of that gain comes from GWAS Catalog alias traits ("diabetes
mellitus" next to "type 2 diabetes mellitus"), and the strict variant, which
drops traits sharing a word with the disease name, still recovers 19 %.

The scorer sees only the index it is given: the benchmark passes the masked
index, the Predictions page the full one. Ranking goes through
``predict_gene_disease_links(..., scores=matrix)``, so enrichment (then
embedding similarity) breaks ties as before.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, diags

DEFAULT_TOP_TRAITS = 20
WEIGHTING = "log1p(GWAS score)"
NAME_STOP_WORDS = frozenset(
    "and of the in to type mellitus disease disorder measurement level levels adjusted for or with "
    "response status count syndrome risk onset age at non".split()
)
_WORD = re.compile(r"[a-z0-9]+")


def name_tokens(name: str) -> set[str]:
    """The words of a trait name that count for the name-sharing rule: lower-case, alphanumeric,
    at least four characters and not in ``NAME_STOP_WORDS``."""
    return {w for w in _WORD.findall(name.lower()) if len(w) >= 4 and w not in NAME_STOP_WORDS}


def shares_name_token(a: str, b: str) -> bool:
    """True when the two trait names share a qualifying word (a word rule, not a synonym rule)."""
    return bool(name_tokens(a) & name_tokens(b))


@dataclass
class CoassociationScores:
    """Co-association scores of every node for each disease, with the traits behind them.

    ``matrix`` is (len(node_list), len(diseases)) in ``node_list`` order; ``similar_traits[j]`` lists
    the kept traits of ``diseases[j]`` as (trait, similarity), most similar first."""

    matrix: np.ndarray
    diseases: list[str]
    similar_traits: list[list[tuple[str, float]]]
    top_traits: int
    exclude_name_sharing: bool
    index: dict = field(repr=False)


def _trait_matrix(index: dict[str, dict], node_list, node_to_idx) -> tuple[csr_matrix, list[str]]:
    """Sparse (n_nodes × n_traits) matrix of log1p(GWAS score), traits in sorted order."""
    traits = sorted(index)
    rows, cols, values = [], [], []
    for col, trait in enumerate(traits):
        for gene, score in index[trait]["genes"].items():
            row = node_to_idx.get(gene)
            if row is not None:
                rows.append(row)
                cols.append(col)
                values.append(math.log1p(max(float(score), 0.0)))
    matrix = csr_matrix((values, (rows, cols)), shape=(len(node_list), len(traits)))
    return matrix, traits


def coassociation_scores(index: dict[str, dict], node_list, node_to_idx, diseases, *,
                         top_traits: int = DEFAULT_TOP_TRAITS,
                         exclude_name_sharing: bool = False) -> CoassociationScores:
    """Score every node of ``node_list`` for each disease (a key of ``index``) by trait co-association.

    For disease *d*: cosine similarity between its column of the log1p-weighted gene × trait matrix
    and every other trait's column (its own set to 0; with ``exclude_name_sharing`` also every trait
    sharing a word with its name); keep the ``top_traits`` most similar traits with similarity > 0;
    score(g, d) = Σ over those traits t of similarity(d, t) × log1p(score(g, t)). Genes with no
    association to any kept trait, and non-gene nodes, score 0. A disease missing from ``index``
    raises ``ValueError``.
    """
    diseases = list(diseases)
    missing = [d for d in diseases if d not in index]
    if missing:
        raise ValueError(f"disease(s) not in the GWAS index: {missing[:5]}")
    matrix, traits = _trait_matrix(index, node_list, node_to_idx)
    trait_col = {t: i for i, t in enumerate(traits)}
    disease_cols = np.array([trait_col[d] for d in diseases], dtype=int)

    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    normalised = matrix @ diags(1.0 / norms)
    similarity = (normalised.T @ normalised[:, disease_cols]).toarray()   # (n_traits, n_diseases)
    similarity[disease_cols, np.arange(len(diseases))] = 0.0
    if exclude_name_sharing:
        tokens = [name_tokens(t) for t in traits]
        for j, disease in enumerate(diseases):
            own = name_tokens(disease)
            if own:
                for t in np.flatnonzero(similarity[:, j] > 0):
                    if tokens[t] & own:
                        similarity[t, j] = 0.0

    weights = np.zeros_like(similarity)
    similar_traits: list[list[tuple[str, float]]] = []
    for j in range(len(diseases)):
        order = np.argsort(-similarity[:, j], kind="stable")[:top_traits]
        kept = [t for t in order if similarity[t, j] > 0]
        weights[kept, j] = similarity[kept, j]
        similar_traits.append([(traits[t], float(similarity[t, j])) for t in kept])

    scores = np.asarray(matrix @ weights, dtype=float)
    return CoassociationScores(matrix=scores, diseases=diseases, similar_traits=similar_traits,
                               top_traits=top_traits, exclude_name_sharing=exclude_name_sharing, index=index)
