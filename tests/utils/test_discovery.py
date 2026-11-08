# tests/utils/test_discovery.py
import math

import networkx as nx
import pytest
import torch
import torch.nn.functional as F

from src.utils.discovery import (
    index_disease_genes,
    known_links_rows,
    predict_gene_disease_links,
    select_diseases,
)


def make_graph():
    """Seven genes plus one disease node.

    Asthma genes: A, B. Height genes: B, C.
    D neighbours a known asthma gene (B); G is the closest gene to the asthma
    signature in embedding space but has no known neighbour.
    """
    G = nx.Graph()
    for g in ["A", "B", "C", "D", "E", "F", "G"]:
        G.add_node(g, node_type="gene", name=f"{g} protein")
    G.add_node("DIS:1", node_type="disease")
    G.nodes["A"]["gwas"] = {"disease": [{"trait": "Asthma", "score": 0.9}]}
    G.nodes["B"]["gwas"] = {"disease": [{"trait": "Asthma", "score": 0.8}],
                            "trait": [{"trait": "Height", "score": 0.5}]}
    G.nodes["C"]["gwas"] = {"trait": [{"trait": "Height", "score": 0.7}]}
    G.add_edges_from([("A", "B"), ("B", "D"), ("C", "E"), ("D", "F"), ("G", "F"), ("A", "DIS:1")])
    return G


def make_embeddings(node_list):
    vectors = {
        "A": (1.0, 0.0), "B": (1.0, 0.1), "D": (0.9, 0.1), "G": (1.0, 0.05),
        "E": (0.0, 1.0), "F": (-1.0, 0.0), "C": (0.0, 1.0),
        "DIS:1": (1.0, 0.0),  # as similar as it gets, but not a gene
    }
    z = torch.tensor([vectors[n] for n in node_list])
    return F.normalize(z, dim=-1)


def setup():
    G = make_graph()
    node_list = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    return G, node_list, node_to_idx, make_embeddings(node_list)


def neg_log10_p_at_least_one(population, known, draws):
    """P(X >= 1) for a hypergeometric draw, as -log10."""
    p_none = math.comb(population - known, draws) / math.comb(population, draws)
    return -math.log10(1 - p_none)


def test_index_disease_genes_groups_genes_by_trait():
    G, node_list, _, _ = setup()
    index = index_disease_genes(G, node_list)
    assert index["Asthma"] == {"category": "disease", "genes": {"A": 0.9, "B": 0.8}}
    assert index["Height"]["genes"] == {"B": 0.5, "C": 0.7}


def test_select_diseases_filters_by_category_and_min_genes():
    G, node_list, _, _ = setup()
    index = index_disease_genes(G, node_list)
    assert select_diseases(index, categories=["disease"], min_known_genes=2) == ["Asthma"]
    assert select_diseases(index, categories=["disease", "trait"], min_known_genes=2) == ["Asthma", "Height"]
    assert select_diseases(index, categories=["disease"], min_known_genes=3) == []


def test_predict_links_ranks_known_neighbour_enrichment_above_cosine():
    G, node_list, node_to_idx, z_norm = setup()
    index = index_disease_genes(G, node_list)
    rows = predict_gene_disease_links(z_norm, node_list, node_to_idx, G, index, ["Asthma"], top_k=2)
    assert [r["gene"] for r in rows] == ["D", "G"]
    assert [r["rank"] for r in rows] == [1, 2]
    # D: 7 genes in the population, 2 known, D has 2 gene neighbours, 1 of them known
    assert rows[0]["enrichment_score"] == pytest.approx(neg_log10_p_at_least_one(7, 2, 2), abs=1e-3)
    assert rows[0]["degree"] == 2
    assert rows[1]["enrichment_score"] == 0
    assert rows[1]["degree"] == 1
    # cosine alone would have put G first
    assert rows[1]["similarity"] > rows[0]["similarity"]


def test_predict_links_breaks_enrichment_ties_by_similarity():
    G, node_list, node_to_idx, z_norm = setup()
    index = index_disease_genes(G, node_list)
    rows = predict_gene_disease_links(z_norm, node_list, node_to_idx, G, index, ["Height"], top_k=2)
    # A and E each have one gene neighbour that is a known Height gene; E is closer to the signature
    assert [r["gene"] for r in rows] == ["E", "A"]
    assert rows[0]["enrichment_score"] == pytest.approx(rows[1]["enrichment_score"])
    assert rows[0]["similarity"] > rows[1]["similarity"]


def test_predict_links_degree_counts_gene_neighbours_only():
    G, node_list, node_to_idx, z_norm = setup()
    index = index_disease_genes(G, node_list)
    rows = predict_gene_disease_links(z_norm, node_list, node_to_idx, G, index, ["Height"], top_k=5)
    a = next(r for r in rows if r["gene"] == "A")
    assert a["degree"] == 1  # the edge to DIS:1 is not a gene neighbour
    assert a["enrichment_score"] == pytest.approx(neg_log10_p_at_least_one(7, 2, 1), abs=1e-3)


def test_predict_links_excludes_known_and_non_gene_nodes():
    G, node_list, node_to_idx, z_norm = setup()
    index = index_disease_genes(G, node_list)
    rows = predict_gene_disease_links(z_norm, node_list, node_to_idx, G, index, ["Asthma"], top_k=10)
    assert {r["gene"] for r in rows} == {"C", "D", "E", "F", "G"}
    assert all(r["disease"] == "Asthma" and r["category"] == "disease" for r in rows)
    assert all(r["n_known_genes"] == 2 for r in rows)
    assert rows[0]["gene_description"] == "D protein"


def test_predict_links_counts_known_interactors():
    G, node_list, node_to_idx, z_norm = setup()
    index = index_disease_genes(G, node_list)
    rows = predict_gene_disease_links(z_norm, node_list, node_to_idx, G, index, ["Asthma"], top_k=1)
    assert rows[0]["gene"] == "D"
    assert rows[0]["known_interactors"] == 1
    assert rows[0]["known_interactor_genes"] == "B"


def test_predict_links_respects_top_k_per_disease():
    G, node_list, node_to_idx, z_norm = setup()
    index = index_disease_genes(G, node_list)
    rows = predict_gene_disease_links(z_norm, node_list, node_to_idx, G, index, ["Asthma", "Height"], top_k=3)
    per_disease = {}
    for r in rows:
        per_disease[r["disease"]] = per_disease.get(r["disease"], 0) + 1
    assert per_disease == {"Asthma": 3, "Height": 3}


def test_known_links_rows_lists_seed_genes_with_scores():
    G, node_list, _, _ = setup()
    index = index_disease_genes(G, node_list)
    rows = known_links_rows(index, ["Asthma"])
    assert rows == [
        {"disease": "Asthma", "category": "disease", "gene": "A", "gwas_score": 0.9},
        {"disease": "Asthma", "category": "disease", "gene": "B", "gwas_score": 0.8},
    ]


# ---- optional model score matrix ----------------------------------------------

def test_predict_links_ranks_by_model_scores_when_given():
    import numpy as np

    G, node_list, node_to_idx, z_norm = setup()
    index = index_disease_genes(G, node_list)
    scores = np.zeros((len(node_list), 1))
    scores[node_to_idx["G"], 0] = 0.7
    scores[node_to_idx["D"], 0] = 0.2
    rows = predict_gene_disease_links(z_norm, node_list, node_to_idx, G, index, ["Asthma"], top_k=2, scores=scores)
    assert [r["gene"] for r in rows] == ["G", "D"]          # enrichment alone would put D first
    assert [r["model_score"] for r in rows] == [0.7, 0.2]
    assert rows[1]["enrichment_score"] > 0 and rows[0]["enrichment_score"] == 0  # still reported


def test_predict_links_breaks_model_score_ties_by_enrichment_then_similarity():
    import numpy as np

    G, node_list, node_to_idx, z_norm = setup()
    index = index_disease_genes(G, node_list)
    scores = np.zeros((len(node_list), 1))
    rows = predict_gene_disease_links(z_norm, node_list, node_to_idx, G, index, ["Asthma"], top_k=2, scores=scores)
    assert [r["gene"] for r in rows] == ["D", "G"]


def test_predict_links_without_scores_has_empty_model_score():
    G, node_list, node_to_idx, z_norm = setup()
    index = index_disease_genes(G, node_list)
    rows = predict_gene_disease_links(z_norm, node_list, node_to_idx, G, index, ["Asthma"], top_k=2)
    assert all(r["model_score"] is None for r in rows)


def test_predict_links_rejects_mismatched_scores_shape():
    import numpy as np

    G, node_list, node_to_idx, z_norm = setup()
    index = index_disease_genes(G, node_list)
    with pytest.raises(ValueError, match="scores"):
        predict_gene_disease_links(z_norm, node_list, node_to_idx, G, index, ["Asthma", "Height"], top_k=2,
                                   scores=np.zeros((len(node_list), 1)))


def test_predict_links_without_embeddings_ranks_by_enrichment_and_leaves_similarity_empty():
    G, node_list, node_to_idx, _ = setup()
    index = index_disease_genes(G, node_list)
    rows = predict_gene_disease_links(None, node_list, node_to_idx, G, index, ["Asthma"], top_k=3)
    assert [r["gene"] for r in rows] == ["D", "C", "E"]      # D has the known partner; ties keep node order
    assert all(r["similarity"] is None for r in rows)


def test_predict_links_without_embeddings_still_ranks_by_model_scores():
    import numpy as np

    G, node_list, node_to_idx, _ = setup()
    index = index_disease_genes(G, node_list)
    scores = np.zeros((len(node_list), 1))
    scores[node_to_idx["G"], 0] = 0.7
    scores[node_to_idx["D"], 0] = 0.2
    rows = predict_gene_disease_links(None, node_list, node_to_idx, G, index, ["Asthma"], top_k=2, scores=scores)
    assert [r["gene"] for r in rows] == ["G", "D"]
    assert [r["model_score"] for r in rows] == [0.7, 0.2] and all(r["similarity"] is None for r in rows)
