# tests/utils/test_discovery.py
import networkx as nx
import torch
import torch.nn.functional as F

from src.utils.discovery import (
    index_disease_genes,
    known_links_rows,
    predict_gene_disease_links,
    select_diseases,
)


def make_graph():
    G = nx.Graph()
    for g in ["A", "B", "C", "D", "E", "F"]:
        G.add_node(g, node_type="gene", name=f"{g} protein")
    G.add_node("DIS:1", node_type="disease")
    G.nodes["A"]["gwas"] = {"disease": [{"trait": "Asthma", "score": 0.9}]}
    G.nodes["B"]["gwas"] = {"disease": [{"trait": "Asthma", "score": 0.8}],
                            "trait": [{"trait": "Height", "score": 0.5}]}
    G.nodes["C"]["gwas"] = {"trait": [{"trait": "Height", "score": 0.7}]}
    G.add_edges_from([("A", "B"), ("B", "D"), ("C", "E"), ("D", "F"), ("A", "DIS:1")])
    return G


def make_embeddings(node_list):
    vectors = {
        "A": (1.0, 0.0), "B": (1.0, 0.1), "D": (0.9, 0.1),
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


def test_predict_links_excludes_known_and_non_gene_nodes_and_ranks():
    G, node_list, node_to_idx, z_norm = setup()
    index = index_disease_genes(G, node_list)
    rows = predict_gene_disease_links(z_norm, node_list, node_to_idx, G, index, ["Asthma"], top_k=2)
    assert [r["gene"] for r in rows] == ["D", "E"] or [r["gene"] for r in rows] == ["D", "C"]
    assert [r["rank"] for r in rows] == [1, 2]
    assert rows[0]["similarity"] >= rows[1]["similarity"]
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
