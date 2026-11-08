# tests/utils/test_hetero_graph_features.py
import networkx as nx
import pytest
import torch

from src.utils.graph_features import nx_to_pyg_data
from src.utils.hetero_graph_features import TRAIT_CATEGORIES, HeteroBundle, nx_to_hetero_data

GENE = ("gene", "has", "trait")
REV = ("trait", "rev_has", "gene")
PPI = ("gene", "interacts", "gene")


def make_graph(with_gwas=True):
    G = nx.Graph()
    for g in ["A", "B", "C", "D"]:
        G.add_node(g, node_type="gene", expression={"liver": 2.0 if g == "A" else 1.0}, mean_plddt=80.0,
                   disordered_fraction=0.1, sequence_length=500)
    G.add_node("DIS:1", node_type="disease")  # not a gene: skipped
    G.add_edges_from([("A", "B"), ("B", "C"), ("A", "DIS:1")])
    if with_gwas:
        G.nodes["A"]["gwas"] = {"disease": [{"trait": "Asthma", "score": 0.9}],
                                "measurement": [{"trait": "Height", "score": 0.4}]}
        G.nodes["B"]["gwas"] = {"disease": [{"trait": "Asthma", "score": 0.8}, {"trait": "Asthma", "score": 0.3}]}
        G.nodes["C"]["gwas"] = {"weird": [{"trait": "Odd", "score": 0.2}]}
    return G


def edge_set(bundle, edge_type):
    ei = bundle.data[edge_type].edge_index
    return set(map(tuple, ei.t().tolist()))


def test_gene_features_and_string_edges_match_homogeneous_conversion():
    G = make_graph()
    bundle = nx_to_hetero_data(G)
    assert isinstance(bundle, HeteroBundle)
    assert bundle.node_list == ["A", "B", "C", "D"]
    genes_only = G.subgraph(bundle.node_list).copy()
    for n in genes_only.nodes:
        genes_only.nodes[n].pop("gwas", None)
    data, _, _ = nx_to_pyg_data(genes_only)
    assert torch.equal(bundle.data["gene"].x, data.x)  # 1 tissue + 3 AlphaFold + degree
    assert edge_set(bundle, PPI) == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_trait_nodes_are_sorted_with_category_one_hot():
    bundle = nx_to_hetero_data(make_graph())
    assert bundle.trait_list == ["Asthma", "Height", "Odd"]
    assert bundle.trait_to_idx == {"Asthma": 0, "Height": 1, "Odd": 2}
    x = bundle.data["trait"].x
    assert x.shape == (3, len(TRAIT_CATEGORIES))
    assert x[0].tolist() == [1, 0, 0, 0, 0, 0]  # disease
    assert x[1].tolist() == [0, 0, 0, 1, 0, 0]  # measurement
    assert x[2].tolist() == [0, 0, 0, 0, 0, 1]  # unknown category -> other
    assert bundle.trait_category == {"Asthma": "disease", "Height": "measurement", "Odd": "other"}


def test_gene_trait_edges_carry_gwas_scores_in_both_directions():
    bundle = nx_to_hetero_data(make_graph())
    ei = bundle.data[GENE].edge_index
    w = bundle.data[GENE].edge_weight
    pairs = {(bundle.node_list[g], bundle.trait_list[t]): round(float(s), 4)
             for g, t, s in zip(ei[0].tolist(), ei[1].tolist(), w.tolist())}
    assert pairs == {("A", "Asthma"): 0.9, ("A", "Height"): 0.4, ("B", "Asthma"): 0.8, ("C", "Odd"): 0.2}
    assert torch.equal(bundle.data[REV].edge_index, ei.flip(0))
    assert torch.equal(bundle.data[REV].edge_weight, w)


def test_exclude_pairs_removes_edges_in_both_directions():
    bundle = nx_to_hetero_data(make_graph(), exclude_pairs={("A", "Asthma")})
    a, asthma = bundle.node_to_idx["A"], bundle.trait_to_idx["Asthma"]
    assert (a, asthma) not in edge_set(bundle, GENE)
    assert (asthma, a) not in edge_set(bundle, REV)
    assert (bundle.node_to_idx["B"], asthma) in edge_set(bundle, GENE)
    assert bundle.data[GENE].edge_index.shape == (2, 3)
    assert "Asthma" in bundle.trait_list  # the trait itself stays


def test_min_trait_genes_drops_small_traits():
    bundle = nx_to_hetero_data(make_graph(), min_trait_genes=2)
    assert bundle.trait_list == ["Asthma"]
    assert bundle.data[GENE].edge_index.shape == (2, 2)
    assert bundle.data["trait"].x.shape[0] == 1


def test_min_trait_genes_counts_genes_after_exclusion():
    bundle = nx_to_hetero_data(make_graph(), exclude_pairs={("A", "Asthma")}, min_trait_genes=2)
    assert bundle.trait_list == []


def test_trait_list_argument_enforces_order_and_rejects_mismatch():
    bundle = nx_to_hetero_data(make_graph(), trait_list=["Odd", "Height", "Asthma"])
    assert bundle.trait_list == ["Odd", "Height", "Asthma"]
    assert bundle.data["trait"].x[0].tolist() == [0, 0, 0, 0, 0, 1]
    with pytest.raises(ValueError, match="trait"):
        nx_to_hetero_data(make_graph(), trait_list=["Asthma", "Height"])


def test_graph_without_gwas_yields_no_traits():
    bundle = nx_to_hetero_data(make_graph(with_gwas=False))
    assert bundle.trait_list == []
    assert bundle.data["trait"].x.shape == (0, len(TRAIT_CATEGORIES))
    assert bundle.data[GENE].edge_index.shape == (2, 0)
    assert bundle.data[REV].edge_index.shape == (2, 0)
    assert bundle.data[PPI].edge_index.shape == (2, 4)


def test_non_gene_nodes_are_skipped():
    bundle = nx_to_hetero_data(make_graph())
    assert "DIS:1" not in bundle.node_to_idx
    assert bundle.data["gene"].x.shape[0] == 4
