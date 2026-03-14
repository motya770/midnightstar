# tests/utils/test_graph_builder.py
import networkx as nx
from src.data.models import GeneNode, DiseaseNode, Association, ExpressionProfile
from src.utils.graph_builder import GraphBuilder

def _sample_genes():
    return [
        GeneNode("SP4", "ENSG00000105866", "SP4", "Sp4 TF", "TF", "Homo sapiens"),
        GeneNode("HSPD1", "ENSG00000144381", "HSPD1", "HSP60", "Chaperone", "Homo sapiens"),
    ]

def _sample_associations():
    return [
        Association("SP4", "HSPD1", "protein-protein", 0.87, "experiments", "STRING"),
        Association("SP4", "EFO_0000249", "gene-disease", 0.6, "p<5e-8", "GWAS"),
    ]

def _sample_diseases():
    return [DiseaseNode("EFO_0000249", "Alzheimer's", "Neuro disease", "neurological", "GWAS")]

def test_build_graph_nodes():
    builder = GraphBuilder()
    builder.add_genes(_sample_genes())
    builder.add_diseases(_sample_diseases())
    G = builder.build()
    assert G.number_of_nodes() == 3
    assert G.nodes["SP4"]["node_type"] == "gene"
    assert G.nodes["EFO_0000249"]["node_type"] == "disease"

def test_build_graph_edges():
    builder = GraphBuilder()
    builder.add_genes(_sample_genes())
    builder.add_diseases(_sample_diseases())
    builder.add_associations(_sample_associations())
    G = builder.build()
    assert G.number_of_edges() == 2
    edge_data = G.edges["SP4", "HSPD1"]
    assert edge_data["score"] == 0.87
    assert edge_data["data_source"] == "STRING"

def test_build_graph_with_expression():
    builder = GraphBuilder()
    builder.add_genes(_sample_genes()[:1])
    builder.add_expression([
        ExpressionProfile("ENSG00000105866", "Brain_Cortex", 15.3, 255),
        ExpressionProfile("ENSG00000105866", "Liver", 2.1, 200),
    ])
    G = builder.build()
    assert "expression" in G.nodes["SP4"]
    assert G.nodes["SP4"]["expression"]["Brain_Cortex"] == 15.3

def test_filter_by_score():
    builder = GraphBuilder()
    builder.add_genes(_sample_genes())
    builder.add_diseases(_sample_diseases())
    builder.add_associations(_sample_associations())
    G = builder.build(min_score=0.7)
    assert G.number_of_edges() == 1

def test_filter_by_source():
    builder = GraphBuilder()
    builder.add_genes(_sample_genes())
    builder.add_diseases(_sample_diseases())
    builder.add_associations(_sample_associations())
    G = builder.build(sources={"STRING"})
    assert G.number_of_edges() == 1

def test_merge_duplicate_edges():
    builder = GraphBuilder()
    builder.add_genes(_sample_genes())
    builder.add_associations([
        Association("SP4", "HSPD1", "protein-protein", 0.87, "exp", "STRING"),
        Association("SP4", "HSPD1", "co-expression", 0.72, "coex", "GTEx"),
    ])
    G = builder.build()
    assert G.number_of_edges() == 1
    edge = G.edges["SP4", "HSPD1"]
    assert len(edge["sources"]) == 2
    assert edge["score"] == 0.87
