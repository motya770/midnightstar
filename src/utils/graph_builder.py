import networkx as nx
from src.data.models import GeneNode, DiseaseNode, Association, ExpressionProfile


class GraphBuilder:
    def __init__(self):
        self._genes: list[GeneNode] = []
        self._diseases: list[DiseaseNode] = []
        self._associations: list[Association] = []
        self._expression: dict[str, dict[str, float]] = {}
        self._ensembl_to_symbol: dict[str, str] = {}

    def add_genes(self, genes: list[GeneNode]):
        for g in genes:
            self._genes.append(g)
            self._ensembl_to_symbol[g.ensembl_id] = g.symbol

    def add_diseases(self, diseases: list[DiseaseNode]):
        self._diseases.extend(diseases)

    def add_associations(self, associations: list[Association]):
        self._associations.extend(associations)

    def add_expression(self, profiles: list[ExpressionProfile]):
        for p in profiles:
            symbol = self._ensembl_to_symbol.get(p.gene_id, p.gene_id)
            if symbol not in self._expression:
                self._expression[symbol] = {}
            self._expression[symbol][p.tissue] = p.expression_level

    def build(self, min_score: float = 0.0, sources: set[str] | None = None) -> nx.Graph:
        G = nx.Graph()
        for gene in self._genes:
            G.add_node(gene.symbol, node_type="gene", **gene.to_dict())
        for disease in self._diseases:
            G.add_node(disease.id, node_type="disease", **disease.to_dict())
        edge_groups: dict[tuple[str, str], list[Association]] = {}
        for assoc in self._associations:
            if sources and assoc.data_source not in sources:
                continue
            if assoc.score < min_score:
                continue
            key = tuple(sorted([assoc.source_id, assoc.target_id]))
            edge_groups.setdefault(key, []).append(assoc)
        for (src, tgt), assocs in edge_groups.items():
            best_score = max(a.score for a in assocs)
            all_sources = list({a.data_source for a in assocs})
            all_evidence = "; ".join(a.evidence for a in assocs if a.evidence)
            G.add_edge(src, tgt, score=best_score, sources=all_sources,
                       data_source=all_sources[0], evidence=all_evidence, type=assocs[0].type)
        for symbol, tissues in self._expression.items():
            if symbol in G.nodes:
                G.nodes[symbol]["expression"] = tissues
        return G
