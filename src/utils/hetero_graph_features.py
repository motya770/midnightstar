"""Convert a saved MidnightStar graph into a heterogeneous gene–trait PyG graph.

Gene nodes keep the numeric features of ``nx_to_pyg_data`` (tissue expression,
AlphaFold, degree). Every GWAS trait found on the gene nodes becomes a ``trait``
node, and each (gene, trait) association becomes an edge in both directions, so
the GWAS signal is structural rather than a node feature. STRING edges stay as
``('gene', 'interacts', 'gene')``.

``exclude_pairs`` removes (gene, trait) associations from the graph entirely:
the benchmark's hidden pairs go here so a model trained on the result has never
seen them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch_geometric.data import HeteroData

TRAIT_CATEGORIES = ("disease", "trait", "phenotype", "measurement", "biological_process", "other")
GENE_TRAIT = ("gene", "has", "trait")
TRAIT_GENE = ("trait", "rev_has", "gene")
GENE_GENE = ("gene", "interacts", "gene")


@dataclass
class HeteroBundle:
    data: HeteroData
    node_list: list[str]
    node_to_idx: dict[str, int]
    trait_list: list[str]
    trait_to_idx: dict[str, int]
    trait_category: dict[str, str] = field(default_factory=dict)


def _gene_features(G, node_list, node_to_idx) -> torch.Tensor:
    """Expression per tissue (+ AlphaFold when present) + gene degree, normalised as in nx_to_pyg_data.

    Degree counts gene neighbours only, so it matches the message-passing graph.
    """
    all_tissues = sorted({t for n in node_list for t in G.nodes[n].get("expression", {})})
    has_alphafold = any(G.nodes[n].get("mean_plddt") is not None for n in node_list)
    expr_cols = max(len(all_tissues), 1)
    num_features = expr_cols + (3 if has_alphafold else 0) + 1
    x = torch.zeros(len(node_list), num_features)
    for i, n in enumerate(node_list):
        expr = G.nodes[n].get("expression", {})
        for j, tissue in enumerate(all_tissues):
            x[i, j] = expr.get(tissue, 0.0)
        offset = expr_cols
        if has_alphafold:
            x[i, offset] = (G.nodes[n].get("mean_plddt") or 0.0) / 100.0
            x[i, offset + 1] = G.nodes[n].get("disordered_fraction") or 0.0
            x[i, offset + 2] = (G.nodes[n].get("sequence_length") or 0) / 5000.0
            offset += 3
        x[i, offset] = float(sum(1 for v in G.neighbors(n) if v in node_to_idx and v != n))
    x[:, :expr_cols] = x[:, :expr_cols] / x[:, :expr_cols].max(dim=0).values.clamp(min=1.0)
    x[:, -1] = x[:, -1] / x[:, -1].max().clamp(min=1.0)
    return x


def nx_to_hetero_data(G, *, exclude_pairs=None, min_trait_genes: int = 1,
                      trait_list: list[str] | None = None) -> HeteroBundle:
    """Build the heterogeneous graph. See the module docstring.

    ``trait_list`` forces the trait order (a checkpoint's) and raises ``ValueError``
    when the graph's traits differ from it.
    """
    exclude = set(exclude_pairs or ())
    node_list = sorted(n for n in G.nodes() if G.nodes[n].get("node_type") == "gene")
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    # (gene, trait) -> (best score, category); category from the first sighting
    associations: dict[tuple[str, str], float] = {}
    trait_category: dict[str, str] = {}
    for gene in node_list:
        for category, entries in G.nodes[gene].get("gwas", {}).items():
            for entry in entries:
                trait = entry["trait"]
                if (gene, trait) in exclude:
                    continue
                trait_category.setdefault(trait, category if category in TRAIT_CATEGORIES else "other")
                key = (gene, trait)
                if entry["score"] > associations.get(key, float("-inf")):
                    associations[key] = entry["score"]

    genes_per_trait: dict[str, int] = {}
    for (_, trait) in associations:
        genes_per_trait[trait] = genes_per_trait.get(trait, 0) + 1
    kept = {t for t, c in genes_per_trait.items() if c >= min_trait_genes}
    if trait_list is None:
        trait_list = sorted(kept)
    elif set(trait_list) != kept:
        missing = sorted(kept - set(trait_list))[:5]
        extra = sorted(set(trait_list) - kept)[:5]
        raise ValueError(f"Graph traits differ from the given trait list "
                         f"(missing from list: {missing}, not in graph: {extra})")
    trait_to_idx = {t: i for i, t in enumerate(trait_list)}
    trait_category = {t: trait_category[t] for t in trait_list}

    data = HeteroData()
    data["gene"].x = _gene_features(G, node_list, node_to_idx)
    data["gene"].num_nodes = len(node_list)
    trait_x = torch.zeros(len(trait_list), len(TRAIT_CATEGORIES))
    for t, i in trait_to_idx.items():
        trait_x[i, TRAIT_CATEGORIES.index(trait_category[t])] = 1.0
    data["trait"].x = trait_x
    data["trait"].num_nodes = len(trait_list)

    src, dst = [], []
    for u, v in G.edges():
        iu, iv = node_to_idx.get(u), node_to_idx.get(v)
        if iu is None or iv is None or iu == iv:
            continue
        src += [iu, iv]
        dst += [iv, iu]
    data[GENE_GENE].edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)

    g_idx, t_idx, weights = [], [], []
    for (gene, trait), score in associations.items():
        if trait in trait_to_idx:
            g_idx.append(node_to_idx[gene])
            t_idx.append(trait_to_idx[trait])
            weights.append(float(score))
    gt = torch.tensor([g_idx, t_idx], dtype=torch.long) if g_idx else torch.zeros((2, 0), dtype=torch.long)
    w = torch.tensor(weights, dtype=torch.float)
    data[GENE_TRAIT].edge_index = gt
    data[GENE_TRAIT].edge_weight = w
    data[TRAIT_GENE].edge_index = gt.flip(0)
    data[TRAIT_GENE].edge_weight = w.clone()

    return HeteroBundle(data=data, node_list=node_list, node_to_idx=node_to_idx,
                        trait_list=list(trait_list), trait_to_idx=trait_to_idx, trait_category=trait_category)
