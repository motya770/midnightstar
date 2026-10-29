"""Convert NetworkX graph to PyG Data with node and edge features."""
import torch
from torch_geometric.data import Data

EDGE_FEATURE_NAMES = ("score", "experimental", "database", "textmining", "coexpression")


def nx_to_pyg_data(G):
    """Convert a NetworkX graph to a PyG Data object.

    Returns: (data, node_list, node_to_idx)
    """
    node_list = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)

    # Collect tissue names
    all_tissues = set()
    for n in node_list:
        expr = G.nodes[n].get("expression", {})
        all_tissues.update(expr.keys())
    all_tissues = sorted(all_tissues)

    # Feature vector: [expression_per_tissue..., plddt, disorder, seq_len, degree]
    has_alphafold = any(G.nodes[n].get("mean_plddt") is not None for n in node_list)
    af_features = 3 if has_alphafold else 0

    num_features = max(len(all_tissues), 1) + af_features + 1  # +1 for degree
    x = torch.zeros(num_nodes, num_features)
    expr_cols = max(len(all_tissues), 1)

    # GWAS features: build trait vocabulary and encode as token sequences
    gwas_categories = ["disease", "trait", "phenotype", "measurement", "biological_process", "other"]
    trait_vocab = {}  # trait_name -> token_id (0 = padding)
    next_id = 1
    for n in node_list:
        for entries in G.nodes[n].get("gwas", {}).values():
            for entry in entries:
                if entry["trait"] not in trait_vocab:
                    trait_vocab[entry["trait"]] = next_id
                    next_id += 1

    # Build category ID mapping for each trait
    cat_to_id = {cat: i for i, cat in enumerate(gwas_categories)}
    trait_to_cat_id = {}
    for n in node_list:
        gwas = G.nodes[n].get("gwas", {})
        for cat, entries in gwas.items():
            for entry in entries:
                trait_to_cat_id[entry["trait"]] = cat_to_id.get(cat, len(gwas_categories))

    # Find max tokens per node for padding
    max_gwas_tokens = 0
    for n in node_list:
        count = sum(len(entries) for entries in G.nodes[n].get("gwas", {}).values())
        max_gwas_tokens = max(max_gwas_tokens, count)
    max_gwas_tokens = max(max_gwas_tokens, 1)

    gwas_token_ids = torch.zeros(num_nodes, max_gwas_tokens, dtype=torch.long)
    gwas_scores = torch.zeros(num_nodes, max_gwas_tokens)
    gwas_cat_ids = torch.zeros(num_nodes, max_gwas_tokens, dtype=torch.long)

    for i, n in enumerate(node_list):
        # Numeric features
        expr = G.nodes[n].get("expression", {})
        for j, tissue in enumerate(all_tissues):
            x[i, j] = expr.get(tissue, 0.0)

        offset = expr_cols
        if has_alphafold:
            # Keys may be present with None when the database row had NULLs.
            x[i, offset] = (G.nodes[n].get("mean_plddt") or 0.0) / 100.0
            x[i, offset + 1] = G.nodes[n].get("disordered_fraction") or 0.0
            x[i, offset + 2] = (G.nodes[n].get("sequence_length") or 0) / 5000.0
            offset += 3

        x[i, offset] = float(G.degree(n))

        # GWAS tokens
        tok_idx = 0
        for entries in G.nodes[n].get("gwas", {}).values():
            for entry in entries:
                if tok_idx < max_gwas_tokens:
                    gwas_token_ids[i, tok_idx] = trait_vocab[entry["trait"]]
                    gwas_scores[i, tok_idx] = entry["score"]
                    gwas_cat_ids[i, tok_idx] = trait_to_cat_id.get(entry["trait"], 0)
                    tok_idx += 1

    # Normalize expression columns
    max_vals = x[:, :expr_cols].max(dim=0).values.clamp(min=1.0)
    x[:, :expr_cols] = x[:, :expr_cols] / max_vals
    # Normalize degree
    deg_col = num_features - 1
    max_deg = x[:, deg_col].max().clamp(min=1.0)
    x[:, deg_col] = x[:, deg_col] / max_deg

    # Build edges
    src, dst, edge_attr_rows = [], [], []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            features = [float(G.edges[u, v].get(name, 0.0)) for name in EDGE_FEATURE_NAMES]
            src.append(node_to_idx[u])
            dst.append(node_to_idx[v])
            edge_attr_rows.append(features)
            src.append(node_to_idx[v])
            dst.append(node_to_idx[u])
            edge_attr_rows.append(features)

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    if edge_attr_rows:
        edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float)
    else:
        edge_attr = torch.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    data.edge_attr = edge_attr
    # Degree is topology-derived; the trainer recomputes this column from the
    # message-passing graph so held-out edges cannot leak through it.
    data.degree_col = deg_col
    data.gwas_token_ids = gwas_token_ids
    data.gwas_scores = gwas_scores
    data.gwas_cat_ids = gwas_cat_ids
    data.gwas_vocab_size = next_id
    data.gwas_num_categories = len(gwas_categories) + 1
    return data, node_list, node_to_idx
