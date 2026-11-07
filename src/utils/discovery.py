"""Gene–disease discovery helpers.

Pure functions over a trained model's node embeddings and the NetworkX graph
they were trained on. Used by the Results page export so the logic can be
tested without Streamlit.
"""
import torch
import torch.nn.functional as F


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


def predict_gene_disease_links(z_norm, node_list, node_to_idx, graph, index, diseases,
                               top_k: int, max_listed_interactors: int = 10) -> list[dict]:
    """Rank candidate genes for each disease by cosine similarity to the disease signature.

    The signature is the normalised mean embedding of the disease's known genes.
    Known genes and non-gene nodes are never returned. One row per (disease, gene).
    """
    z_norm = z_norm.detach().cpu()
    gene_mask = torch.tensor(
        [graph.nodes[n].get("node_type") == "gene" for n in node_list], dtype=torch.bool
    )
    rows: list[dict] = []
    for disease in diseases:
        record = index[disease]
        known = [g for g in record["genes"] if g in node_to_idx]
        if not known:
            continue
        known_idx = torch.tensor([node_to_idx[g] for g in known])
        signature = F.normalize(z_norm[known_idx].mean(dim=0, keepdim=True), dim=-1)
        similarities = (z_norm @ signature.T).squeeze(1)

        allowed = gene_mask.clone()
        allowed[known_idx] = False
        k = min(top_k, int(allowed.sum().item()))
        if k <= 0:
            continue
        similarities = similarities.masked_fill(~allowed, float("-inf"))
        top_values, top_indices = torch.topk(similarities, k)

        known_set = set(known)
        for rank, (value, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
            gene = node_list[idx]
            interactors = sorted(set(graph.neighbors(gene)) & known_set) if gene in graph else []
            rows.append({
                "disease": disease,
                "category": record["category"],
                "gene": gene,
                "gene_description": graph.nodes[gene].get("name", ""),
                "similarity": round(value, 4),
                "rank": rank,
                "n_known_genes": len(known),
                "known_interactors": len(interactors),
                "known_interactor_genes": ";".join(interactors[:max_listed_interactors]),
            })
    return rows


def known_links_rows(index: dict[str, dict], diseases) -> list[dict]:
    """The known gene–disease associations that seeded each disease signature."""
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
