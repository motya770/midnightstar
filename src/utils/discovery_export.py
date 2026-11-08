"""Write gene–disease discovery results as a self-describing folder bundle.

The bundle is plain CSV plus a JSON manifest and a README, so it can be handed
to Claude Science (grant the folder, or attach the files) or any other tool.
"""
import datetime
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

PREDICTED_FILE = "predicted_gene_disease_links.csv"
KNOWN_FILE = "known_gene_disease_links.csv"
MANIFEST_FILE = "manifest.json"
README_FILE = "README.md"

PREDICTED_COLUMNS = (
    "disease", "category", "gene", "gene_description", "enrichment_score", "similarity",
    "rank", "n_known_genes", "known_interactors", "degree", "known_interactor_genes",
    "model_score",
)
KNOWN_COLUMNS = ("disease", "category", "gene", "gwas_score")

PREDICTED_COLUMN_DOCS = {
    "disease": "GWAS Catalog trait name, matched exactly",
    "category": "Trait category assigned by MidnightStar (disease, trait, ...)",
    "gene": "Candidate gene symbol; never one already associated with the disease",
    "gene_description": "Protein description from the Human Protein Atlas, when known",
    "enrichment_score": "Primary ranking score: -log10 hypergeometric p-value that the gene's STRING partners "
                        "contain at least the observed number of known disease genes (0 = no known partner)",
    "similarity": "Cosine similarity between the gene's embedding and the disease signature; "
                  "breaks ties in enrichment_score only; empty when no protein–protein model was loaded",
    "rank": "1 = strongest candidate for that disease",
    "n_known_genes": "Number of known disease genes used as the seed set",
    "known_interactors": "How many known disease genes have a STRING interaction with the candidate",
    "degree": "Number of STRING interaction partners of the candidate",
    "known_interactor_genes": "Up to ten of those known interactors, separated by ';'",
    "model_score": "Logit of the heterogeneous gene–disease model for this pair when that model produced the "
                   "ranking (see the manifest's scorer); empty when enrichment ranked the candidates",
}
KNOWN_COLUMN_DOCS = {
    "disease": "GWAS Catalog trait name",
    "category": "Trait category assigned by MidnightStar",
    "gene": "Gene with a reported GWAS association to the trait",
    "gwas_score": "MidnightStar's association score derived from the GWAS Catalog entry",
}

ENRICHMENT_SCORER = "enrichment"
HETERO_SCORER = "hetero-gnn"
NO_PROTEIN_MODEL = "none"   # manifest model.type when the export was made without a protein–protein model

# Method texts: one per scorer, each with and without a protein–protein model (which only supplies
# the similarity tie-break).
_HETERO_METHOD_HEAD = (
    "Candidates are ranked by a heterogeneous graph neural network trained on gene–disease "
    "links: genes and GWAS traits are nodes, STRING protein interactions and GWAS gene–trait "
    "associations are edges, and the model learns to score (gene, disease) pairs directly. "
    "The model_score column is its logit; enrichment_score (hypergeometric enrichment of "
    "known genes among a gene's STRING partners) "
)
HETERO_METHOD_TEXT = _HETERO_METHOD_HEAD + (
    "and similarity are kept for reference and break ties. Candidates are hypotheses for follow-up."
)
HETERO_METHOD_TEXT_NO_PROTEIN = _HETERO_METHOD_HEAD + (
    "is kept for reference and breaks ties; no protein–protein model was loaded for this export, so "
    "the similarity column is empty. Candidates are hypotheses for follow-up."
)
_ENRICHMENT_METHOD_HEAD = (
    "Candidates are ranked by guilt-by-association on the STRING protein interaction "
    "network: for each disease, every gene is scored by the hypergeometric enrichment of "
    "known GWAS genes among its interaction partners (-log10 p, population = all genes in "
    "the network). "
)
_SIMILARITY_TIE_TEXT = (
    "Ties, in particular genes with no known partner, are broken by cosine "
    "similarity between the gene's embedding and the disease signature (the normalised "
    "mean embedding of the known genes) from a link-prediction model trained on the same "
    "network. "
)
NO_PROTEIN_TIE_TEXT = (
    "No protein–protein model was loaded for this export, so ties, in particular genes with "
    "no known partner, are not broken (they keep the network's gene order) and the similarity "
    "column is empty. "
)
_ENRICHMENT_METHOD_TAIL = (
    "This is a network-based heuristic, not a supervised disease classifier, and "
    "candidates are hypotheses for follow-up."
)
METHOD_TEXT = _ENRICHMENT_METHOD_HEAD + _SIMILARITY_TIE_TEXT + _ENRICHMENT_METHOD_TAIL
METHOD_TEXT_NO_PROTEIN = _ENRICHMENT_METHOD_HEAD + NO_PROTEIN_TIE_TEXT + _ENRICHMENT_METHOD_TAIL


def build_manifest(*, model_type, metrics, checkpoint_path, graph_name, n_nodes, n_edges,
                   embedding_dim, categories, min_known_genes, top_k, links, known,
                   scorer: str = ENRICHMENT_SCORER, scorer_checkpoint: str | None = None) -> dict:
    """``scorer`` names what produced the ranking: "enrichment" (default) or "hetero-gnn".

    ``model_type=None`` records an export made without a protein–protein model (no embeddings,
    no similarity tie-break): ``model.type`` becomes ``"none"`` and the method text says so."""
    diseases = {row["disease"] for row in links} | {row["disease"] for row in known}
    if scorer == HETERO_SCORER:
        method = HETERO_METHOD_TEXT if model_type is not None else HETERO_METHOD_TEXT_NO_PROTEIN
    else:
        method = METHOD_TEXT if model_type is not None else METHOD_TEXT_NO_PROTEIN
    return {
        "app": "MidnightStar",
        "export": "gene-disease discovery",
        "exported_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {
            "type": model_type or NO_PROTEIN_MODEL,
            "checkpoint_path": checkpoint_path,
            "metrics": dict(metrics),
            "embedding_dim": embedding_dim,
        },
        "scorer": {"name": scorer, "checkpoint": scorer_checkpoint},
        "graph": {"name": graph_name, "n_nodes": n_nodes, "n_edges": n_edges},
        "parameters": {
            "categories": list(categories),
            "min_known_genes": min_known_genes,
            "top_k": top_k,
            "trait_matching": "exact",
        },
        "counts": {
            "diseases": len(diseases),
            "predicted_links": len(links),
            "known_links": len(known),
        },
        "method": method,
        "files": {
            PREDICTED_FILE: {
                "description": "Predicted (new) gene–disease connections, one row per disease and candidate gene",
                "columns": PREDICTED_COLUMN_DOCS,
            },
            KNOWN_FILE: {
                "description": "Known GWAS gene–disease associations used to build each disease signature",
                "columns": KNOWN_COLUMN_DOCS,
            },
        },
    }


def render_readme(manifest: dict) -> str:
    model = manifest["model"]
    graph = manifest["graph"]
    params = manifest["parameters"]
    counts = manifest["counts"]
    metrics = ", ".join(f"{k} = {v:.4f}" if isinstance(v, (int, float)) else f"{k} = {v}"
                        for k, v in model["metrics"].items()) or "n/a"
    lines = [
        "# MidnightStar discovery export",
        "",
        f"Exported {manifest['exported_at']} from MidnightStar.",
        "",
        "## What is here",
        "",
        f"- `{PREDICTED_FILE}`: {manifest['files'][PREDICTED_FILE]['description']} "
        f"({counts['predicted_links']} rows across {counts['diseases']} diseases).",
        f"- `{KNOWN_FILE}`: {manifest['files'][KNOWN_FILE]['description']} ({counts['known_links']} rows).",
        f"- `{MANIFEST_FILE}`: model, graph, parameters and column descriptions in machine-readable form.",
        "",
        "To see which diseases a protein may be newly connected to, filter "
        f"`{PREDICTED_FILE}` by the `gene` column; to see candidate genes for a disease, filter by `disease`.",
        "",
        "## Model",
        "",
        f"- Type: {model['type']}" + (" (no protein–protein model loaded)" if model["type"] == NO_PROTEIN_MODEL else ""),
        f"- Ranking scorer: {manifest.get('scorer', {}).get('name', ENRICHMENT_SCORER)}"
        + (f" ({manifest['scorer']['checkpoint']})" if manifest.get('scorer', {}).get('checkpoint') else ""),
        f"- Test metrics: {metrics}",
        f"- Checkpoint: {model['checkpoint_path'] or 'not recorded'}",
        f"- Network: {graph['name'] or 'unnamed'} ({graph['n_nodes']} nodes, {graph['n_edges']} edges)",
        "",
        "## Parameters",
        "",
        f"- Categories: {', '.join(params['categories']) or 'none'}",
        f"- Minimum known genes per disease: {params['min_known_genes']}",
        f"- Candidates per disease (top-k): {params['top_k']}",
        f"- Trait matching: {params['trait_matching']}",
        "",
        "## Method",
        "",
        manifest["method"],
        "",
        "## Columns",
        "",
        f"### {PREDICTED_FILE}",
        "",
        *[f"- `{col}`: {doc}" for col, doc in PREDICTED_COLUMN_DOCS.items()],
        "",
        f"### {KNOWN_FILE}",
        "",
        *[f"- `{col}`: {doc}" for col, doc in KNOWN_COLUMN_DOCS.items()],
        "",
    ]
    return "\n".join(lines)


def write_discovery_bundle(out_dir, links: list[dict], known: list[dict], manifest: dict) -> Path:
    """Write the CSVs, manifest and README into out_dir and return it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(links, columns=list(PREDICTED_COLUMNS)).to_csv(out_dir / PREDICTED_FILE, index=False)
    pd.DataFrame(known, columns=list(KNOWN_COLUMNS)).to_csv(out_dir / KNOWN_FILE, index=False)
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    (out_dir / README_FILE).write_text(render_readme(manifest))
    return out_dir


def zip_bundle(bundle_dir) -> bytes:
    """Zip every file in the bundle under a top-level folder of the same name."""
    bundle_dir = Path(bundle_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in bundle_dir.iterdir() if p.is_file()):
            archive.write(path, arcname=f"{bundle_dir.name}/{path.name}")
    return buffer.getvalue()
