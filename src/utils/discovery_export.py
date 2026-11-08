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
                  "breaks ties in enrichment_score only",
    "rank": "1 = strongest candidate for that disease",
    "n_known_genes": "Number of known disease genes used as the seed set",
    "known_interactors": "How many known disease genes have a STRING interaction with the candidate",
    "degree": "Number of STRING interaction partners of the candidate",
    "known_interactor_genes": "Up to ten of those known interactors, separated by ';'",
}
KNOWN_COLUMN_DOCS = {
    "disease": "GWAS Catalog trait name",
    "category": "Trait category assigned by MidnightStar",
    "gene": "Gene with a reported GWAS association to the trait",
    "gwas_score": "MidnightStar's association score derived from the GWAS Catalog entry",
}

METHOD_TEXT = (
    "Candidates are ranked by guilt-by-association on the STRING protein interaction "
    "network: for each disease, every gene is scored by the hypergeometric enrichment of "
    "known GWAS genes among its interaction partners (-log10 p, population = all genes in "
    "the network). Ties, in particular genes with no known partner, are broken by cosine "
    "similarity between the gene's embedding and the disease signature (the normalised "
    "mean embedding of the known genes) from a link-prediction model trained on the same "
    "network. This is a network-based heuristic, not a supervised disease classifier, and "
    "candidates are hypotheses for follow-up."
)


def build_manifest(*, model_type, metrics, checkpoint_path, graph_name, n_nodes, n_edges,
                   embedding_dim, categories, min_known_genes, top_k, links, known) -> dict:
    diseases = {row["disease"] for row in links} | {row["disease"] for row in known}
    return {
        "app": "MidnightStar",
        "export": "gene-disease discovery",
        "exported_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {
            "type": model_type,
            "checkpoint_path": checkpoint_path,
            "metrics": dict(metrics),
            "embedding_dim": embedding_dim,
        },
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
        "method": METHOD_TEXT,
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
        f"- Type: {model['type']}",
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
