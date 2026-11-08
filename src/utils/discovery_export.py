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
SIMILAR_TRAITS_FILE = "similar_traits.csv"   # only in bundles ranked by trait co-association
MANIFEST_FILE = "manifest.json"
README_FILE = "README.md"

PREDICTED_COLUMNS = (
    "disease", "category", "gene", "gene_description", "enrichment_score", "similarity",
    "rank", "n_known_genes", "known_interactors", "degree", "known_interactor_genes",
    "model_score", "supporting_traits",
)
KNOWN_COLUMNS = ("disease", "category", "gene", "gwas_score")
SIMILAR_TRAITS_COLUMNS = ("disease", "trait", "similarity", "rank")

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
    "model_score": "Ranking score of the manifest's scorer when one other than enrichment ranked: the heterogeneous "
                   "gene–disease model's logit (scorer hetero-gnn); the sum of the per-disease z-scores of "
                   "enrichment_score and that logit, each z-scored over all nodes of the network (scorer "
                   "hetero-gnn+enrichment-z); or the trait co-association score (scorers trait-coassociation and "
                   "trait-coassociation-strict), the sum over the disease's most similar GWAS traits of the trait's "
                   "similarity to the disease times log1p of the candidate's GWAS score for that trait; "
                   "empty when enrichment alone ranked the candidates",
    "supporting_traits": "For the trait co-association scorers: up to three of the disease's most similar GWAS "
                         "traits that the candidate is associated with, strongest contribution first, each with the "
                         "trait's similarity to the disease in parentheses, separated by '; '; empty for other scorers",
}
SIMILAR_TRAITS_COLUMN_DOCS = {
    "disease": "GWAS Catalog trait name of an exported disease",
    "trait": "A GWAS trait whose gene set is similar to the disease's; the scorer summed over these",
    "similarity": "Cosine similarity between the two traits' columns of the log1p(GWAS score) gene × trait matrix",
    "rank": "1 = most similar trait of that disease",
}
KNOWN_COLUMN_DOCS = {
    "disease": "GWAS Catalog trait name",
    "category": "Trait category assigned by MidnightStar",
    "gene": "Gene with a reported GWAS association to the trait",
    "gwas_score": "MidnightStar's association score derived from the GWAS Catalog entry",
}

ENRICHMENT_SCORER = "enrichment"
HETERO_SCORER = "hetero-gnn"
BLEND_SCORER = "hetero-gnn+enrichment-z"   # the benchmark's hetero+enrichment-z: z-score sum of enrichment and logit
COASSOCIATION_SCORER = "trait-coassociation"             # the benchmark's coassociation
COASSOCIATION_STRICT_SCORER = "trait-coassociation-strict"   # the benchmark's coassociation-strict
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
_BLEND_METHOD_HEAD = (
    "Candidates are ranked by the sum of two per-disease z-scores, each taken over all nodes of the "
    "network: the hypergeometric enrichment of "
    "known GWAS genes among the gene's STRING partners (enrichment_score, -log10 p) and the logit of a "
    "heterogeneous graph neural network trained on gene–disease links (genes and GWAS traits as nodes, "
    "STRING interactions and GWAS associations as edges). The model_score column is that sum; "
    "enrichment_score "
)
_BLEND_METHOD_TAIL = (
    "On the held-out benchmark this blend recovered about as many hidden disease genes in the top 50 as "
    "enrichment alone or somewhat more (−2 % to +16 % across the checkpoints tried, with mixed "
    "recall@200), while the model alone recovered far fewer. Candidates are hypotheses for follow-up."
)
BLEND_METHOD_TEXT = _BLEND_METHOD_HEAD + "and similarity are kept for reference and break ties. " + _BLEND_METHOD_TAIL
BLEND_METHOD_TEXT_NO_PROTEIN = _BLEND_METHOD_HEAD + (
    "is kept for reference and breaks ties; no protein–protein model was loaded for this export, so "
    "the similarity column is empty. "
) + _BLEND_METHOD_TAIL
_COASSOCIATION_METHOD_HEAD = (
    "Candidates are ranked by trait co-association: the GWAS traits most similar to the disease are "
    "found by cosine similarity of their gene sets (a gene × trait matrix of log1p GWAS scores over "
    "every association in the network, the disease's own column excluded), and each gene is scored by "
    "the sum, over the disease's up to 20 most similar traits (fewer when fewer traits share a gene "
    "with it), of that similarity times log1p of the gene's GWAS score for the trait. "
)
_COASSOCIATION_STRICT_SENTENCE = (
    "Traits whose name shares a word with the disease name (alias and multi-trait study names) were "
    "excluded from the similar traits. "
)
_COASSOCIATION_METHOD_MIDDLE = (
    "The model_score column is that sum and supporting_traits lists the traits behind it; ties, in "
    "particular genes with no association to any of those traits, are broken by enrichment_score "
    "(hypergeometric enrichment of known disease genes among the gene's STRING partners)"
)
_COASSOCIATION_TIE_WITH_PROTEIN = " and then by similarity. "
_COASSOCIATION_TIE_NO_PROTEIN = (
    "; no protein–protein model was loaded for this export, so the similarity column is empty. "
)
_COASSOCIATION_METHOD_TAIL = (
    "On the held-out benchmark trait co-association recovered 26 % of hidden disease genes in the top 50 "
    "against 4 % for enrichment (19 % with name-sharing traits excluded); part of the gain comes from GWAS "
    "Catalog alias traits. Candidates are hypotheses for follow-up."
)


def _coassociation_method_text(strict: bool, no_protein: bool) -> str:
    return (_COASSOCIATION_METHOD_HEAD + (_COASSOCIATION_STRICT_SENTENCE if strict else "")
            + _COASSOCIATION_METHOD_MIDDLE
            + (_COASSOCIATION_TIE_NO_PROTEIN if no_protein else _COASSOCIATION_TIE_WITH_PROTEIN)
            + _COASSOCIATION_METHOD_TAIL)


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
                   scorer: str = ENRICHMENT_SCORER, scorer_checkpoint: str | None = None,
                   scorer_parameters: dict | None = None, similar_traits: list[dict] | None = None) -> dict:
    """``scorer`` names what produced the ranking: "enrichment" (the parameter's default), "hetero-gnn",
    "hetero-gnn+enrichment-z", "trait-coassociation" or "trait-coassociation-strict".

    ``scorer_parameters`` (the co-association scorers' ``top_traits``, ``weighting`` and
    ``exclude_name_sharing``) is recorded under ``scorer.parameters`` only when given. ``similar_traits``
    rows (disease, trait, similarity, rank) describe the bundle's third file, ``similar_traits.csv``, and
    are listed under ``files`` and ``counts`` only when given.

    ``model_type=None`` records an export made without a protein–protein model (no embeddings,
    no similarity tie-break): ``model.type`` becomes ``"none"`` and the method text says so."""
    diseases = {row["disease"] for row in links} | {row["disease"] for row in known}
    no_protein = model_type is None
    if scorer == HETERO_SCORER:
        method = HETERO_METHOD_TEXT_NO_PROTEIN if no_protein else HETERO_METHOD_TEXT
    elif scorer == BLEND_SCORER:
        method = BLEND_METHOD_TEXT_NO_PROTEIN if no_protein else BLEND_METHOD_TEXT
    elif scorer in (COASSOCIATION_SCORER, COASSOCIATION_STRICT_SCORER):
        method = _coassociation_method_text(strict=(scorer == COASSOCIATION_STRICT_SCORER), no_protein=no_protein)
    else:
        method = METHOD_TEXT_NO_PROTEIN if no_protein else METHOD_TEXT
    scorer_block = {"name": scorer, "checkpoint": scorer_checkpoint}
    if scorer_parameters is not None:
        scorer_block["parameters"] = dict(scorer_parameters)
    counts = {"diseases": len(diseases), "predicted_links": len(links), "known_links": len(known)}
    files = {
        PREDICTED_FILE: {
            "description": "Predicted (new) gene–disease connections, one row per disease and candidate gene",
            "columns": PREDICTED_COLUMN_DOCS,
        },
        KNOWN_FILE: {
            "description": "Known GWAS gene–disease associations used to build each disease signature",
            "columns": KNOWN_COLUMN_DOCS,
        },
    }
    if similar_traits is not None:
        counts["similar_traits"] = len(similar_traits)
        files[SIMILAR_TRAITS_FILE] = {
            "description": "The GWAS traits most similar to each exported disease, which the trait "
                           "co-association scorer summed over",
            "columns": SIMILAR_TRAITS_COLUMN_DOCS,
        }
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
        "scorer": scorer_block,
        "graph": {"name": graph_name, "n_nodes": n_nodes, "n_edges": n_edges},
        "parameters": {
            "categories": list(categories),
            "min_known_genes": min_known_genes,
            "top_k": top_k,
            "trait_matching": "exact",
        },
        "counts": counts,
        "method": method,
        "files": files,
    }


def render_readme(manifest: dict) -> str:
    model = manifest["model"]
    graph = manifest["graph"]
    params = manifest["parameters"]
    counts = manifest["counts"]
    metrics = ", ".join(f"{k} = {v:.4f}" if isinstance(v, (int, float)) else f"{k} = {v}"
                        for k, v in model["metrics"].items()) or "n/a"
    scorer = manifest.get("scorer", {})
    scorer_detail = scorer.get("checkpoint") or (
        ", ".join(f"{k}={v}" for k, v in scorer["parameters"].items()) if scorer.get("parameters") else "")
    similar = manifest["files"].get(SIMILAR_TRAITS_FILE)
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
        *([f"- `{SIMILAR_TRAITS_FILE}`: {similar['description']} ({counts.get('similar_traits', 0)} rows)."]
          if similar else []),
        f"- `{MANIFEST_FILE}`: model, graph, parameters and column descriptions in machine-readable form.",
        "",
        "To see which diseases a protein may be newly connected to, filter "
        f"`{PREDICTED_FILE}` by the `gene` column; to see candidate genes for a disease, filter by `disease`.",
        "",
        "## Model",
        "",
        f"- Type: {model['type']}" + (" (no protein–protein model loaded)" if model["type"] == NO_PROTEIN_MODEL else ""),
        f"- Ranking scorer: {scorer.get('name', ENRICHMENT_SCORER)}" + (f" ({scorer_detail})" if scorer_detail else ""),
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
        *([f"### {SIMILAR_TRAITS_FILE}", "",
           *[f"- `{col}`: {doc}" for col, doc in similar["columns"].items()], ""] if similar else []),
    ]
    return "\n".join(lines)


def write_discovery_bundle(out_dir, links: list[dict], known: list[dict], manifest: dict,
                           similar_traits: list[dict] | None = None) -> Path:
    """Write the CSVs, manifest and README into out_dir and return it.

    ``similar_traits`` rows (from ``CoassociationScores.similar_trait_rows``) add ``similar_traits.csv``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(links, columns=list(PREDICTED_COLUMNS)).to_csv(out_dir / PREDICTED_FILE, index=False)
    pd.DataFrame(known, columns=list(KNOWN_COLUMNS)).to_csv(out_dir / KNOWN_FILE, index=False)
    if similar_traits is not None:
        pd.DataFrame(similar_traits, columns=list(SIMILAR_TRAITS_COLUMNS)).to_csv(out_dir / SIMILAR_TRAITS_FILE, index=False)
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
