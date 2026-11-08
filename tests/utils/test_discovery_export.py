# tests/utils/test_discovery_export.py
import io
import json
import zipfile

import pandas as pd

from src.utils.discovery_export import (
    BLEND_SCORER,
    HETERO_SCORER,
    KNOWN_FILE,
    MANIFEST_FILE,
    NO_PROTEIN_MODEL,
    PREDICTED_COLUMN_DOCS,
    PREDICTED_COLUMNS,
    PREDICTED_FILE,
    README_FILE,
    build_manifest,
    render_readme,
    write_discovery_bundle,
    zip_bundle,
)


def sample_rows():
    links = [{
        "disease": "Asthma", "category": "disease", "gene": "D", "gene_description": "D protein",
        "enrichment_score": 0.28, "similarity": 0.99, "rank": 1, "n_known_genes": 2,
        "known_interactors": 1, "degree": 2, "known_interactor_genes": "B",
    }]
    known = [
        {"disease": "Asthma", "category": "disease", "gene": "A", "gwas_score": 0.9},
        {"disease": "Asthma", "category": "disease", "gene": "B", "gwas_score": 0.8},
    ]
    return links, known


def sample_manifest(links, known):
    return build_manifest(
        model_type="GNN", metrics={"auc_roc": 0.9, "avg_precision": 0.8},
        checkpoint_path="/x/ckpt.pt", graph_name="full_s800_gwas",
        n_nodes=7, n_edges=5, embedding_dim=2,
        categories=["disease"], min_known_genes=2, top_k=50,
        links=links, known=known,
    )


def test_build_manifest_records_model_parameters_and_counts():
    links, known = sample_rows()
    manifest = sample_manifest(links, known)
    assert manifest["model"]["type"] == "GNN"
    assert manifest["model"]["metrics"]["auc_roc"] == 0.9
    assert manifest["model"]["checkpoint_path"] == "/x/ckpt.pt"
    assert manifest["model"]["embedding_dim"] == 2
    assert manifest["graph"] == {"name": "full_s800_gwas", "n_nodes": 7, "n_edges": 5}
    assert manifest["parameters"] == {
        "categories": ["disease"], "min_known_genes": 2, "top_k": 50, "trait_matching": "exact",
    }
    assert manifest["counts"] == {"diseases": 1, "predicted_links": 1, "known_links": 2}
    assert set(manifest["files"]) == {PREDICTED_FILE, KNOWN_FILE}
    assert "exported_at" in manifest
    assert "enrichment" in manifest["method"].lower()
    assert "enrichment_score" in manifest["files"][PREDICTED_FILE]["columns"]


def test_write_discovery_bundle_writes_csvs_manifest_and_readme(tmp_path):
    links, known = sample_rows()
    bundle = write_discovery_bundle(tmp_path / "discovery_x", links, known, sample_manifest(links, known))
    assert bundle == tmp_path / "discovery_x"
    assert sorted(p.name for p in bundle.iterdir()) == sorted(
        [PREDICTED_FILE, KNOWN_FILE, MANIFEST_FILE, README_FILE]
    )
    predicted = pd.read_csv(bundle / PREDICTED_FILE)
    assert list(predicted.columns) == list(PREDICTED_COLUMNS)
    assert predicted.iloc[0]["gene"] == "D"
    assert len(pd.read_csv(bundle / KNOWN_FILE)) == 2
    manifest = json.loads((bundle / MANIFEST_FILE).read_text())
    assert manifest["counts"]["predicted_links"] == 1
    readme = (bundle / README_FILE).read_text()
    assert PREDICTED_FILE in readme and KNOWN_FILE in readme


def test_write_discovery_bundle_with_no_links_still_writes_headers(tmp_path):
    bundle = write_discovery_bundle(tmp_path / "empty", [], [], sample_manifest([], []))
    predicted = pd.read_csv(bundle / PREDICTED_FILE)
    assert len(predicted) == 0
    assert list(predicted.columns) == list(PREDICTED_COLUMNS)


def test_zip_bundle_contains_every_file_under_the_bundle_name(tmp_path):
    links, known = sample_rows()
    bundle = write_discovery_bundle(tmp_path / "discovery_x", links, known, sample_manifest(links, known))
    archive = zipfile.ZipFile(io.BytesIO(zip_bundle(bundle)))
    assert sorted(archive.namelist()) == sorted(
        f"discovery_x/{name}" for name in [PREDICTED_FILE, KNOWN_FILE, MANIFEST_FILE, README_FILE]
    )


# ---- model scorer in the bundle -------------------------------------------------

def test_predicted_columns_keep_existing_order_and_append_model_score_then_supporting_traits():
    from src.utils.discovery_export import PREDICTED_COLUMN_DOCS

    assert PREDICTED_COLUMNS[:-2] == (
        "disease", "category", "gene", "gene_description", "enrichment_score", "similarity",
        "rank", "n_known_genes", "known_interactors", "degree", "known_interactor_genes",
    )
    assert PREDICTED_COLUMNS[-2:] == ("model_score", "supporting_traits")
    assert "model_score" in PREDICTED_COLUMN_DOCS and "supporting_traits" in PREDICTED_COLUMN_DOCS


def test_manifest_records_scorer_and_hetero_method_text():
    from src.utils.discovery_export import render_readme

    links, known = sample_rows()
    default = sample_manifest(links, known)
    assert default["scorer"] == {"name": "enrichment", "checkpoint": None}
    assert "enrichment" in default["method"].lower() and "heterogeneous" not in default["method"].lower()

    hetero = build_manifest(
        model_type="GNN", metrics={"auc_roc": 0.9}, checkpoint_path="/x/ckpt.pt", graph_name="g",
        n_nodes=7, n_edges=5, embedding_dim=2, categories=["disease"], min_known_genes=2, top_k=50,
        links=links, known=known, scorer="hetero-gnn", scorer_checkpoint="/x/hetero_gnn_1_checkpoint.pt",
    )
    assert hetero["scorer"] == {"name": "hetero-gnn", "checkpoint": "/x/hetero_gnn_1_checkpoint.pt"}
    assert "heterogeneous" in hetero["method"].lower()
    assert "Ranking scorer: hetero-gnn" in render_readme(hetero)
    assert "Ranking scorer: enrichment" in render_readme(default)


def test_bundle_csv_writes_model_score_column(tmp_path):
    links, known = sample_rows()
    links.append({**links[0], "gene": "E", "rank": 2, "model_score": 0.7})
    manifest = sample_manifest(links, known)
    bundle = write_discovery_bundle(tmp_path / "b", links, known, manifest)
    df = pd.read_csv(bundle / PREDICTED_FILE)
    assert list(df.columns) == list(PREDICTED_COLUMNS)
    assert df.columns[-2] == "model_score"
    assert pd.isna(df.loc[0, "model_score"]) and df.loc[1, "model_score"] == 0.7


def test_manifest_without_protein_model_says_so():
    links, known = sample_rows()
    links[0]["similarity"] = None
    manifest = build_manifest(
        model_type=None, metrics={}, checkpoint_path=None, graph_name="full_s800_gwas",
        n_nodes=7, n_edges=5, embedding_dim=None, categories=["disease"], min_known_genes=2, top_k=50,
        links=links, known=known,
    )
    assert manifest["model"]["type"] == NO_PROTEIN_MODEL == "none"
    assert "No protein–protein model" in manifest["method"] and "cosine similarity" not in manifest["method"]
    hetero = build_manifest(
        model_type=None, metrics={}, checkpoint_path=None, graph_name="full_s800_gwas",
        n_nodes=7, n_edges=5, embedding_dim=None, categories=["disease"], min_known_genes=2, top_k=50,
        links=links, known=known, scorer=HETERO_SCORER, scorer_checkpoint="/x/hetero.pt",
    )
    assert "similarity column is empty" in hetero["method"] and "and similarity are kept" not in hetero["method"]
    readme = render_readme(manifest)
    assert "- Type: none (no protein–protein model loaded)" in readme and "- Test metrics: n/a" in readme
    assert "empty when no protein" in PREDICTED_COLUMN_DOCS["similarity"]


def test_manifest_records_blend_scorer_method_text():
    links, known = sample_rows()
    kwargs = dict(metrics={"auc_roc": 0.9}, checkpoint_path="/x/ckpt.pt", graph_name="g", n_nodes=7, n_edges=5,
                  embedding_dim=2, categories=["disease"], min_known_genes=2, top_k=50, links=links, known=known,
                  scorer=BLEND_SCORER, scorer_checkpoint="/x/hetero.pt")
    manifest = build_manifest(model_type="GNN", **kwargs)
    assert manifest["scorer"] == {"name": "hetero-gnn+enrichment-z", "checkpoint": "/x/hetero.pt"}
    assert "z-score" in manifest["method"] and "cosine similarity" not in manifest["method"]
    assert "similarity column is empty" in build_manifest(model_type=None, **kwargs)["method"]
    assert "z-score" in PREDICTED_COLUMN_DOCS["model_score"]


def test_blend_method_text_names_the_z_score_population():
    from src.utils.discovery_export import BLEND_METHOD_TEXT, BLEND_METHOD_TEXT_NO_PROTEIN
    for text in (BLEND_METHOD_TEXT, BLEND_METHOD_TEXT_NO_PROTEIN, PREDICTED_COLUMN_DOCS["model_score"]):
        assert "over all nodes of the network" in text


def test_blend_method_text_reports_the_benchmark_range_including_the_run_below_enrichment():
    from src.utils.discovery_export import BLEND_METHOD_TEXT
    assert "−2 % to +16 %" in BLEND_METHOD_TEXT
    assert "slightly more" not in BLEND_METHOD_TEXT


def test_manifest_records_coassociation_scorer_parameters_and_method_text():
    from src.utils.discovery_export import (
        COASSOCIATION_SCORER, COASSOCIATION_STRICT_SCORER, SIMILAR_TRAITS_FILE, render_readme,
    )

    links, known = sample_rows()
    kwargs = dict(metrics={"auc_roc": 0.9}, checkpoint_path="/x/ckpt.pt", graph_name="g", n_nodes=7, n_edges=5,
                  embedding_dim=2, categories=["disease"], min_known_genes=2, top_k=50, links=links, known=known)
    params = {"top_traits": 20, "weighting": "log1p(GWAS score)", "exclude_name_sharing": False}
    manifest = build_manifest(model_type="GNN", scorer=COASSOCIATION_SCORER, scorer_parameters=params, **kwargs)
    assert manifest["scorer"] == {"name": "trait-coassociation", "checkpoint": None, "parameters": params}
    assert "20 most similar" in manifest["method"] and "then by similarity" in manifest["method"]
    assert "shares a word" not in manifest["method"]
    strict = build_manifest(model_type=None, scorer=COASSOCIATION_STRICT_SCORER,
                            scorer_parameters={**params, "exclude_name_sharing": True}, **kwargs)
    assert "shares a word" in strict["method"] and "similarity column is empty" in strict["method"]
    assert "parameters" not in build_manifest(model_type="GNN", **kwargs)["scorer"]
    assert "co-association" in PREDICTED_COLUMN_DOCS["model_score"] and "supporting_traits" in PREDICTED_COLUMN_DOCS
    readme = render_readme(manifest)
    assert "Ranking scorer: trait-coassociation (top_traits=20" in readme
    assert "similar_traits.csv" not in readme and SIMILAR_TRAITS_FILE not in manifest["files"]


def test_bundle_writes_similar_traits_file_only_for_coassociation_exports(tmp_path):
    from src.utils.discovery_export import (
        COASSOCIATION_SCORER, SIMILAR_TRAITS_COLUMNS, SIMILAR_TRAITS_FILE, render_readme,
    )

    links, known = sample_rows()
    similar = [{"disease": "Asthma", "trait": "Allergic rhinitis", "similarity": 0.6174, "rank": 1},
               {"disease": "Asthma", "trait": "Eczema", "similarity": 0.3361, "rank": 2}]
    kwargs = dict(metrics={}, checkpoint_path=None, graph_name="g", n_nodes=7, n_edges=5, embedding_dim=None,
                  categories=["disease"], min_known_genes=2, top_k=50, links=links, known=known)
    manifest = build_manifest(model_type=None, scorer=COASSOCIATION_SCORER,
                              scorer_parameters={"top_traits": 20}, similar_traits=similar, **kwargs)
    assert manifest["counts"]["similar_traits"] == 2
    assert list(manifest["files"][SIMILAR_TRAITS_FILE]["columns"]) == list(SIMILAR_TRAITS_COLUMNS)
    bundle = write_discovery_bundle(tmp_path / "b", links, known, manifest, similar_traits=similar)
    df = pd.read_csv(bundle / SIMILAR_TRAITS_FILE)
    assert list(df.columns) == list(SIMILAR_TRAITS_COLUMNS) and df["trait"].tolist() == ["Allergic rhinitis", "Eczema"]
    assert f"`{SIMILAR_TRAITS_FILE}`" in render_readme(manifest)
    plain = write_discovery_bundle(tmp_path / "p", links, known, sample_manifest(links, known))
    assert not (plain / SIMILAR_TRAITS_FILE).exists()
