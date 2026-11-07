# tests/utils/test_discovery_export.py
import io
import json
import zipfile

import pandas as pd

from src.utils.discovery_export import (
    KNOWN_FILE,
    MANIFEST_FILE,
    PREDICTED_COLUMNS,
    PREDICTED_FILE,
    README_FILE,
    build_manifest,
    write_discovery_bundle,
    zip_bundle,
)


def sample_rows():
    links = [{
        "disease": "Asthma", "category": "disease", "gene": "D", "gene_description": "D protein",
        "similarity": 0.99, "rank": 1, "n_known_genes": 2,
        "known_interactors": 1, "known_interactor_genes": "B",
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
