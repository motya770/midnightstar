# Predictions Without a Protein Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Predictions page rank disease genes (by network enrichment or a gene–disease model) and export them without a protein–protein model loaded; the two tabs that need protein embeddings say so.

**Architecture:** The ranking function and the export manifest learn to work without embeddings (`z_norm=None`, `model_type=None`); a session flag records that the user chose to continue without a protein model; the page offers that choice next to *Load Model*, skips the embedding step in that mode, and disables *Find Similar Genes* / *Check Gene Pair* with a notice.

**Tech Stack:** Python 3, Streamlit, NumPy/PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md` Section 6 (Results page). The user's request: "Make the Predictions page work with the gene–disease model alone, without loading a protein model" (answered "Yes, do it").

## Global Constraints

- Hetero models never go into `training_results` (spec Section 9): the no-protein mode is a separate session flag.
- Existing export columns and their order are unchanged (spec Section 6); `similarity` stays a column and is empty without a protein model.
- Network enrichment stays the default ranking; the gene–disease model stays labelled experimental.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. No saved network at all: choosing *predict without a protein model* must end in the existing "No network data available… Train Model" error, not a crash on `graph.nodes()`.
2. A gene–disease checkpoint from another network in no-protein mode: the ranking must fall back to enrichment with the existing warning (no protein model to fall back to is irrelevant; enrichment needs only the graph).
3. Export preview after switching modes: the success box keyed on `model_token` must not show a protein-model export after switching to no-protein mode or the reverse.
4. A network without GWAS data in no-protein mode: the Predict tab and export keep their existing "No disease/GWAS data" messages.
5. A protein model loaded later (from Train Model or *Load Model*): the flag is ignored and the page behaves exactly as before.

---

### Task 1: Rank without protein embeddings

**Files:**
- Modify: `src/utils/discovery.py` (`predict_gene_disease_links`)
- Test: `tests/utils/test_discovery.py`

**Interfaces:**
- Produces: `predict_gene_disease_links(z_norm | None, node_list, node_to_idx, graph, index, diseases, top_k, *, scores=None)`; with `z_norm=None` every row's `similarity` is `None` and ties in the ranking keep `node_list` order.

- [ ] **Step 1: Write the failing tests**

```python
def test_predict_links_without_embeddings_ranks_by_enrichment_and_leaves_similarity_empty():
    G, node_list, node_to_idx, _ = setup()
    index = index_disease_genes(G, node_list)
    rows = predict_gene_disease_links(None, node_list, node_to_idx, G, index, ["Asthma"], top_k=3)
    assert [r["gene"] for r in rows] == ["D", "C", "E"]      # D has the known partner; ties keep node order
    assert all(r["similarity"] is None for r in rows)


def test_predict_links_without_embeddings_still_ranks_by_model_scores():
    import numpy as np

    G, node_list, node_to_idx, _ = setup()
    index = index_disease_genes(G, node_list)
    scores = np.zeros((len(node_list), 1))
    scores[node_to_idx["G"], 0] = 0.7
    scores[node_to_idx["D"], 0] = 0.2
    rows = predict_gene_disease_links(None, node_list, node_to_idx, G, index, ["Asthma"], top_k=2, scores=scores)
    assert [r["gene"] for r in rows] == ["G", "D"]
    assert [r["model_score"] for r in rows] == [0.7, 0.2] and all(r["similarity"] is None for r in rows)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/utils/test_discovery.py -k without_embeddings -v`
Expected: FAIL with `AttributeError: 'NoneType' object has no attribute 'detach'`

- [ ] **Step 3: Implement** — `z_norm` may be `None`: skip the similarity key in `np.lexsort`, write `None` into `similarity`; update the docstring ("Without ``z_norm`` ties keep node order and ``similarity`` is None").

- [ ] **Step 4: Run the file**

Run: `pytest tests/utils/test_discovery.py -v`
Expected: all PASS

- [ ] **Step 5: Commit** `git commit -m "Rank disease genes without protein embeddings"`

### Task 2: Manifest without a protein model

**Files:**
- Modify: `src/utils/discovery_export.py` (`build_manifest`, `render_readme`, `PREDICTED_COLUMN_DOCS["similarity"]`)
- Test: `tests/utils/test_discovery_export.py`

**Interfaces:**
- Produces: `NO_PROTEIN_MODEL = "none"`; `build_manifest(model_type=None, metrics={}, checkpoint_path=None, embedding_dim=None, ...)` records `model.type == "none"` and appends a sentence to `method`; `render_readme` prints `- Type: none (no protein–protein model loaded)`.

- [ ] **Step 1: Write the failing test**

```python
def test_manifest_without_protein_model_says_so():
    links, known = sample_rows()
    links[0]["similarity"] = None
    manifest = build_manifest(
        model_type=None, metrics={}, checkpoint_path=None, graph_name="full_s800_gwas",
        n_nodes=7, n_edges=5, embedding_dim=None, categories=["disease"], min_known_genes=2, top_k=50,
        links=links, known=known,
    )
    assert manifest["model"]["type"] == NO_PROTEIN_MODEL == "none"
    assert "No protein–protein model" in manifest["method"]
    readme = render_readme(manifest)
    assert "- Type: none (no protein–protein model loaded)" in readme and "- Test metrics: n/a" in readme
    assert "empty when no protein" in PREDICTED_COLUMN_DOCS["similarity"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/utils/test_discovery_export.py -k without_protein -v`
Expected: FAIL with `ImportError: cannot import name 'NO_PROTEIN_MODEL'`

- [ ] **Step 3: Implement** — constant, `"type": model_type or NO_PROTEIN_MODEL`, method text `+ NO_PROTEIN_MODEL_TEXT` when `model_type is None` (text: "No protein–protein model was loaded for this export, so the similarity column is empty and ties in the ranking score keep the network's gene order."), README line, column doc suffix "; empty when no protein–protein model was loaded".

- [ ] **Step 4: Run the file** — `pytest tests/utils/test_discovery_export.py -v` → all PASS

- [ ] **Step 5: Commit** `git commit -m "Export manifest without a protein model"`

### Task 3: Session flag

**Files:**
- Modify: `src/utils/session.py`
- Test: `tests/utils/test_session.py`

**Interfaces:**
- Produces: `get_no_protein_model() -> bool` (default False), `set_no_protein_model(flag: bool)`, key `"no_protein_model"`.

- [ ] **Step 1: Write the failing test**

```python
def test_no_protein_model_flag_defaults_to_false_and_round_trips(monkeypatch):
    monkeypatch.setattr(session.st, "session_state", {})
    assert session.get_no_protein_model() is False
    session.set_no_protein_model(True)
    assert session.get_no_protein_model() is True
    session.set_no_protein_model(False)
    assert session.get_no_protein_model() is False
```

- [ ] **Step 2: Run** `pytest tests/utils/test_session.py -v` → FAIL `AttributeError: module ... has no attribute 'get_no_protein_model'`
- [ ] **Step 3: Implement** the two functions.
- [ ] **Step 4: Run** `pytest tests/utils/test_session.py -v` → PASS
- [ ] **Step 5: Commit** `git commit -m "Session flag for predicting without a protein model"`

### Task 4: Predictions page

**Files:**
- Modify: `pages/4_Results.py`

No automated page tests exist in this repo; the task is verified in the running app (port 8521) after Step 2.

- [ ] **Step 1: Implement**
  - Model selector block runs when `results is None and not session.get_no_protein_model()`. When there is no protein checkpoint or run, keep the info text but do not stop: below the selector (or the info) show **Or predict without a protein–protein model** with a caption ("The Predict Disease Genes tab and the export rank by network enrichment, or by a gene–disease model checkpoint when one exists (N found); only the network is needed. Find Similar Genes and Check Gene Pair need a protein–protein model.") and the button **🧬 Predict without a protein model** → `session.set_no_protein_model(True)`; `st.rerun()`. Without any graph (session or saved) show the existing Train Model hint instead of the button. Remove the old "not listed here: load a protein–protein model" caption.
  - After the block: if `results` → unchanged embedding step; else `model = pyg_data = z = z_norm = None`, `node_list = sorted(graph.nodes())`, `metrics = {}`, `model_type = None`, info bar "**Active model:** none — ranking by network enrichment or a gene–disease model · **Genes loaded:** N" and a button **Load a protein–protein model instead** → `set_no_protein_model(False)`; `st.rerun()`. `graph is None` → existing "No network data available" error + stop.
  - Predict tab: intro sentence about the similarity tie-break only when `z_norm is not None`; drop the *Similarity* column from the candidates table when `z_norm is None`.
  - Find Similar Genes / Check Gene Pair: `st.info("… needs a protein–protein model's learned profiles. Use **Load a protein–protein model instead** above.")` and skip the tab body when `z_norm is None`.
  - *Save Model to Disk* only when `model is not None`.
  - Export: `checkpoint_path=results.get("checkpoint_path") if results else None`, `n_nodes`/`n_edges` from `pyg_data` or the graph, `embedding_dim=z.size(1) if z is not None else None`; `model_token = id(model) if model is not None else "no-protein-model"`.
- [ ] **Step 2: Verify in the browser** (reload the page for a fresh session): button shown and works; info bar; Predict with enrichment and with a gene–disease checkpoint; the two notices; export writes a bundle whose `manifest.json` has `model.type == "none"` and `method` mentioning the missing protein model; *Load a protein–protein model instead* returns to the selector; loading a protein model still works.
- [ ] **Step 3: Run the suite** `pytest -q` → all PASS
- [ ] **Step 4: Commit** `git commit -m "Predictions page: rank and export without a protein model"`

### Task 5: Docs and PR

**Files:**
- Modify: `README.md` (paragraph after "When a hetero checkpoint exists…"), `docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md` (Section 6 Results-page bullet), the PR #4 body (new part).

- [ ] **Step 1: README**: one paragraph: the Predictions page no longer requires a protein–protein model; *Predict without a protein model* ranks by enrichment or a gene–disease model; Find Similar Genes and Check Gene Pair need one; exports then record `model.type = "none"` and an empty `similarity` column.
- [ ] **Step 2: Spec Section 6 bullet**: add "The page also works without a protein–protein model (*Predict without a protein model*): enrichment and the gene–disease model need only the network; the similarity tie-break and the two embedding tabs need one."
- [ ] **Step 3: Commit** `git commit -m "Docs: Predictions page without a protein model"`
