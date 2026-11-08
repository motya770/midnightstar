# Enrichment + Gene–Disease Model Blend Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offer the z-score blend of network enrichment and the gene–disease model's logit (the benchmark's `hetero+enrichment-z`, the only scorer consistently at or above enrichment) as a *Ranking score* option on the Predictions page and in the export.

**Architecture:** One pure helper computes the blend from a model score matrix and the graph (`enrichment_matrix` + `benchmark.zscore_sum`); the export gains a scorer name and method texts for it; the page lists one blend option per checkpoint ahead of the model-alone options and ranks with the blended matrix through the existing `scores=` path. Enrichment stays the default.

**Tech Stack:** Python 3, NumPy, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md` Sections 6 and 8 (the blend was CLI-only "unless the model beats enrichment"; the user chose to offer it after the 2026-11-08 export review: raw z-sum recall@50 0.0423–0.0461 vs enrichment 0.0415 over five checkpoints, recall@200 mixed).

## Global Constraints

- Enrichment stays the default; existing export columns and order unchanged; `model_score` carries the ranking score of whichever model-based scorer ranked (logit or blend) and its doc says so.
- The blend is computed exactly as the benchmark does (`zscore_sum` over all nodes of the enrichment matrix from the full index and the checkpoint's logits), so page numbers and benchmark numbers agree.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A disease with no trait node in the checkpoint: its logit column is all zeros (constant), so the blend must reduce to enrichment's z-score for that disease, not crash or flatten.
2. A checkpoint that cannot score the network: the blend option must fall back to enrichment with the existing warning, like the model-alone option.
3. No-protein mode: the blend works without embeddings (z_norm None) and the manifest's blend method text has a no-protein variant.
4. The candidates table and CSV: the score column is labelled for the blend, and `model_score` in the bundle is the blend value, documented in the manifest.
5. Option order and labels: enrichment first, then blends (newest checkpoint first), then model-alone, each note honest about the benchmark.

---

### Task 1: Blend helper

**Files:** Modify `src/utils/hetero_scorer.py`; Test `tests/utils/test_hetero_scorer.py`

**Interfaces:** Produces `blend_with_enrichment(model_scores, graph, node_list, node_to_idx, index, diseases) -> np.ndarray` (n_nodes × n_diseases) = `zscore_sum(enrichment_matrix(graph, node_list, node_to_idx, index, diseases), model_scores)`.

- [ ] Step 1: failing test
```python
def test_blend_with_enrichment_is_the_benchmark_zscore_sum_and_keeps_enrichment_where_the_model_is_flat():
    G = make_graph()
    node_list = sorted(G.nodes()); node_to_idx = {n: i for i, n in enumerate(node_list)}
    index = index_disease_genes(G, node_list)
    model = np.zeros((len(node_list), 2)); model[node_to_idx[GENES[20]], 0] = 5.0
    enrichment = enrichment_matrix(G, node_list, node_to_idx, index, ["Asthma", "BMI"])
    blend = blend_with_enrichment(model, G, node_list, node_to_idx, index, ["Asthma", "BMI"])
    np.testing.assert_allclose(blend, zscore_sum(enrichment, model))
    assert blend[:, 0].argmax() == node_to_idx[GENES[20]]                     # the model's one strong gene wins Asthma
    assert np.corrcoef(blend[:, 1], enrichment[:, 1])[0, 1] > 0.999           # flat model column: BMI is enrichment's order
```
- [ ] Step 2: `pytest tests/utils/test_hetero_scorer.py -k blend -v` → FAIL (ImportError)
- [ ] Step 3: implement; Step 4: file passes; Step 5: commit "Blend of enrichment and gene–disease model score".

### Task 2: Export scorer name and method texts

**Files:** Modify `src/utils/discovery_export.py`; Test `tests/utils/test_discovery_export.py`

**Interfaces:** Produces `BLEND_SCORER = "hetero-gnn+enrichment-z"`, `BLEND_METHOD_TEXT`, `BLEND_METHOD_TEXT_NO_PROTEIN`; `build_manifest(scorer=BLEND_SCORER)` picks them; `PREDICTED_COLUMN_DOCS["model_score"]` names both meanings.

- [ ] Step 1: failing test
```python
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
```
- [ ] Step 2: FAIL (ImportError); Step 3: implement; Step 4: pass; Step 5: commit "Export manifest for the blend scorer".

### Task 3: Page option and docs

**Files:** Modify `pages/4_Results.py`, `README.md`, spec Sections 6 and 8.

- [ ] Step 1: page — options `[enrichment] + [blend per checkpoint] + [model per checkpoint]`, labels "Enrichment + gene–disease model: <name> (z-score blend, slightly above enrichment on the benchmark)" and the existing model label; `_scorer_kinds[label] = (kind, path)`; `scorer_for` blends with `blend_with_enrichment(matrix, graph, node_list, node_to_idx, disease_index, diseases)` and returns `BLEND_SCORER`; candidates caption and score column ("Blend score" / "Model score") per scorer; `SCORER_HELP` gains the blend sentence with the benchmark numbers.
- [ ] Step 2: verify in the app (both modes): blend ranking for type 2 diabetes, column label, an export to the scratch folder with `scorer.name` "hetero-gnn+enrichment-z".
- [ ] Step 3: README (Ranking score options) and spec (Section 6 bullet; Section 8 sentence "available in the benchmark CLI, not in the UI" → now also on the page, 2026-11-08).
- [ ] Step 4: `pytest -q` green; commit "Predictions page: enrichment + gene–disease model blend scorer".
