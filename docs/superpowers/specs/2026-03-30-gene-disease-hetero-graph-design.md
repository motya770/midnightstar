# Gene-Disease Heterogeneous Graph Link Prediction

**Date:** 2026-03-30
**Status:** Approved

## Problem

The current pipeline trains link prediction on protein-protein edges only. Disease gene prediction is a post-hoc cosine similarity heuristic over gene embeddings — not a trained predictor. The model never sees gene-disease associations as training signal, so it cannot learn disease-specific patterns.

## Goal

Add a new graph type and model pipeline for **bidirectional gene-disease link prediction**:
- Given a disease, rank candidate genes
- Given a gene, rank candidate diseases

This is fully additive — nothing in the existing protein-protein pipeline is modified.

## Approach: Heterogeneous Graph with HeteroConv

Build a bipartite graph with two node types and two edge types. Use PyG's `HeteroData` and `HeteroConv` to learn separate relation patterns for protein interactions and gene-disease associations.

---

## Section 1: Graph Construction

**New method:** `build_disease_graph()` on `BulkDatasetManager`.

**Node types:**
- **gene** (~20K nodes from HPA/STRING) — Features: GTEx expression, AlphaFold structure, HPA annotations. Same enrichment as existing `build_full_graph`.
- **disease** (5K-15K nodes, one per unique GWAS mapped trait) — Features: ontology category (one-hot over disease/phenotype/measurement/trait/biological_process/other = 6 dims), log(associated gene count), mean association score, max association score. ~9 features total.

**Edge types:**
- **gene-gene** (protein-protein) — from STRING, same logic as `build_full_graph`. Confidence threshold configurable.
- **gene-disease** — from GWAS. One edge per (gene, trait) pair meeting the p-value threshold. Edge weight = `-log10(p-value) / 50`, clamped to [0, 1].

**No disease-disease edges.** GWAS does not provide direct disease-disease relationships.

**Key difference from existing graphs:** The existing `build_full_graph` stores GWAS as node features on genes. The new builder stores GWAS as explicit edges to disease nodes. Gene nodes in the disease graph do NOT carry GWAS token features — the GWAS signal is structural.

**Graph type marker:** `G.graph["type"] = "gene-disease"` stored as a graph-level attribute. Existing graphs get `G.graph["type"] = "protein-protein"` (set on load if missing, for backwards compatibility).

---

## Section 2: Graph-to-Tensor Conversion

**New file:** `src/utils/hetero_graph_features.py`
**New function:** `nx_to_hetero_pyg_data(G)`

Returns a PyG `HeteroData` object:

- `data['gene'].x` — gene feature matrix. Tissue expression + AlphaFold (plddt, disorder, seq_len) + degree. Same normalization as `nx_to_pyg_data`, but only gene nodes.
- `data['disease'].x` — disease feature matrix. One-hot category (6 dims) + log(gene_count) + mean_score + max_score. ~9 features.
- `data['gene', 'interacts', 'gene'].edge_index` — STRING edges (bidirectional).
- `data['gene', 'associated', 'disease'].edge_index` — GWAS gene-disease edges.
- `data['disease', 'rev_associated', 'gene'].edge_index` — reverse edges for message passing.

**Return value:** `(hetero_data, gene_list, disease_list, gene_to_idx, disease_to_idx)`

No GWAS token encoding on this path.

---

## Section 3: Model

**New file:** `src/models/hetero_gnn.py`
**New class:** `HeteroGNNLinkPredictor`

**Architecture:**

Input projection: separate `nn.Linear` for gene features and disease features, both projecting to `hidden_channels`.

Message passing: `HeteroConv` wrapping `SAGEConv` per edge type:
- `SAGEConv` for `('gene', 'interacts', 'gene')`
- `SAGEConv` for `('gene', 'associated', 'disease')`
- `SAGEConv` for `('disease', 'rev_associated', 'gene')`

Sum aggregation across relation types for nodes receiving messages from multiple edge types. Stacked `num_layers` deep with ReLU + LayerNorm between layers.

**Decoder:** Dot product — `score = (z_gene[src] * z_disease[dst]).sum(dim=-1)` with sigmoid. Both node types are projected into the same `hidden_channels` space by the input projections and message passing layers, so the dot product captures learned similarity.

**Interface:**
- `encode(data) -> (z_gene, z_disease)` — separate embedding tensors per node type
- `decode(z_gene, z_disease, src, dst) -> scores` — sigmoid probabilities
- `decode_logits(z_gene, z_disease, src, dst) -> logits` — raw logits for loss

No RWSE, no GWAS tokens on this model. The graph structure carries the GWAS signal.

---

## Section 4: Trainer

**New file:** `src/models/hetero_trainer.py`
**New classes:** `HeteroTrainConfig`, `HeteroTrainer`

**Split strategy:** Only gene-disease edges are split into train/val/test. Gene-gene (STRING) edges remain in the message-passing graph at all times — they are structural context, not prediction targets.

**Negative sampling:** For each positive (gene, disease) pair, sample negatives by corrupting one side:
- Corrupt gene: replace with a random gene not associated with that disease
- Corrupt disease: replace with a random disease the gene is not associated with
- 1:1 ratio of both corruption types

Negatives are sampled against ALL positive gene-disease edges (not just the current split) to avoid false negatives.

**Training loop:** Full-batch by default. Mini-batch option for large graphs. Same early stopping, epoch callbacks, and metric reporting as existing trainer.

**Evaluation metrics:**
- ROC-AUC and Average Precision (same as existing)
- Hits@K: fraction of true edges appearing in top K predictions per disease. More interpretable for discovery use case.

**Config:** `HeteroTrainConfig` dataclass — similar to `TrainConfig` but without GWAS token params and with `neg_sampling_strategy` field.

---

## Section 5: UI Integration

### Training page (`3_Model_Training.py`)

New graph mode added to the radio selector: **"Gene-Disease network"**.

When selected, shows STRING confidence threshold and GWAS p-value threshold with a note explaining this builds a heterogeneous graph for gene-disease discovery.

Model selection is gated by graph type:
- Protein-protein graph loaded: shows GNN, Graph Transformer, VAE (existing)
- Gene-disease graph loaded: shows only "Heterogeneous GNN" (new)

### Results page (`4_Results.py`)

When a hetero model is loaded, the tabs become:

1. **"Predict Disease Genes"** — uses `model.decode(z_gene, z_disease, ...)` with real trained scores. Pick a disease, get ranked gene predictions. Replaces the cosine heuristic.
2. **"Predict Gene Diseases"** — reverse direction. Pick a gene, get ranked disease predictions.
3. **"Check Gene-Disease Link"** — enter a gene and disease, get the model's predicted association score.

Existing protein-protein model tabs remain unchanged. Graph type detection via `G.graph["type"]` and checkpoint metadata drives which UI renders.

---

## Section 6: Testing

**New test files with semantic assertions:**

| File | Tests |
|------|-------|
| `tests/data/test_disease_graph.py` | `build_disease_graph()` with small SQLite fixture. Asserts node types, edge types, feature shapes, p-value filtering. |
| `tests/utils/test_hetero_graph_features.py` | `nx_to_hetero_pyg_data()`. Asserts tensor shapes, bidirectional edges, normalization, no NaN, separate index mappings. |
| `tests/models/test_hetero_gnn.py` | Forward pass, encode, decode on synthetic HeteroData. Asserts shapes, score bounds, gradient flow. |
| `tests/models/test_hetero_trainer.py` | Training epochs, early stopping, metric validity, negative sampling correctness, gene-gene edges preserved across splits. |

---

## New Files Summary

| File | Purpose |
|------|---------|
| `src/data/bulk_datasets.py` | New `build_disease_graph()` method (additive) |
| `src/utils/hetero_graph_features.py` | `nx_to_hetero_pyg_data()` |
| `src/models/hetero_gnn.py` | `HeteroGNNLinkPredictor` |
| `src/models/hetero_trainer.py` | `HeteroTrainer` + `HeteroTrainConfig` |
| `pages/3_Model_Training.py` | New graph mode + model gating (additive) |
| `pages/4_Results.py` | Trained decode for predictions + reverse tab (additive) |
| `tests/data/test_disease_graph.py` | Graph builder tests |
| `tests/utils/test_hetero_graph_features.py` | Conversion tests |
| `tests/models/test_hetero_gnn.py` | Model tests |
| `tests/models/test_hetero_trainer.py` | Trainer tests |

Nothing in the existing protein-protein pipeline is modified.
