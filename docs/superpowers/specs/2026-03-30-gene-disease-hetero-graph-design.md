# Gene–Disease Heterogeneous Graph Link Prediction

**Date:** 2026-03-30, revised 2026-11-08
**Status:** Approved (revision approved 2026-11-08; defaults confirmed by the user, who asked to proceed without further questions)

## Problem

The current pipeline trains link prediction on protein–protein edges only. Disease-gene prediction is a post-hoc heuristic over the resulting embeddings. The scoring review of 2026-11-08 measured this on a held-out benchmark (`src/utils/benchmark.py`): ranking genes by cosine similarity to a disease's mean embedding recovers hidden known genes at recall@50 0.007 (chance 0.0025) and is dominated by hub genes; plain network guilt-by-association (hypergeometric enrichment of known genes among a gene's STRING partners) reaches 0.042 and is now the production ranking. The models learn who interacts with whom, not who is involved in the same disease, and their protein–protein AUC says nothing about disease ranking.

## Goal

Train a model on the objective we care about, **which genes are involved in the same disease**, and measure it on the same benchmark:

- Given a disease, rank candidate genes (the Predictions page and the export bundle).
- Given a gene, rank candidate diseases (available through the same decoder; no UI in this revision).

Acceptance is the benchmark, not AUC: enrichment stays the default scorer unless the model beats recall@50 0.042 / MRR 0.086 on the same held-out split. The combination of the model's score with enrichment is reported too.

Everything is additive. The protein–protein pipeline is unchanged except for the optional `scores` argument on `predict_gene_disease_links` and one appended export column.

---

## Section 1: Graph construction — from the saved graph, not from SQLite

The March draft added `build_disease_graph()` on `BulkDatasetManager`. This revision instead converts the **existing saved protein–protein graph** (`full_s800_gwas` and friends), whose gene nodes already carry the GWAS associations as a `gwas` attribute. Reasons: no SQLite rebuild, the gene set and STRING edges are exactly those of the benchmark and the production scorer, and every saved graph works.

**Node types**

- `gene` (20,306 on `full_s800_gwas`). Features as in `nx_to_pyg_data`: GTEx expression per tissue (54), AlphaFold (mean pLDDT / 100, disordered fraction, sequence length / 5000) when present, STRING degree (normalised). No GWAS tokens: the GWAS signal is structural.
- `trait` (14,286). **All** GWAS traits become nodes, not only the 724 of category *disease*. Trait and measurement associations are 93 % of the 305,152 gene–trait pairs (a gene's BMI or lipid associations are what predicts type 2 diabetes genes), so they are both message-passing context and training signal. Features: category one-hot over `disease, trait, phenotype, measurement, biological_process, other` (6 dims) plus a learnable per-trait embedding inside the model. No gene-count or score statistics: they would have to be recomputed from the training edges to avoid leakage, and the embedding carries the same information.

**Edge types**

- `('gene', 'interacts', 'gene')`: STRING edges, both directions. Message passing only, never a prediction target.
- `('gene', 'has', 'trait')` and `('trait', 'rev_has', 'gene')`: one pair per (gene, trait) association in the `gwas` attribute, with `edge_weight` = the GWAS score (already `-log10 p / 50` clamped to [0, 1]).

No trait–trait edges. Traits with fewer than `min_trait_genes` genes can be dropped from the graph (default: keep all; knob provided because 1–2-gene traits dominate the node count).

**Hold-out integration (critical).** The converter takes an optional set of (gene, trait) pairs to **exclude entirely**: those edges appear in neither direction of the message-passing graph and in no label set. The training CLI passes the benchmark split's hidden pairs (`make_holdout(index)` with the default protocol), so the trained model has never seen the pairs the benchmark asks it to recover. The split is saved in the checkpoint and the benchmark is run with that split, never a fresh one.

**Output:** a `HeteroBundle` with the PyG `HeteroData`, `node_list` / `node_to_idx` (sorted gene nodes, identical to `nx_to_pyg_data`'s order for graphs with gene nodes only), and `trait_list` / `trait_to_idx` (sorted trait names). The checkpoint stores `trait_list`; loading verifies it matches the graph.

---

## Section 2: Model

`src/models/hetero_gnn.py`, class `HeteroGNNLinkPredictor`.

- Input projections: `Linear(gene_in, hidden)` for genes; for traits `Embedding(num_traits, hidden)` added to `Linear(trait_in, hidden)` of the one-hot category.
- Message passing: `num_layers` (default 2) `HeteroConv` layers, `SAGEConv` per edge type, `aggr="sum"` across relations, `LayerNorm` + `ReLU` + dropout (default 0.2) between layers, none after the last.
- Decoder, selectable: `"dot"` (default) = dot product plus a learnable per-trait bias; `"bilinear"` = `z_gene · W · z_trait` + bias; `"mlp"` = 2-layer MLP on `[z_gene, z_trait, z_gene * z_trait]`. The alternatives exist because the dot-product decoder on the protein–protein GNN collapsed the embedding space and produced the hub artifact; if the hetero model's top-10 lists are again dominated by a few genes, try them before concluding the approach fails.
- Interface: `encode(data) -> (z_gene, z_trait)`; `decode_logits(z_gene, z_trait, gene_idx, trait_idx) -> logits`; `decode(...) -> sigmoid`; `score_matrix(z_gene, z_trait, trait_idx) -> (n_genes, len(trait_idx))` logits, the ranking score.

Hidden size default 128.

---

## Section 3: Trainer

`src/models/hetero_trainer.py`, `HeteroTrainConfig` and `HeteroTrainer`.

- **Targets:** gene–trait edges only. STRING edges are always in the message-passing graph.
- **Validation:** from the visible known genes of the benchmark diseases, a second seeded hold-out (`hidden_fraction` 0.1, seed 1, same protocol) gives validation (gene, disease) pairs. They are removed from the training edges (messages and labels). Early stopping monitors validation **MRR** computed with `run_benchmark` on the model's score matrix (patience 20, best weights restored). Validation AUC against gene-corrupted negatives is also recorded so the run fits the existing training-runs table.
- **Per-epoch edge masking:** each epoch a fresh random 30 % of training gene–trait edges are supervision targets and the other 70 % carry messages (both directions). The model never scores an edge it can see, and every edge is supervised over time.
- **Negatives:** per positive (g, t), 4 negatives with a corrupted gene and 1 with a corrupted trait (uniform by default; `neg_sampling="degree"` draws genes proportionally to their gene–trait degree, the degree-aware option). Negatives that coincide with any known edge of the bundle are dropped (no filtering against the benchmark's hidden pairs: the trainer does not know them).
- **Loss / optimiser:** BCE with logits; Adam, lr 1e-3, weight decay 1e-5; full batch; up to 300 epochs; seed 42; `device` cpu or mps (0.44 s / 0.34 s per epoch at full size).
- `train(bundle, index, split, on_epoch=None) -> history` and `evaluate(split) -> BenchmarkResult`.

---

## Section 4: Training CLI and checkpoint

`scripts/train_hetero_disease_model.py`: loads a saved graph, builds the benchmark split (or none with `--no-holdout`, for a production model after the honest one has been validated), converts, trains, benchmarks and saves `.datasets/trained_models/hetero_gnn_<stamp>_checkpoint.pt` with `model_class`, `model_kwargs`, `model_state`, `model_type = "Hetero GNN"`, `graph_name`, `train_config`, `split` (`HoldoutSplit.to_dict()`, or null), `trait_list`, `metrics` (validation and benchmark numbers), and records a training run in the database.

`src/utils/hetero_scorer.py`: `load_hetero_checkpoint(path)` and `hetero_score_matrix(checkpoint, graph, diseases, honest=False)`, the shared way to get a (n_nodes × n_diseases) score matrix in `node_list` order for the benchmark CLI and the Results page. `honest=True` excludes the checkpoint's hidden pairs from the message-passing graph (benchmark); production scoring uses every known edge.

---

## Section 5: Benchmark additions

`scripts/benchmark_disease_genes.py` gains `--hetero-checkpoint` and the scorers `hetero`, `hetero+enrichment` (average of ranks), `hetero+enrichment-z` (sum of per-disease z-scores) and `enrichment>hetero` (enrichment where it is non-zero, the model for genes with no known partner). Every result also reports **hub concentration**: distinct genes in the top-10 lists across diseases and the largest number of diseases any one gene ranks first for.

---

## Section 6: Production integration (only if the model beats enrichment)

- `predict_gene_disease_links(..., scores=None)`: with a (n_nodes × n_diseases) matrix, rank by it, then enrichment, then cosine; each row gains `model_score` (empty when ranking by enrichment). **Existing columns are unchanged; `model_score` is appended.**
- `build_manifest(..., scorer="enrichment", scorer_checkpoint=None)`: the manifest's `scorer` block and method text name which scorer produced the bundle; the README in the bundle shows it.
- Results page: when a hetero checkpoint exists in `.datasets/trained_models`, the Predict Disease Genes tab and the export form offer **Ranking score: Network enrichment (default) / Hetero gene–disease model**. The Training page gets no new mode in this revision; training is the CLI.

If the model does not beat enrichment, the scorer stays available in the CLI and the benchmark only.

---

## Section 7: Testing

| File | Tests |
|------|-------|
| `tests/utils/test_hetero_graph_features.py` | node/trait indexing, feature shapes, STRING edges bidirectional, gene–trait edges and weights, excluded pairs absent in both directions, `min_trait_genes`, trait-list check |
| `tests/models/test_hetero_gnn.py` | encode shapes, decoder variants, score bounds, score matrix equals decoder, gradients flow to trait embeddings |
| `tests/models/test_hetero_trainer.py` | runs epochs, hidden pairs never appear in supervision or messages, validation pairs removed from training, negatives never coincide with known edges, early stopping restores best, evaluate returns a `BenchmarkResult` |
| `tests/utils/test_hetero_scorer.py` | checkpoint round trip, score-matrix shape and order, honest mode drops hidden edges |
| `tests/utils/test_benchmark.py` | combination scorers, hub concentration |
| `tests/utils/test_discovery.py`, `test_discovery_export.py` | `scores` ranking and `model_score`, manifest scorer block, appended column |

## New / modified files

| File | Purpose |
|------|---------|
| `src/utils/hetero_graph_features.py` | `nx_to_hetero_data()` |
| `src/models/hetero_gnn.py` | `HeteroGNNLinkPredictor` |
| `src/models/hetero_trainer.py` | `HeteroTrainer`, `HeteroTrainConfig` |
| `src/utils/hetero_scorer.py` | checkpoint loading and score matrices |
| `scripts/train_hetero_disease_model.py` | training CLI |
| `scripts/benchmark_disease_genes.py` | hetero and combination scorers, hub concentration |
| `src/utils/benchmark.py` | combination helpers, hub concentration |
| `src/utils/discovery.py`, `src/utils/discovery_export.py`, `pages/4_Results.py` | optional model scorer, appended column, manifest scorer |
