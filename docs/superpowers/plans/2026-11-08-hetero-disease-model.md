# Heterogeneous Gene–Disease Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a heterogeneous gene–trait graph model on gene–disease links, with the benchmark's hidden pairs held out of training, and compare it with the enrichment baseline on `scripts/benchmark_disease_genes.py`.

**Architecture:** The saved protein–protein graph is converted into a PyG `HeteroData` (gene nodes, trait nodes, STRING and GWAS edges) minus the benchmark's hidden pairs; a HeteroConv/SAGE encoder with a dot-product decoder is trained on gene–trait edges with per-epoch edge masking and early stopping on validation MRR; a shared scorer module turns a checkpoint into a genes × diseases score matrix for the benchmark CLI and the Results page.

**Tech Stack:** Python 3.11, torch 2.10, torch_geometric 2.7 (`HeteroData`, `HeteroConv`, `SAGEConv`), numpy, scipy, pytest. CPU by default, MPS optional.

**Spec:** `docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md` (revision 2026-11-08)

## Global Constraints

- Tests are CPU-only, run with bare `pytest`, and the suite must stay around 15 s: tiny synthetic graphs only.
- `node_list` is `sorted(G.nodes())`, as everywhere else; `trait_list` is the sorted trait names.
- Benchmark protocol unchanged: `make_holdout(index)` defaults (category `disease`, `min_known_genes=10`, 20 %, seed 0).
- Hidden pairs of the split appear nowhere in training: not in `('gene','has','trait')`, not in `('trait','rev_has','gene')`, not in labels.
- Existing export columns unchanged; `model_score` is appended as the last column.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A graph with no `gwas` attributes at all (protein-only build) → `nx_to_hetero_data` must return zero trait nodes and zero gene–trait edges, not crash (Task 1 test).
2. A checkpoint loaded against a graph with a different trait set → `hetero_score_matrix` must raise, not silently misalign embeddings (Task 4 test).
3. A benchmark disease missing from the model's trait list → the scorer must fill that column with zeros and warn, so the benchmark still runs (Task 4 test).
4. All negatives for a positive coincide with known edges (dense toy graph) → the trainer must not crash on an empty negative set (Task 3 test).
5. `predict_gene_disease_links` with `scores` whose shape does not match `(len(node_list), len(diseases))` → `ValueError` (Task 6 test).

---

### Task 1: Hetero graph converter

**Files:**
- Create: `src/utils/hetero_graph_features.py`
- Test: `tests/utils/test_hetero_graph_features.py`

**Interfaces:**
- Produces:
  ```python
  TRAIT_CATEGORIES = ("disease", "trait", "phenotype", "measurement", "biological_process", "other")

  @dataclass
  class HeteroBundle:
      data: HeteroData            # node types "gene", "trait"; edge types below
      node_list: list[str]; node_to_idx: dict[str, int]      # genes, sorted
      trait_list: list[str]; trait_to_idx: dict[str, int]    # traits, sorted
      trait_category: dict[str, str]

  def nx_to_hetero_data(G, *, exclude_pairs: set[tuple[str, str]] | None = None,
                        min_trait_genes: int = 1, trait_list: list[str] | None = None) -> HeteroBundle
  ```
  `data["gene"].x`: expression columns (+3 AlphaFold when any node has `mean_plddt`) + degree, normalised like `nx_to_pyg_data`. `data["trait"].x`: one-hot over `TRAIT_CATEGORIES` (6 dims; unknown → "other"). `data["gene","interacts","gene"].edge_index` both directions. `data["gene","has","trait"].edge_index` + `.edge_weight` (GWAS score) and `data["trait","rev_has","gene"]` its flip. Only gene nodes (`node_type == "gene"`) become nodes. Passing `trait_list` forces that order and raises `ValueError` when the graph's traits differ.

- [ ] **Step 1: Write the failing tests**

```python
def test_gene_features_and_string_edges_match_homogeneous_conversion()   # x shape (n_genes, 58-style width), edge_index has 2*E_string columns, equals set of nx_to_pyg_data edges
def test_trait_nodes_are_sorted_with_category_one_hot()                   # trait_list == sorted(traits); x[trait_to_idx["Asthma"]] == one-hot "disease"
def test_gene_trait_edges_carry_gwas_scores_in_both_directions()          # (gene,has,trait) edge for (A, Asthma) with weight 0.9; rev edge is flip
def test_exclude_pairs_removes_edges_in_both_directions()                 # exclude {("A","Asthma")}: pair absent from has and rev_has, other edges intact
def test_min_trait_genes_drops_small_traits()                              # min_trait_genes=2 drops a 1-gene trait from trait_list and its edges
def test_trait_list_argument_enforces_order_and_rejects_mismatch()        # reversed order honoured; unknown name -> ValueError
def test_graph_without_gwas_yields_no_traits()                             # 0 trait nodes, (0,) edge_index shapes (2, 0)
def test_non_gene_nodes_are_skipped()                                      # a node_type="disease" node is not in node_list
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/utils/test_hetero_graph_features.py -q` — Expected: ImportError on `src.utils.hetero_graph_features`.

- [ ] **Step 3: Implement `nx_to_hetero_data` in `src/utils/hetero_graph_features.py`**

Gene features: reuse the tissue/AlphaFold/degree logic of `nx_to_pyg_data` (copy the numeric part; do not import GWAS tokens). Collect `(gene, trait, score, category)` from `G.nodes[g]["gwas"]`, keeping the highest score per pair, skip pairs in `exclude_pairs`, count genes per trait *after* exclusion for `min_trait_genes`.

- [ ] **Step 4: Run tests to verify they pass** — `pytest tests/utils/test_hetero_graph_features.py -q`

- [ ] **Step 5: Commit** — `git add src/utils/hetero_graph_features.py tests/utils/test_hetero_graph_features.py && git commit -m "Add heterogeneous gene–trait graph conversion with pair exclusion"`

---

### Task 2: Hetero model

**Files:**
- Create: `src/models/hetero_gnn.py`
- Test: `tests/models/test_hetero_gnn.py`

**Interfaces:**
- Consumes: `HeteroBundle.data` from Task 1 (node types `gene`/`trait`, the three edge types, optional `edge_weight` ignored by SAGEConv).
- Produces:
  ```python
  class HeteroGNNLinkPredictor(nn.Module):
      def __init__(self, gene_in: int, num_traits: int, trait_in: int = 6, hidden_channels: int = 128,
                   num_layers: int = 2, dropout: float = 0.2, decoder: str = "dot")   # "dot" | "bilinear" | "mlp"
      def encode(self, data) -> tuple[Tensor, Tensor]                      # (z_gene [n_genes, h], z_trait [num_traits, h])
      def decode_logits(self, z_gene, z_trait, gene_idx, trait_idx) -> Tensor
      def decode(self, z_gene, z_trait, gene_idx, trait_idx) -> Tensor     # sigmoid
      def score_matrix(self, z_gene, z_trait, trait_idx) -> Tensor         # [n_genes, len(trait_idx)] logits
  ```
  Trait input = `Embedding(num_traits, h)(arange) + Linear(trait_in, h)(data["trait"].x)`. Per-trait bias `nn.Parameter(zeros(num_traits))` added by every decoder.

- [ ] **Step 1: Write the failing tests**

```python
def test_encode_returns_gene_and_trait_embeddings_of_hidden_size()        # shapes (n_genes, 16), (n_traits, 16)
def test_decode_scores_are_probabilities_and_match_logits()               # sigmoid(decode_logits) == decode, in [0,1]
@pytest.mark.parametrize("decoder", ["dot", "bilinear", "mlp"])
def test_score_matrix_equals_pairwise_decode_logits(decoder)              # score_matrix[:, j][i] == decode_logits(i, trait_idx[j])
def test_unknown_decoder_raises()                                          # ValueError
def test_gradients_reach_trait_embeddings_and_bias()                       # after loss.backward(), trait_embedding.weight.grad and trait_bias.grad not None/nonzero
def test_model_handles_graph_with_no_gene_trait_edges()                    # encode works when edge types have shape (2, 0)
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/models/test_hetero_gnn.py -q` — Expected: ImportError.

- [ ] **Step 3: Implement the model in `src/models/hetero_gnn.py`**

`HeteroConv({edge_type: SAGEConv((-1, -1) or (h, h), h) for the three types}, aggr="sum")` × `num_layers`; `LayerNorm` + `ReLU` + `Dropout` between layers only. Guard the empty-edge case: PyG `HeteroConv` skips edge types whose `edge_index` is empty only if the destination still gets a tensor; ensure both node types always come back (fall back to the projected input for a node type that received no messages).

- [ ] **Step 4: Run tests to verify they pass** — `pytest tests/models/test_hetero_gnn.py -q`

- [ ] **Step 5: Commit** — `git commit -m "Add HeteroGNNLinkPredictor with dot, bilinear and MLP decoders"`

---

### Task 3: Hetero trainer

**Files:**
- Create: `src/models/hetero_trainer.py`
- Test: `tests/models/test_hetero_trainer.py`

**Interfaces:**
- Consumes: `HeteroBundle` (Task 1), `HeteroGNNLinkPredictor` (Task 2), `make_holdout`, `mask_index`, `run_benchmark`, `HoldoutSplit` from `src.utils.benchmark`.
- Produces:
  ```python
  @dataclass
  class HeteroTrainConfig:
      epochs: int = 300; lr: float = 1e-3; weight_decay: float = 1e-5
      supervision_ratio: float = 0.3; neg_gene: int = 4; neg_trait: int = 1
      neg_sampling: str = "uniform"            # or "degree"
      val_fraction: float = 0.1; val_seed: int = 1
      early_stopping: bool = True; patience: int = 20
      seed: int = 42; device: str = "cpu"

  class HeteroTrainer:
      def __init__(self, model, config)
      def train(self, bundle, index, split: HoldoutSplit | None, graph, on_epoch=None) -> dict
          # history: {"train_loss": [...], "val_mrr": [...], "val_recall50": [...], "val_auc": [...], "best_epoch": int}
      def score_matrix(self, diseases: list[str]) -> np.ndarray          # (len(bundle.node_list), len(diseases)) from the final model, messages = all training edges
      def evaluate(self, split: HoldoutSplit) -> BenchmarkResult          # run_benchmark(self.score_matrix(split.diseases), ...)
      val_split: HoldoutSplit                                             # the validation hold-out actually used
  ```
  `train` builds `val_split = make_holdout(mask_index(index, split), categories=split.categories, min_known_genes=split.min_known_genes, hidden_fraction=config.val_fraction, seed=config.val_seed)` (when `split` is None: `make_holdout(index, hidden_fraction=val_fraction, seed=val_seed)`), removes the val pairs from the bundle's gene–trait edges, and keeps `train_edges` (2 × E) plus a set of all bundle edges for negative filtering.

- [ ] **Step 1: Write the failing tests** (toy graph: 30 genes, 6 traits, one disease with 12 known genes so the default protocol hides 2 and validation hides 1)

```python
def test_trainer_runs_requested_epochs_and_records_history()               # len(train_loss) == 3; keys present
def test_hidden_and_validation_pairs_are_absent_from_training_edges()      # trainer._train_edges has no (gene, trait) from split.hidden or trainer.val_split.hidden, in either direction of the message graph used in an epoch
def test_negatives_never_coincide_with_known_edges()                       # call trainer._sample_negatives(pos) and assert set ∩ known == ∅
def test_negative_sampling_survives_a_dense_graph()                        # every gene linked to the trait: returns possibly-empty tensors, loss step does not raise
def test_early_stopping_restores_best_weights()                            # patience=1, epochs=50: len(history) < 50 and state_dict == snapshot at best_epoch
def test_evaluate_returns_benchmark_result_with_hidden_ranks()             # isinstance BenchmarkResult; hidden_ranks keys == split.diseases
def test_degree_aware_negatives_prefer_high_degree_genes()                 # neg_sampling="degree": high-degree gene sampled more often than isolated one over 2000 draws
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/models/test_hetero_trainer.py -q` — Expected: ImportError.

- [ ] **Step 3: Implement `HeteroTrainer` in `src/models/hetero_trainer.py`**

Epoch: `perm = randperm(E)`; first `int(0.3·E)` are supervision, rest are messages (`edge_index` and flipped for `rev_has`); negatives via `_sample_negatives(pos) -> (gene_idx, trait_idx)` using a `torch.Generator` seeded from `config.seed`, filtered with a Python set of `gene_idx * num_traits + trait_idx` keys; loss BCE-with-logits over `[pos, neg]`. Validation: `score_matrix(val_split.diseases)` with messages = all training edges, `run_benchmark(matrix, graph, node_list, node_to_idx, index, val_split)` → `mrr`, `recall[50]`; `val_auc` = `roc_auc_score` on val positives vs one gene-corrupted negative each. Early stopping on `val_mrr` (ties do not reset patience). `_seed_everything` and device handling as in `Trainer`.

- [ ] **Step 4: Run tests to verify they pass** — `pytest tests/models/test_hetero_trainer.py -q`

- [ ] **Step 5: Commit** — `git commit -m "Add HeteroTrainer with edge masking, filtered negatives and MRR early stopping"`

---

### Task 4: Scorer module and training CLI

**Files:**
- Create: `src/utils/hetero_scorer.py`, `scripts/train_hetero_disease_model.py`
- Test: `tests/utils/test_hetero_scorer.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  ```python
  HETERO_MODEL_CLASS = "HeteroGNNLinkPredictor"; HETERO_MODEL_TYPE = "Hetero GNN"
  def save_hetero_checkpoint(path, *, model, model_kwargs, bundle, split, config, metrics, graph_name) -> None
  def load_hetero_checkpoint(path) -> dict                      # torch.load(weights_only=False); validates model_class
  def hetero_score_matrix(checkpoint: dict, graph, diseases, *, honest: bool = False) -> np.ndarray
      # rebuilds the bundle from graph (exclude the checkpoint split's hidden pairs when honest), checks trait_list
      # equality (ValueError otherwise), encodes, returns (len(node_list), len(diseases)); a disease not in
      # trait_list gets a zero column and a warnings.warn
  def find_hetero_checkpoints(checkpoint_dir) -> list[str]      # sorted paths matching hetero_gnn_*_checkpoint.pt
  ```
  CLI `scripts/train_hetero_disease_model.py`: `--data-dir --graph --out --hidden 128 --layers 2 --dropout 0.2 --decoder dot --epochs 300 --lr 1e-3 --weight-decay 1e-5 --neg-gene 4 --neg-trait 1 --neg-sampling uniform --min-trait-genes 1 --patience 20 --seed 42 --device cpu --no-holdout --json`. Prints history summary, the benchmark table (hetero alone and the combinations from Task 5) against enrichment, saves the checkpoint and `manager.save_training_run(model_type="Hetero GNN", ..., auc_roc=val_auc, avg_precision=benchmark recall@50)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_checkpoint_round_trip_reproduces_score_matrix(tmp_path)         # train 2 epochs on toy, save, load, hetero_score_matrix equals trainer.score_matrix (atol 1e-5)
def test_honest_mode_excludes_hidden_pairs_from_messages(tmp_path)        # honest=True vs False differ for a hidden gene's row
def test_mismatched_trait_list_raises(tmp_path)                           # graph with an extra trait -> ValueError
def test_unknown_disease_gets_zero_column_and_warning(tmp_path)           # pytest.warns(UserWarning); column all zeros
def test_find_hetero_checkpoints_lists_only_hetero_files(tmp_path)        # ignores gnn_*_checkpoint.pt
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/utils/test_hetero_scorer.py -q`

- [ ] **Step 3: Implement `src/utils/hetero_scorer.py`, then the CLI** (the CLI has no unit test; its smoke test is Task 7's real run).

- [ ] **Step 4: Run tests to verify they pass**, then `pytest -q` for the whole suite.

- [ ] **Step 5: Commit** — `git commit -m "Add hetero checkpoint scorer and training CLI"`

---

### Task 5: Benchmark combinations and hub concentration

**Files:**
- Modify: `src/utils/benchmark.py`, `scripts/benchmark_disease_genes.py`
- Test: `tests/utils/test_benchmark.py`

**Interfaces:**
- Produces:
  ```python
  def rank_average(a: np.ndarray, b: np.ndarray) -> np.ndarray         # per column: mean of descending ranks, negated so higher is better
  def zscore_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray           # per column z-scores (std floored at 1e-9), summed
  def fallback(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray
      # primary where primary > 0, else secondary shifted below primary's minimum positive value
  def hub_concentration(scores: np.ndarray, graph, node_list, node_to_idx, split, k: int = 10) -> dict
      # {"distinct_top10": int, "max_rank1": int} over allowed candidates per disease
  ```
  `run_benchmark` on a matrix fills `result.extra` with `hub_concentration(...)`; on a callable, from the rows' top-k. `BenchmarkResult.summary()` appends `distinct top-10 N, max rank-1 M`.
  CLI: `--hetero-checkpoint PATH`; new scorer names `hetero`, `hetero+enrichment`, `hetero+enrichment-z`, `enrichment>hetero` (all need the hetero checkpoint; enrichment matrix from `enrichment_matrix` on the masked index). `reference_scorers(..., hetero_scores=None)` gains these names.

- [ ] **Step 1: Write the failing tests**

```python
def test_rank_average_is_symmetric_and_orders_by_mean_rank()
def test_zscore_sum_standardises_each_column()
def test_fallback_uses_secondary_only_where_primary_is_zero_and_keeps_it_below()
def test_hub_concentration_counts_distinct_top_genes_and_repeat_winners()  # scores where one gene is first for all diseases -> max_rank1 == n_diseases
def test_run_benchmark_reports_hub_concentration_in_extra()
def test_reference_scorers_accepts_hetero_matrix_names()
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/utils/test_benchmark.py -q`

- [ ] **Step 3: Implement** the helpers and CLI additions.

- [ ] **Step 4: Run tests to verify they pass**, then full `pytest -q`.

- [ ] **Step 5: Commit** — `git commit -m "Benchmark: combination scorers, hub concentration, hetero checkpoint option"`

---

### Task 6: Optional model scorer in discovery and export

**Files:**
- Modify: `src/utils/discovery.py` (`predict_gene_disease_links`), `src/utils/discovery_export.py`
- Test: `tests/utils/test_discovery.py`, `tests/utils/test_discovery_export.py`

**Interfaces:**
- Produces:
  - `predict_gene_disease_links(z_norm, node_list, node_to_idx, graph, index, diseases, top_k, max_listed_interactors=10, *, scores=None)`: `scores` is `None` or an array `(len(node_list), len(diseases))` (ValueError otherwise). Ranking key with scores: `(-scores, -enrichment, -similarity)`; every row gets `"model_score": round(float, 4)` or `None`.
  - `PREDICTED_COLUMNS` ends with `"model_score"`; `PREDICTED_COLUMN_DOCS["model_score"]` describes it ("logit of the hetero gene–disease model when it produced the ranking; empty otherwise").
  - `build_manifest(..., scorer: str = "enrichment", scorer_checkpoint: str | None = None)`: `manifest["scorer"] = {"name": scorer, "checkpoint": scorer_checkpoint}`; `manifest["method"]` is `METHOD_TEXT` for `"enrichment"` and `HETERO_METHOD_TEXT` for `"hetero-gnn"`; `render_readme` adds `- Ranking scorer: …` under Model.

- [ ] **Step 1: Write the failing tests**

```python
# test_discovery.py
def test_predict_links_ranks_by_model_scores_when_given()                  # scores put G first; rows[0]["gene"] == "G", model_score == 0.7; enrichment still populated
def test_predict_links_without_scores_has_empty_model_score()              # row["model_score"] is None
def test_predict_links_rejects_mismatched_scores_shape()                   # ValueError
# test_discovery_export.py
def test_predicted_columns_end_with_model_score_and_keep_existing_order()  # PREDICTED_COLUMNS[:-1] == old tuple
def test_manifest_records_scorer_and_hetero_method_text()                  # scorer block; "heterogeneous" in method for hetero-gnn
def test_bundle_csv_writes_model_score_column(tmp_path)                    # header's last column == "model_score"; empty for None
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run tests, then full `pytest -q`**

- [ ] **Step 5: Commit** — `git commit -m "Allow a model score matrix to rank discovery candidates; record the scorer in the export"`

---

### Task 7: Train the real checkpoint and benchmark it

**Files:** none in the repo beyond `docs/` numbers; checkpoint under the main checkout's `.datasets/trained_models/` (git-ignored).

- [ ] **Step 1:** `python scripts/train_hetero_disease_model.py --data-dir <main>/.datasets --graph full_s800_gwas --device cpu --json <scratch>/hetero_run1.json` (background; ~0.5 s/epoch).
- [ ] **Step 2:** Read the table. If top-10 lists are hub-dominated (distinct top-10 ≪ enrichment's, max rank-1 ≫), rerun with `--decoder bilinear`, `--decoder mlp`, `--neg-sampling degree`; if 1–2-gene traits hurt, `--min-trait-genes 3`. Keep every run's JSON.
- [ ] **Step 3:** `python scripts/benchmark_disease_genes.py --graph full_s800_gwas --checkpoint <gnn ckpt> --hetero-checkpoint <best hetero ckpt> --scorers enrichment,hetero,hetero+enrichment,hetero+enrichment-z,enrichment>hetero` and record the table.
- [ ] **Step 4:** Write the numbers into the shared notes file and the PR description (next to the baseline).

---

### Task 8: Results page, README, PR

**Files:**
- Modify: `pages/4_Results.py`, `README.md`, `docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md` (outcome section)

- [ ] **Step 1:** Results page: `find_hetero_checkpoints(checkpoint_dir)`; when non-empty, a `st.selectbox("Ranking score", ["Network enrichment (default)", "Hetero gene–disease model: <name>"...])` in the disease tab and the export form; on the hetero choice compute `hetero_score_matrix(ckpt, graph, diseases)` (cached with `st.cache_resource` keyed on path + graph token) and pass `scores=`; pass `scorer="hetero-gnn"`, `scorer_checkpoint=path` to `build_manifest`. Only wire the hetero option as default-eligible if Task 7 beat enrichment; otherwise it is still offered but labelled "(experimental, below enrichment on the benchmark)".
- [ ] **Step 2:** README: training command, benchmark table with the hetero rows, scorer selector, layout lines; spec: "Outcome" section with the numbers and the decision.
- [ ] **Step 3:** `pytest -q` green; `python -c "import ast,sys; ast.parse(open('pages/4_Results.py').read())"`.
- [ ] **Step 4:** Commit, push, update the PR description with the table; dispatch one fresh code-review subagent on the branch and address findings.
