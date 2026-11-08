# Hetero GNN Training-Page Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the gene–disease hetero model from the Model Training page, on this computer's CPU, the Mac's GPU or a Modal cloud GPU, with the same pipeline and outputs as the CLI.

**Architecture:** The CLI's pipeline (hold-out → convert → train → benchmark against enrichment → save checkpoint + run record) moves into `src/models/hetero_pipeline.py`; the CLI, the page's local path and a new Modal remote function all call it. The remote path ships a picklable payload (bundle, index, split, model kwargs, config) and gets weights, history and metrics back; the benchmark and the save then run locally, so a cloud run leaves exactly what a local run leaves. The page adds a fourth model option with its own settings and shows validation metrics, the benchmark table and where the checkpoint went.

**Tech Stack:** Streamlit, PyTorch, PyG `HeteroData`, Modal, pytest.

**Spec:** `docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md` (Sections 3, 4, 6 and the new Section 9 written in Task 4).

## Global Constraints

- Checkpoints are written by `save_hetero_checkpoint` to `<data_dir>/trained_models/hetero_gnn_<YYYYmmdd_HHMMSS>_checkpoint.pt` (`model_class = "HeteroGNNLinkPredictor"`, `model_type = "Hetero GNN"`, split stored when a hold-out was used).
- Training runs are recorded with `model_type = HETERO_MODEL_TYPE` (`"Hetero GNN"`): the Results page filters that value out of the protein–protein loader.
- Benchmark protocol unchanged: `make_holdout(index)` defaults (disease traits with ≥ 10 genes, 20 % hidden, seed 0); hidden pairs excluded from the bundle; scorers `"enrichment"` + `HETERO_SCORERS`; AUC column of the run = validation AUC, precision column = benchmark recall@50 (validation recall@50 without a hold-out).
- The page never calls `session.set_training_results` for a hetero model (the Results page would treat it as a protein–protein model).
- The protein–protein training flow on the page (controls, Modal call, checkpoint contents) is unchanged.
- Modal functions follow the existing pattern: one `@app.function` per GPU type (T4 3600 s, A10G 3600 s, A100 7200 s), source tree shipped per job, `cuda` when available.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A network built without GWAS associations (no gene–trait edges): a person expects a message telling them to rebuild with "Include disease associations", not a NaN loss or a tensor error. Test in Task 1 (`test_prepare_refuses_graph_without_gwas`).
2. A small network where no disease keeps 10+ known genes: with the hold-out on, a person expects a message (untick the hold-out or build a larger network), not an empty benchmark. Test in Task 1 (`test_prepare_refuses_holdout_when_no_disease_qualifies`).
3. A cloud run must leave the same model a local run leaves: the payload round trip reproduces local training weight for weight with the same seed. Test in Task 3 (`test_payload_round_trip_reproduces_local_training`).
4. The checkpoint the page writes must be usable by the Predictions page: `load_hetero_checkpoint` + `hetero_score_matrix(..., honest=True)` on the saved file. Test in Task 1 (`test_save_hetero_run_writes_loadable_checkpoint_and_records_run`).
5. The run record must carry `model_type = "Hetero GNN"` and the benchmark recall@50 in the precision column, so the runs table and the Results page's filter behave. Same test as 4.

---

### Task 1: Shared pipeline module

**Files:**
- Create: `src/models/hetero_pipeline.py`
- Test: `tests/models/test_hetero_pipeline.py`

**Interfaces:**
- Consumes: `nx_to_hetero_data`, `TRAIT_CATEGORIES`, `GENE_GENE`, `GENE_TRAIT` (`src/utils/hetero_graph_features.py`); `make_holdout`, `reference_scorers`, `HETERO_SCORERS`, `BenchmarkResult`, `HoldoutSplit` (`src/utils/benchmark.py`); `index_disease_genes` (`src/utils/discovery.py`); `HeteroTrainer`, `HeteroTrainConfig` (`src/models/hetero_trainer.py`); `HeteroGNNLinkPredictor` (`src/models/hetero_gnn.py`); `save_hetero_checkpoint`, `HETERO_MODEL_TYPE`, `align_gene_rows`, `score_from_encoding`, `HeteroEncoding` (`src/utils/hetero_scorer.py`); `BulkDatasetManager.save_training_run`.
- Produces (all in `src/models/hetero_pipeline.py`):

```python
@dataclass
class HeteroPrepared:
    bundle: HeteroBundle
    index: dict                      # index_disease_genes(graph, node_list), unmasked
    split: HoldoutSplit | None
    node_list: list[str]             # sorted(graph.nodes()): benchmark rows
    node_to_idx: dict[str, int]
    n_hidden: int                    # hidden (gene, disease) pairs removed from the bundle
    @property n_string -> int        # undirected STRING edges of the bundle
    @property n_assoc -> int         # gene–trait edges of the bundle

def prepare_hetero_training(graph, *, holdout: bool = True, min_trait_genes: int = 1) -> HeteroPrepared
    # ValueError("... no GWAS associations ...") when the bundle has 0 gene–trait edges
    # ValueError("... no disease has 10+ known genes ...") when holdout and split.diseases is empty
def hetero_model_kwargs(bundle, *, hidden: int = 128, layers: int = 2, dropout: float = 0.2, decoder: str = "dot") -> dict
    # gene_in, num_traits, trait_in=len(TRAIT_CATEGORIES), hidden_channels, num_layers, dropout, decoder
def gene_only_graph(node_list) -> nx.Graph        # every node with node_type="gene", no edges (validation stand-in)

@dataclass
class HeteroTrained:
    model: HeteroGNNLinkPredictor    # on CPU
    history: dict
    metrics: dict                    # epochs_run, best_epoch, val_mrr, val_recall50, val_auc, train_minutes

def train_hetero(bundle, index, split, model_kwargs: dict, config: HeteroTrainConfig, on_epoch=None) -> HeteroTrained
    # torch.manual_seed(config.seed) before building the model; trainer.train(bundle, index, split, gene_only_graph(bundle.node_list), on_epoch=on_epoch)
def benchmark_hetero(model, prepared: HeteroPrepared, graph) -> dict[str, BenchmarkResult]
    # {} when prepared.split is None; otherwise reference_scorers(["enrichment", *HETERO_SCORERS], ...) with
    # hetero_scores = align_gene_rows(score_from_encoding(HeteroEncoding(bundle, model, z_gene, z_trait, bundle.node_list), split.diseases), bundle.node_list, prepared.node_list)
def describe_hetero_params(model_kwargs: dict, config: HeteroTrainConfig, *, holdout: bool, gpu: str | None = None) -> str
    # "hidden=128, layers=2, decoder=dot, dropout=0.2, neg=4g+1t/uniform, lr=0.001, wd=1e-05, epochs=300, seed=42, holdout=benchmark" (+ ", gpu=A100"); holdout=no without a hold-out

@dataclass
class HeteroSaved:
    checkpoint_path: str
    run_id: int | None
    model_name: str                  # hetero_gnn_<stamp>

def save_hetero_run(manager, *, model, model_kwargs: dict, prepared: HeteroPrepared, config: HeteroTrainConfig,
                    metrics: dict, benchmark: dict | None, graph_name: str, min_trait_genes: int, params: str,
                    out: str | None = None, save_run: bool = True) -> HeteroSaved
    # metrics copy gains "benchmark": {name: result.to_dict()} when benchmark is non-empty; checkpoint via save_hetero_checkpoint;
    # run via manager.save_training_run(model_type=HETERO_MODEL_TYPE, nodes=len(bundle.node_list), edges=n_string,
    # features=gene_in, epochs_run, auc_roc=val_auc, avg_precision=benchmark hetero recall@50 or val_recall50, model=model.cpu(), model_name)
```

- [ ] **Step 1: Write the failing tests**

Build the toy graph with `make_graph` from `tests/utils/test_hetero_scorer.py` (24 genes in a ring, Asthma on 12 genes, BMI on 8). Every training config in this file uses `HeteroTrainConfig(epochs=3, early_stopping=False)`.

```python
def test_prepare_builds_split_and_bundle_without_hidden_pairs():
    prepared = prepare_hetero_training(make_graph())
    assert prepared.split.diseases == ["Asthma"]
    assert prepared.n_hidden == sum(len(g) for g in prepared.split.hidden.values()) > 0
    assert prepared.n_assoc == 20 - prepared.n_hidden
    assert prepared.n_string == 24
    assert prepared.node_list == sorted(make_graph().nodes())
    production = prepare_hetero_training(make_graph(), holdout=False)
    assert production.split is None and production.n_hidden == 0 and production.n_assoc == 20

def test_prepare_refuses_graph_without_gwas():
    G = make_graph()
    for n in G.nodes: G.nodes[n].pop("gwas", None)
    with pytest.raises(ValueError, match="GWAS"):
        prepare_hetero_training(G)

def test_prepare_refuses_holdout_when_no_disease_qualifies():
    G = make_graph()
    for g in GENES[6:12]: G.nodes[g].pop("gwas")            # Asthma keeps 6 genes
    with pytest.raises(ValueError, match="10"):
        prepare_hetero_training(G)
    assert prepare_hetero_training(G, holdout=False).split is None

def test_train_hetero_returns_metrics_and_cpu_model():
    prepared = prepare_hetero_training(make_graph())
    kwargs = hetero_model_kwargs(prepared.bundle, hidden=8)
    trained = train_hetero(prepared.bundle, prepared.index, prepared.split, kwargs, HeteroTrainConfig(epochs=3, early_stopping=False))
    assert set(trained.metrics) == {"epochs_run", "best_epoch", "val_mrr", "val_recall50", "val_auc", "train_minutes"}
    assert trained.metrics["epochs_run"] == 3 == len(trained.history["train_loss"])
    assert all(p.device.type == "cpu" for p in trained.model.parameters())

def test_benchmark_hetero_reports_enrichment_and_hetero_scorers():
    G = make_graph(); prepared = prepare_hetero_training(G)
    trained = train_hetero(...)
    results = benchmark_hetero(trained.model, prepared, G)
    assert list(results) == ["enrichment", *HETERO_SCORERS]
    assert all(isinstance(r, BenchmarkResult) and r.n_diseases == 1 for r in results.values())
    assert benchmark_hetero(trained.model, prepare_hetero_training(G, holdout=False), G) == {}

def test_save_hetero_run_writes_loadable_checkpoint_and_records_run(tmp_path):
    manager = BulkDatasetManager(str(tmp_path))
    ... prepared, kwargs, config, trained, bench = benchmark_hetero(...)
    params = describe_hetero_params(kwargs, config, holdout=True)
    saved = save_hetero_run(manager, model=trained.model, model_kwargs=kwargs, prepared=prepared, config=config,
                            metrics=trained.metrics, benchmark=bench, graph_name="toy", min_trait_genes=1, params=params)
    assert os.path.dirname(saved.checkpoint_path) == str(tmp_path / "trained_models")
    assert is_hetero_checkpoint_path(saved.checkpoint_path) and saved.model_name == os.path.basename(saved.checkpoint_path)[: -len("_checkpoint.pt")]
    checkpoint = load_hetero_checkpoint(saved.checkpoint_path)
    assert checkpoint["split"]["hidden"] == prepared.split.hidden
    assert checkpoint["metrics"]["benchmark"]["hetero"]["recall@50"] == bench["hetero"].recall[50]
    assert hetero_score_matrix(checkpoint, G, ["Asthma"], honest=True).shape == (len(prepared.node_list), 1)
    run = manager.list_training_runs()[0]
    assert run["id"] == saved.run_id and run["model_type"] == "Hetero GNN" and run["parameters"] == params
    assert run["avg_precision"] == round(bench["hetero"].recall[50], 6) and run["auc_roc"] == round(trained.metrics["val_auc"], 6)
    assert save_hetero_run(..., benchmark=None, save_run=False, out=str(tmp_path / "x.pt")).run_id is None and (tmp_path / "x.pt").exists()

def test_describe_hetero_params_matches_cli_format():
    kwargs = dict(gene_in=3, num_traits=2, trait_in=6, hidden_channels=128, num_layers=2, dropout=0.2, decoder="dot")
    config = HeteroTrainConfig()
    assert describe_hetero_params(kwargs, config, holdout=True) == \
        "hidden=128, layers=2, decoder=dot, dropout=0.2, neg=4g+1t/uniform, lr=0.001, wd=1e-05, epochs=300, seed=42, holdout=benchmark"
    assert describe_hetero_params(kwargs, config, holdout=False, gpu="A100").endswith("holdout=no, gpu=A100")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/models/test_hetero_pipeline.py -q`
Expected: FAIL at import (`No module named src.models.hetero_pipeline`).

- [ ] **Step 3: Implement `src/models/hetero_pipeline.py`** with the Interfaces above. `prepare_hetero_training` counts gene–trait edges with `bundle.data[GENE_TRAIT].edge_index.size(1)`; its two error messages name the fix ("rebuild the network with *Include disease associations*" / "untick the benchmark hold-out or build a larger network").

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/models/test_hetero_pipeline.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/hetero_pipeline.py tests/models/test_hetero_pipeline.py
git commit -m "Extract the hetero training pipeline into src/models/hetero_pipeline.py"
```

### Task 2: CLI on the pipeline

**Files:**
- Modify: `scripts/train_hetero_disease_model.py` (`main`)
- Create: `tests/scripts/__init__.py`, `tests/scripts/test_train_hetero_cli.py`

**Interfaces:**
- Consumes: Task 1's `prepare_hetero_training`, `hetero_model_kwargs`, `train_hetero`, `benchmark_hetero`, `describe_hetero_params`, `save_hetero_run`.
- Produces: unchanged CLI arguments and JSON (`args`, `checkpoint`, `run_id`, `metrics`, `history`).

- [ ] **Step 1: Write the failing test**

Load the script as a module with `importlib.util.spec_from_file_location("train_hetero_cli", "<repo>/scripts/train_hetero_disease_model.py")`.

```python
def test_cli_trains_benchmarks_and_records_a_run_on_a_saved_graph(tmp_path):
    BulkDatasetManager(str(tmp_path)).save_graph(make_graph(), "toy")
    out = tmp_path / "run.json"
    assert cli.main(["--data-dir", str(tmp_path), "--graph", "toy", "--hidden", "8", "--epochs", "2", "--json", str(out)]) == 0
    report = json.loads(out.read_text())
    assert os.path.exists(report["checkpoint"]) and report["run_id"] == 1
    assert "hetero" in report["metrics"]["benchmark"] and report["metrics"]["epochs_run"] == 2
    assert BulkDatasetManager(str(tmp_path)).list_training_runs()[0]["model_type"] == "Hetero GNN"

def test_cli_no_holdout_skips_the_benchmark_and_no_save_run_skips_the_record(tmp_path):
    ... "--no-holdout", "--no-save-run", "--out", str(tmp_path / "prod.pt") → 0; "benchmark" not in metrics; run_id is None; prod.pt exists
    and load_hetero_checkpoint(prod.pt)["split"] is None

def test_cli_reports_a_missing_graph(tmp_path, capsys):
    assert cli.main(["--data-dir", str(tmp_path), "--graph", "nope"]) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/scripts -q`
Expected: the first two FAIL (the current `main` works end to end, so they fail only if the pipeline changes behaviour) — if they PASS against the old `main`, that is the finding: the tests pin existing behaviour, note it in the ledger and continue.

- [ ] **Step 3: Rewrite `main` on the pipeline**: `prepare_hetero_training(graph, holdout=not args.no_holdout, min_trait_genes=args.min_trait_genes)`, `hetero_model_kwargs(...)`, `HeteroTrainConfig(...)` as today, `train_hetero(..., on_epoch=None)`, `benchmark_hetero`, `describe_hetero_params(..., holdout=not args.no_holdout)`, `save_hetero_run(manager, ..., out=args.out, save_run=not args.no_save_run)`. Keep the printed lines (graph summary, model/config, training summary, one line per scorer, checkpoint path, run id). The `ValueError`s of `prepare_hetero_training` print to stderr and return 1.

- [ ] **Step 4: Run the tests and the suite**

Run: `venv/bin/python -m pytest tests/scripts tests/models/test_hetero_pipeline.py -q` then `venv/bin/python -m pytest -q`
Expected: all pass; the suite count rises by 10 (169 → 172 including Task 1).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_hetero_disease_model.py tests/scripts
git commit -m "Run the hetero training CLI on the shared pipeline"
```

### Task 3: Remote payload and Modal function

**Files:**
- Modify: `src/models/hetero_pipeline.py` (payload helpers), `src/models/modal_app.py`
- Test: `tests/models/test_hetero_pipeline.py`

**Interfaces:**
- Produces in `hetero_pipeline.py`:

```python
def hetero_training_payload(prepared: HeteroPrepared, model_kwargs: dict, config: HeteroTrainConfig) -> dict
    # {"bundle": prepared.bundle, "index": prepared.index, "split": split.to_dict() or None,
    #  "model_kwargs": dict, "config": asdict(config) without "device"}; pickle-safe
def train_hetero_from_payload(payload: dict, device: str = "cpu", on_epoch=None) -> dict
    # HeteroTrainConfig(**payload["config"], device=device); HoldoutSplit.from_dict when split given;
    # returns {"model_state_dict": pickle.dumps(cpu state dict), "history", "metrics", "epochs_run", "device_used": device}
```

- Produces in `modal_app.py`: `_stage_source(src_code) -> str` (writes the shipped tree to a temp dir and puts it on `sys.path`; `_do_train` uses it too), `_do_train_hetero(payload_bytes: bytes, src_code: dict) -> dict` (cuda when available), `train_hetero_on_gpu` (T4, 3600), `train_hetero_on_a10g` (A10G, 3600), `train_hetero_on_a100` (A100, 7200), `HETERO_TRAIN_FUNCTIONS = {"T4": ..., "A10G": ..., "A100": ...}`; the image adds `"tqdm>=4.66"` and `"scipy>=1.11"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_payload_round_trip_reproduces_local_training():
    G = make_graph(); prepared = prepare_hetero_training(G)
    kwargs = hetero_model_kwargs(prepared.bundle, hidden=8); config = HeteroTrainConfig(epochs=3, early_stopping=False)
    payload = pickle.loads(pickle.dumps(hetero_training_payload(prepared, kwargs, config)))
    assert "device" not in payload["config"] and payload["split"]["hidden"] == prepared.split.hidden
    remote = train_hetero_from_payload(payload)
    local = train_hetero(prepared.bundle, prepared.index, prepared.split, kwargs, config)
    remote_state = pickle.loads(remote["model_state_dict"])
    assert all(torch.allclose(remote_state[k], v) for k, v in local.model.state_dict().items())
    assert remote["metrics"]["epochs_run"] == 3 and remote["device_used"] == "cpu"

def test_modal_hetero_entrypoint_runs_the_payload_on_cpu():
    from src.models.modal_app import HETERO_TRAIN_FUNCTIONS, _do_train_hetero, collect_source_code
    assert set(HETERO_TRAIN_FUNCTIONS) == {"T4", "A10G", "A100"}
    result = _do_train_hetero(pickle.dumps(payload), collect_source_code())
    assert set(result) >= {"model_state_dict", "history", "metrics", "epochs_run", "device_used"}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/bin/python -m pytest tests/models/test_hetero_pipeline.py -q -k "payload or modal"`
Expected: FAIL with ImportError on `hetero_training_payload` / `HETERO_TRAIN_FUNCTIONS`.

- [ ] **Step 3: Implement** the two pipeline functions and the Modal additions. `_do_train_hetero` stages the source, unpickles the payload, picks `"cuda" if torch.cuda.is_available() else "cpu"`, calls `train_hetero_from_payload`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/models/test_hetero_pipeline.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/hetero_pipeline.py src/models/modal_app.py tests/models/test_hetero_pipeline.py
git commit -m "Hetero training payload and Modal GPU entry points"
```

### Task 4: Training-page mode and docs

**Files:**
- Modify: `pages/3_Model_Training.py` (model radio at line 206, settings expanders, "Where to train", the Start Training handler)
- Modify: `docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md` (Section 6 bullet, new Section 9), `README.md` (training page docs, hetero section, test count)

**Interfaces:**
- Consumes: Task 1 and Task 3 functions; `HETERO_MODEL_TYPE`; `DECODERS`; `NEG_SAMPLING`; `HETERO_TRAIN_FUNCTIONS`, `collect_source_code`, `app` from `modal_app` (imported inside the Modal branch, as today).

- [ ] **Step 1: Model radio.** Fourth option `"Hetero GNN (gene–disease)"` (so `model_type_clean == HETERO_MODEL_TYPE`), help line: "**Hetero GNN** — Learns gene–disease links directly: genes and GWAS traits are nodes of one graph. Its scores appear on the Predictions page under *Ranking score*."

- [ ] **Step 2: Hetero settings.** In the hetero branch of the model settings expander: `hidden` select_slider `[32, 64, 128, 256]` default 128; `layers` slider 1–3 default 2; `decoder` selectbox `DECODERS`; `dropout` slider 0.0–0.5 step 0.05 default 0.2; `min_trait_genes` slider 1–10 default 1 ("Ignore traits with fewer known genes"); `neg_gene` slider 1–8 default 4 and `neg_trait` slider 0–4 default 1 ("negatives per known pair"); `neg_sampling` selectbox `NEG_SAMPLING`. Caption: "Learns gene–disease links directly; benchmarked against network enrichment when the hold-out is on." In the shared training-settings expander: the epochs slider becomes 10–2000 default 300 for hetero (10–500 default 100 otherwise); mini-batch controls are hidden for hetero; hetero adds `holdout` checkbox default True ("Hold out benchmark pairs (20 % of each disease's known genes) and benchmark against enrichment"), `val_fraction` slider 0.05–0.30 step 0.05 default 0.10, `eval_every` slider 1–20 default 1, `patience` slider 5–50 default 20 ("evaluations without improvement"), and the caption "The edge split under Network Summary applies to protein–protein models; this model holds out disease genes instead."

- [ ] **Step 3: Where to train.** Same radio for all models. For hetero on "This Mac's GPU" the caption reads "For this model the CPU was faster than the Mac GPU in our runs."

- [ ] **Step 4: Start Training handler.** Replace `if st.button("🚀 Start Training", ...):` with `start_training = st.button(...)`, then `if start_training and model_type_clean == HETERO_MODEL_TYPE: run_hetero_training(...)` and `if start_training and model_type_clean != HETERO_MODEL_TYPE:` guarding the existing block unchanged. `run_hetero_training` (a function defined above the columns in the page) does: `prepare_hetero_training` under a spinner, `st.error` + return on `ValueError`; shows "genes · traits · gene–trait associations (hidden pairs removed)"; builds `hetero_model_kwargs` and `HeteroTrainConfig(epochs, lr, neg_gene, neg_trait, neg_sampling, val_fraction, early_stopping, patience, eval_every, seed, device)`; local path: progress bar + status text + a line chart of loss and validation MRR redrawn every `max(1, epochs // 100)` epochs via `on_epoch`, then `progress_bar.progress(1.0)`; Modal path: `payload_bytes = pickle.dumps(hetero_training_payload(...))`, `with modal_app.run(): result = HETERO_TRAIN_FUNCTIONS[gpu_type].remote(payload_bytes, collect_source_code())`, rebuild the model from `model_kwargs` and `pickle.loads(result["model_state_dict"])`, `st.error` + return on exception; then `benchmark_hetero` under a spinner, `save_hetero_run(manager, ..., graph_name=_saved_graph_name, params=describe_hetero_params(..., gpu=gpu_type))`; shows metrics (validation MRR, recall@50, AUC, epochs run / best epoch), the benchmark table (rows = scorer names, columns recall@50, recall@200, median rank, MRR) when a hold-out was used, a caption comparing the model's recall@50 and MRR with enrichment's, "Model saved (run #id) — <path>", and `st.page_link("pages/4_Results.py", label="📊 Use it under Ranking score on the Predictions page")`. It never calls `session.set_training_results`.

- [ ] **Step 5: Syntax check**

Run: `venv/bin/python -c "import ast,sys; ast.parse(open('pages/3_Model_Training.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Docs.** Spec Section 6: replace "The Training page gets no new mode in this revision; training is the CLI." with "The Training page offers the model as *Hetero GNN (gene–disease)* (Section 9)." Add "Section 9: Training-page mode (2026-11-08)": controls, compute options (CPU, Mac GPU, Modal T4/A10G/A100 through `train_hetero_on_*`), what is shown, what is saved, the shared pipeline module, and that `training_results` is not set. README: a Hetero GNN bullet in the training-page model list, the hetero section says the page trains it too (any of the three compute options), the test count.

- [ ] **Step 7: Run the suite**

Run: `venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add pages/3_Model_Training.py docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md README.md docs/superpowers/plans/2026-11-08-hetero-training-page.md
git commit -m "Model Training page: Hetero GNN mode on CPU, Mac GPU or Modal"
```

- [ ] **Step 9: Verify in the running app** (port 8521): pick Hetero GNN, train a short run (CPU, 20 epochs, eval every 5) on a saved network, confirm the metrics, the benchmark table, the run row in "Past training runs" and the new option under Ranking score on the Predictions page; then delete the verification run and its checkpoint files. The Modal path is not exercised (it costs money); say so in the report.
