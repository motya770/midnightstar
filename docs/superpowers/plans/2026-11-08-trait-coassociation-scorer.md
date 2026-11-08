# Trait Co-association Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rank disease genes by their GWAS associations with the traits most similar to the disease (trait co-association), as a benchmark row, a Predictions page *Ranking score* option with supporting traits per candidate, and a documented export scorer.

**Architecture:** A new pure module `src/utils/coassociation.py` builds the sparse gene × trait matrix (log1p GWAS scores) from the existing `index_disease_genes` index, takes the cosine similarity of each disease's column to every other trait, keeps the 20 most similar, and returns a result object with the (n_nodes × n_diseases) score matrix, the per-disease similar traits and a "which traits support this gene" query. The benchmark gains an exact tie-break helper and two scorer names; the export gains two scorer names, method texts, a `supporting_traits` column and optional scorer parameters; the page lists the two options after enrichment and annotates the candidates. Everything ranks through the existing `predict_gene_disease_links(..., scores=matrix)` path, so enrichment (then embedding similarity) breaks ties exactly as before.

**Tech Stack:** Python 3, NumPy, SciPy sparse, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-11-08-trait-coassociation-scorer-design.md`

## Global Constraints

- Enrichment stays the default ranking; the co-association options come right after it, always available (no checkpoint).
- Weights `log1p(score)`, `top_traits = 20`, similarity > 0 only, the disease's own column zeroed; strict variant drops traits that share a word with the disease name under the spec's rule (words of 4+ characters, stop list `and of the in to type mellitus disease disorder measurement level levels adjusted for or with response status count syndrome risk onset age at non`).
- The benchmark rows use the masked index, the same weights, the same `top_traits` and enrichment as tie-break (exact dense-rank composite, no epsilon).
- Existing export columns keep names and order; `supporting_traits` is appended after `model_score`; the manifest's `scorer` block is unchanged unless `scorer_parameters` is given.
- Supporting traits: at most 3, contribution order, formatted `trait (0.42); other (0.31)` (similarity to two decimals, `; ` separator). The bundle's third CSV `similar_traits.csv` (`disease, trait, similarity, rank`) exists only for co-association exports.
- The benchmark rows never apply the embedding-similarity tie-break, even with `--checkpoint` (spec anti-goals).
- Benchmark figures quoted in copy: co-association recall@50 0.260 (26 %) vs enrichment 0.042 (4 %), strict 0.193 (19 %); say that part of the gain comes from alias traits.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A disease whose known genes are all absent from the graph (zero-norm column): no NaN, a zero column and an empty similar-trait list. → Task 1 test `test_coassociation_disease_with_no_graph_genes_scores_zero_without_nan`.
2. A disease with fewer than 20 positively similar traits keeps only the positive ones, never zero-similarity padding. → Task 1 test (`similar_traits` has exactly the three positive traits).
3. A gene with no associations at all, or a non-gene node, asked for supporting traits: an empty list and an empty string, never an exception. → Task 2 test (gene F, node DIS:1).
4. Export without a protein–protein model and the co-association scorer: the method text says the similarity column is empty and the manifest records the parameters. → Task 4 test.
5. Exact ties in both the co-association score and enrichment on the benchmark get their average rank, and a higher secondary never beats a higher primary. → Task 3 test.

---

### Task 1: Co-association scores and the name rule

**Files:**
- Create: `src/utils/coassociation.py`
- Test: `tests/utils/test_coassociation.py` (new)
- Also commit the spec and this plan in this task's first commit.

**Interfaces:**
- Consumes: the index shape of `src.utils.discovery.index_disease_genes` (`{trait: {"category": str, "genes": {gene: score}}}`), `node_list`, `node_to_idx`.
- Produces:
  - `DEFAULT_TOP_TRAITS = 20`, `WEIGHTING = "log1p(GWAS score)"`, `NAME_STOP_WORDS: frozenset[str]`
  - `name_tokens(name: str) -> set[str]`, `shares_name_token(a: str, b: str) -> bool`
  - `@dataclass class CoassociationScores: matrix: np.ndarray; diseases: list[str]; similar_traits: list[list[tuple[str, float]]]; top_traits: int; exclude_name_sharing: bool; index: dict` (index kept for Task 2's queries; excluded from `repr`)
  - `coassociation_scores(index, node_list, node_to_idx, diseases, *, top_traits: int = DEFAULT_TOP_TRAITS, exclude_name_sharing: bool = False) -> CoassociationScores`; raises `ValueError` naming any disease that is not a key of `index`.

- [ ] **Step 0: Commit the spec and the plan**

```bash
git add docs/superpowers/specs/2026-11-08-trait-coassociation-scorer-design.md docs/superpowers/plans/2026-11-08-trait-coassociation-scorer.md
git commit -m "Spec and plan: trait co-association scorer"
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/utils/test_coassociation.py
import math
import numpy as np
import pytest

from src.utils.coassociation import (
    DEFAULT_TOP_TRAITS, CoassociationScores, coassociation_scores, name_tokens, shares_name_token,
)

NODES = sorted(["A", "B", "C", "D", "E", "F", "G", "H", "DIS:1"])
IDX = {n: i for i, n in enumerate(NODES)}


def make_index():
    return {
        "Asthma": {"category": "disease", "genes": {"A": 0.9, "B": 0.8, "C": 0.6}},
        "Allergic rhinitis": {"category": "disease", "genes": {"B": 0.5, "C": 0.5, "D": 0.5}},
        "Eczema": {"category": "disease", "genes": {"C": 0.4, "E": 0.4}},
        "Height": {"category": "trait", "genes": {"F": 0.7, "G": 0.7}},
        "Asthma severity": {"category": "trait", "genes": {"A": 0.3, "H": 0.3}},
    }


def expected(index, disease, top_traits=DEFAULT_TOP_TRAITS, exclude=()):
    """The spec's formula written out densely: cosine of log1p-weighted trait columns, top traits, weighted sum."""
    vec = {t: np.array([math.log1p(index[t]["genes"].get(n, 0.0)) for n in NODES]) for t in index}
    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    sims = sorted(((cos(vec[disease], vec[t]), t) for t in index if t != disease and t not in exclude), reverse=True)
    sims = [(s, t) for s, t in sims if s > 0][:top_traits]
    return sum((s * vec[t] for s, t in sims), np.zeros(len(NODES))), [(t, s) for s, t in sims]


def test_coassociation_scores_follow_the_spec_formula():
    co = coassociation_scores(make_index(), NODES, IDX, ["Asthma", "Eczema"])
    assert isinstance(co, CoassociationScores) and co.diseases == ["Asthma", "Eczema"]
    assert co.matrix.shape == (len(NODES), 2)
    scores, sims = expected(make_index(), "Asthma")
    np.testing.assert_allclose(co.matrix[:, 0], scores)
    assert [t for t, _ in co.similar_traits[0]] == ["Allergic rhinitis", "Asthma severity", "Eczema"]  # Height (0) and Asthma itself excluded
    assert co.similar_traits[0] == [(t, pytest.approx(s)) for t, s in sims]
    assert co.matrix[IDX["D"], 0] > co.matrix[IDX["H"], 0] > co.matrix[IDX["E"], 0] > 0
    assert co.matrix[IDX["F"], 0] == co.matrix[IDX["G"], 0] == co.matrix[IDX["DIS:1"], 0] == 0
    np.testing.assert_allclose(co.matrix[:, 1], expected(make_index(), "Eczema")[0])


def test_coassociation_top_traits_limits_the_similar_traits():
    co = coassociation_scores(make_index(), NODES, IDX, ["Asthma"], top_traits=1)
    assert [t for t, _ in co.similar_traits[0]] == ["Allergic rhinitis"] and co.top_traits == 1
    assert co.matrix[IDX["E"], 0] == co.matrix[IDX["H"], 0] == 0 and co.matrix[IDX["D"], 0] > 0


def test_coassociation_strict_drops_traits_sharing_a_word_with_the_disease():
    co = coassociation_scores(make_index(), NODES, IDX, ["Asthma"], exclude_name_sharing=True)
    assert [t for t, _ in co.similar_traits[0]] == ["Allergic rhinitis", "Eczema"] and co.exclude_name_sharing
    np.testing.assert_allclose(co.matrix[:, 0], expected(make_index(), "Asthma", exclude=("Asthma severity",))[0])
    assert co.matrix[IDX["H"], 0] == 0


def test_coassociation_disease_with_no_graph_genes_scores_zero_without_nan():
    index = {**make_index(), "Orphan": {"category": "disease", "genes": {"ZZ1": 0.9, "ZZ2": 0.9}}}
    co = coassociation_scores(index, NODES, IDX, ["Orphan", "Asthma"])
    assert np.all(co.matrix[:, 0] == 0) and co.similar_traits[0] == []
    assert np.isfinite(co.matrix).all() and co.matrix[IDX["D"], 1] > 0


def test_coassociation_rejects_a_disease_missing_from_the_index():
    with pytest.raises(ValueError, match="Nonexistent"):
        coassociation_scores(make_index(), NODES, IDX, ["Asthma", "Nonexistent"])


def test_name_tokens_and_shares_name_token_follow_the_spec_rule():
    assert name_tokens("type 2 diabetes mellitus") == {"diabetes"}
    assert name_tokens("Crohn's disease") == {"crohn"}
    assert shares_name_token("type 2 diabetes mellitus", "diabetes mellitus")
    assert shares_name_token("type 2 diabetes mellitus", "Drugs used in diabetes use measurement")
    assert not shares_name_token("type 2 diabetes mellitus", "diabetic retinopathy")
    assert not shares_name_token("type 2 diabetes mellitus", "glucose measurement")
    assert shares_name_token("asthma", "Asthma severity")
    assert not shares_name_token("coronary artery disease", "Crohn's disease")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/utils/test_coassociation.py -v`
Expected: FAIL at import (`ModuleNotFoundError: src.utils.coassociation`)

- [ ] **Step 3: Implement `src/utils/coassociation.py`**

Module docstring states the method and the benchmark numbers in one paragraph. Build one `scipy.sparse.csr_matrix` of shape `(len(node_list), n_traits)` over `sorted(index)` with `log1p(max(score, 0))` entries for genes in `node_to_idx`; normalise columns (zero norms become 1); `similarity = (Mn.T @ Mn[:, disease_cols]).toarray()` with the diseases' own entries set to 0 and, when `exclude_name_sharing`, every trait sharing a token with the disease set to 0 (tokenise each trait name once); per disease keep the `top_traits` largest positive similarities as a dense `(n_traits, n_diseases)` weight matrix; `matrix = np.asarray(M @ weights)`. `name_tokens` = lower-case, `re.findall(r"[a-z0-9]+")`, keep length ≥ 4 and not in `NAME_STOP_WORDS`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/utils/test_coassociation.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/utils/coassociation.py tests/utils/test_coassociation.py
git commit -m "Trait co-association scores over the GWAS index"
```

---

### Task 2: Supporting traits and row annotation

**Files:**
- Modify: `src/utils/coassociation.py`
- Test: `tests/utils/test_coassociation.py`

**Interfaces:**
- Consumes: `CoassociationScores` from Task 1.
- Produces:
  - `SUPPORTING_TRAITS_LIMIT = 3`
  - `CoassociationScores.supporting_traits(gene: str, disease: str, limit: int = SUPPORTING_TRAITS_LIMIT) -> list[tuple[str, float]]` — the disease's kept similar traits the gene is associated with, ordered by contribution `similarity × log1p(score)` descending, each as `(trait, similarity)`; `[]` for a gene with none (or not in the index).
  - `CoassociationScores.parameters -> dict` = `{"top_traits": ..., "weighting": WEIGHTING, "exclude_name_sharing": ...}` (for the manifest).
  - `format_supporting_traits(pairs: list[tuple[str, float]]) -> str` → `"Allergic rhinitis (0.62); Eczema (0.34)"`, `""` for an empty list.
  - `annotate_supporting_traits(rows: list[dict], scores: CoassociationScores, limit: int = SUPPORTING_TRAITS_LIMIT) -> list[dict]` — sets `row["supporting_traits"]` on every row from `row["disease"]` / `row["gene"]` (`""` when the row's disease was not scored) and returns the same list.
  - `CoassociationScores.similar_trait_rows() -> list[dict]` — one row per (disease, kept trait): `{"disease", "trait", "similarity" (rounded to 4 decimals), "rank"}` with rank 1 = most similar, diseases in `self.diseases` order.

- [ ] **Step 1: Write the failing tests** (append to the test file)

```python
def test_supporting_traits_lists_the_similar_traits_the_gene_has_in_contribution_order():
    co = coassociation_scores(make_index(), NODES, IDX, ["Asthma"])
    assert [t for t, _ in co.supporting_traits("C", "Asthma")] == ["Allergic rhinitis", "Eczema"]
    assert co.supporting_traits("D", "Asthma") == [("Allergic rhinitis", pytest.approx(0.617, abs=1e-3))]
    assert [t for t, _ in co.supporting_traits("C", "Asthma", limit=1)] == ["Allergic rhinitis"]
    assert co.supporting_traits("F", "Asthma") == [] and co.supporting_traits("DIS:1", "Asthma") == []


def test_annotate_supporting_traits_formats_up_to_three_traits_per_row():
    from src.utils.coassociation import SUPPORTING_TRAITS_LIMIT, annotate_supporting_traits, format_supporting_traits

    co = coassociation_scores(make_index(), NODES, IDX, ["Asthma"])
    rows = [{"disease": "Asthma", "gene": "C"}, {"disease": "Asthma", "gene": "F"}, {"disease": "Other", "gene": "C"}]
    assert annotate_supporting_traits(rows, co) is rows
    assert rows[0]["supporting_traits"] == "Allergic rhinitis (0.62); Eczema (0.34)"
    assert rows[1]["supporting_traits"] == "" and rows[2]["supporting_traits"] == ""
    assert SUPPORTING_TRAITS_LIMIT == 3 and format_supporting_traits([]) == ""
    assert co.parameters == {"top_traits": 20, "weighting": "log1p(GWAS score)", "exclude_name_sharing": False}
    assert co.similar_trait_rows()[:2] == [
        {"disease": "Asthma", "trait": "Allergic rhinitis", "similarity": pytest.approx(0.6174, abs=1e-4), "rank": 1},
        {"disease": "Asthma", "trait": "Asthma severity", "similarity": pytest.approx(0.4588, abs=1e-4), "rank": 2},
    ] and len(co.similar_trait_rows()) == 3
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/utils/test_coassociation.py -k "supporting or annotate" -v`
Expected: FAIL (`AttributeError: 'CoassociationScores' object has no attribute 'supporting_traits'`, then ImportError for the helpers)

- [ ] **Step 3: Implement the method, property and helpers** in `src/utils/coassociation.py` (the method reads `self.index[trait]["genes"].get(gene)` for each kept trait of the disease; a disease → column map built once in `__post_init__`).

- [ ] **Step 4: Run the file to verify it passes**

Run: `pytest tests/utils/test_coassociation.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/utils/coassociation.py tests/utils/test_coassociation.py
git commit -m "Supporting traits per co-association candidate"
```

---

### Task 3: Benchmark rows

**Files:**
- Modify: `src/utils/benchmark.py` (combining helpers section; `REFERENCE_SCORERS`; `reference_scorers`; module docstring: the co-association rows take the matrix path even when embeddings are given)
- Modify: `scripts/benchmark_disease_genes.py` (docstring example; default scorer list)
- Test: `tests/utils/test_benchmark.py`

**Interfaces:**
- Consumes: `coassociation_scores(masked, node_list, node_to_idx, split.diseases, exclude_name_sharing=...)` from Task 1 (`.matrix`).
- Produces:
  - `tie_break(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray` — per column `dense_rank(primary) * (n + 1) + dense_rank(secondary)` (ranks ascending, so higher is better), float matrix.
  - `COASSOCIATION_SCORERS = ("coassociation", "coassociation-strict")`; `REFERENCE_SCORERS = ("enrichment",) + COASSOCIATION_SCORERS + EMBEDDING_SCORERS + HETERO_SCORERS`; `reference_scorers` runs them as `run_benchmark(tie_break(co.matrix, enrichment), ...)` with `enrichment` from the masked index (no `z`, no hetero scores needed).
  - CLI default `names = ["enrichment", "coassociation", "coassociation-strict"]` (+ the embedding / hetero scorers when their inputs are given).

- [ ] **Step 1: Write the failing tests** (append to `tests/utils/test_benchmark.py`)

```python
def test_tie_break_orders_by_primary_then_secondary_and_keeps_remaining_ties():
    from scipy.stats import rankdata
    from src.utils.benchmark import tie_break

    primary = np.array([[1.0], [1.0], [0.0], [1.0], [0.0]])
    secondary = np.array([[0.2], [0.9], [5.0], [0.9], [1.0]])
    ranks = rankdata(-tie_break(primary, secondary)[:, 0], method="average")
    assert ranks.tolist() == [3.0, 1.5, 4.0, 1.5, 5.0]     # a higher secondary never beats a higher primary


def test_reference_scorers_runs_coassociation_rows_without_embeddings_or_hetero_scores():
    from src.utils.benchmark import COASSOCIATION_SCORERS, REFERENCE_SCORERS, reference_scorers

    assert COASSOCIATION_SCORERS == ("coassociation", "coassociation-strict")
    assert all(name in REFERENCE_SCORERS for name in COASSOCIATION_SCORERS)
    G, node_list, node_to_idx, index, split = setup()
    results = reference_scorers(list(COASSOCIATION_SCORERS), G, node_list, node_to_idx, index, split)
    assert list(results) == list(COASSOCIATION_SCORERS)
    hidden = split.hidden["Asthma"][0]          # an asthma gene; every asthma gene also has Height
    assert results["coassociation"].hidden_ranks["Asthma"][hidden] <= 6   # among the six Height genes, above G10/G11
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/utils/test_benchmark.py -k "tie_break or coassociation" -v`
Expected: FAIL with ImportError (`tie_break`, `COASSOCIATION_SCORERS`)

- [ ] **Step 3: Implement** `tie_break` next to `fallback`, the names, the branch in `reference_scorers` (compute `enrichment` whenever a co-association name or a hetero combination is requested; import `coassociation_scores` at module top), and the CLI changes (docstring example `--scorers enrichment,coassociation,coassociation-strict`; default list).

- [ ] **Step 4: Run the file to verify it passes**

Run: `pytest tests/utils/test_benchmark.py -v`
Expected: all passed (26)

- [ ] **Step 5: Commit**

```bash
git add src/utils/benchmark.py scripts/benchmark_disease_genes.py tests/utils/test_benchmark.py
git commit -m "Benchmark rows for trait co-association"
```

---

### Task 4: Export scorer names, column and manifest parameters

**Files:**
- Modify: `src/utils/discovery_export.py`
- Test: `tests/utils/test_discovery_export.py`

**Interfaces:**
- Produces:
  - `COASSOCIATION_SCORER = "trait-coassociation"`, `COASSOCIATION_STRICT_SCORER = "trait-coassociation-strict"`
  - `PREDICTED_COLUMNS = (... , "model_score", "supporting_traits")`; `PREDICTED_COLUMN_DOCS["supporting_traits"]` and the co-association meaning added to `PREDICTED_COLUMN_DOCS["model_score"]`.
  - `COASSOCIATION_METHOD_TEXT`, `COASSOCIATION_METHOD_TEXT_NO_PROTEIN`, `COASSOCIATION_STRICT_METHOD_TEXT`, `COASSOCIATION_STRICT_METHOD_TEXT_NO_PROTEIN` (shared head: cosine similarity of log1p-weighted gene sets, 20 most similar traits, score = Σ similarity × log1p(GWAS score); `model_score` is that sum and `supporting_traits` lists the traits; ties broken by `enrichment_score` "and then by similarity" / "; no protein–protein model was loaded for this export, so the similarity column is empty"; strict adds "Traits whose name shares a word with the disease name (alias and multi-trait study names) were excluded from the similar traits."; shared tail with the benchmark figures of the Global Constraints and "Candidates are hypotheses for follow-up.").
  - `build_manifest(..., scorer_parameters: dict | None = None, similar_traits: list[dict] | None = None)`: adds `"parameters": scorer_parameters` to the `scorer` block only when given; when `similar_traits` is given, adds `files[SIMILAR_TRAITS_FILE]` (`SIMILAR_TRAITS_FILE = "similar_traits.csv"`, description "The GWAS traits most similar to each exported disease, which the trait co-association scorer summed over", columns `SIMILAR_TRAITS_COLUMN_DOCS` for `disease, trait, similarity, rank`) and `counts["similar_traits"]`; `render_readme` appends ` (top_traits=20, weighting=log1p(GWAS score), exclude_name_sharing=False)` to the *Ranking scorer* line when parameters exist and lists the third file under "What is here" and "Columns" when the manifest describes it.
  - `write_discovery_bundle(out_dir, links, known, manifest, similar_traits: list[dict] | None = None)` writes `similar_traits.csv` (columns `SIMILAR_TRAITS_COLUMNS = ("disease", "trait", "similarity", "rank")`) only when rows are given.

- [ ] **Step 1: Amend the two column tests and write the failing test**

Change `test_predicted_columns_end_with_model_score_and_keep_existing_order` to `test_predicted_columns_keep_existing_order_and_append_model_score_then_supporting_traits` asserting `PREDICTED_COLUMNS[:-2]` is the original eleven, `PREDICTED_COLUMNS[-2:] == ("model_score", "supporting_traits")`, and both documented. In `test_bundle_csv_writes_model_score_column` replace `df.columns[-1] == "model_score"` with `df.columns[-2] == "model_score"`. Append:

```python
def test_manifest_records_coassociation_scorer_parameters_and_method_text():
    from src.utils.discovery_export import COASSOCIATION_SCORER, COASSOCIATION_STRICT_SCORER, render_readme

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
```

- [ ] **Step 2: Run to verify the new and amended tests fail**

Run: `pytest tests/utils/test_discovery_export.py -v`
Expected: 4 failed (the two amended column tests, the two new tests), the rest pass

- [ ] **Step 3: Implement** the constants, column, docs, method texts, `scorer_parameters`, `similar_traits` (manifest, README rendering, bundle file) in `src/utils/discovery_export.py`.

- [ ] **Step 4: Run the file to verify it passes**

Run: `pytest tests/utils/test_discovery_export.py -v`
Expected: all passed (16)

- [ ] **Step 5: Commit**

```bash
git add src/utils/discovery_export.py tests/utils/test_discovery_export.py
git commit -m "Export manifest and supporting_traits column for the co-association scorers"
```

---

### Task 5: Ranking score options on the Predictions page

**Files:**
- Modify: `pages/4_Results.py` (imports; scorer section; `scorer_for`; `pick_scorer`; tab-1 intro, results and caption; export)

**Interfaces:**
- Consumes: `coassociation_scores`, `annotate_supporting_traits`, `CoassociationScores.parameters` / `.similar_traits` (Tasks 1–2); `COASSOCIATION_SCORER`, `COASSOCIATION_STRICT_SCORER` (Task 4).
- Produces: `scorer_for(label, diseases) -> (matrix | None, scorer_name, checkpoint_path | None, coassociation: CoassociationScores | None)`; both callers updated.

- [ ] **Step 1: Options and scorer** — labels `SCORER_COASSOCIATION_LABEL = "Trait co-association (genes of the GWAS traits most similar to the disease; ~6× enrichment on the benchmark)"` and `SCORER_COASSOCIATION_STRICT_LABEL = "Trait co-association, strict (traits sharing a word with the disease name excluded; ~4.5× enrichment)"`, registered in `_scorer_kinds` as `(COASSOCIATION_SCORER, None)` / `(COASSOCIATION_STRICT_SCORER, None)` before the blend entries, so `scorer_options` = enrichment, co-association, strict, blends, models. `scorer_for` returns the four-tuple; for the co-association kinds it calls `coassociation_scores(disease_index, node_list, node_to_idx, diseases, exclude_name_sharing=<strict>)` and returns `(co.matrix, name, None, co)`; every other branch returns `None` as the fourth element. `pick_scorer` always shows the selectbox (there are always at least three options). `SCORER_HELP` gains the co-association sentences (26 % vs 4 % recall@50, no hub genes, alias traits, strict 19 %, enrichment breaks ties). The tab-1 intro mentions co-association.

- [ ] **Step 2: Tab 1** — after `predict_gene_disease_links`, `annotate_supporting_traits(candidates, co)` when `co` is not None; `score_column` maps the two co-association names to "Co-association score"; the table adds a "Supporting traits" column right after the score column for those scorers; caption "Ranked by the gene's GWAS associations with the {n} traits most similar to this disease, weighted by similarity and association strength; enrichment breaks ties" (n = `len(co.similar_traits[0])`); an `st.expander("Traits most similar to <disease>")` with a two-column DataFrame (Trait, Similarity) of `co.similar_traits[0]`.

- [ ] **Step 3: Export** — unpack the fourth element, annotate `links` when present, pass `scorer_parameters=exp_co.parameters if exp_co else None` and `similar_traits=exp_co.similar_trait_rows() if exp_co else None` to `build_manifest` and the same rows to `write_discovery_bundle`.

- [ ] **Step 4: Verify in the running app** (`preview_start` config `midnightstar-hetero`, restart it first since modules changed): in no-protein mode rank *type 2 diabetes mellitus* with *Trait co-association* (expect HMGA2, THADA, SLC30A8 among the top; the Supporting traits column filled; the expander lists "diabetes mellitus" first) and with the strict variant (the expander no longer lists "diabetes mellitus"); export a bundle with the co-association scorer to the scratchpad and check `manifest.json` has `scorer.name = "trait-coassociation"` with `parameters`, `predicted_gene_disease_links.csv` ends with `supporting_traits`, and `similar_traits.csv` exists with 20 rows per disease.

- [ ] **Step 5: Commit**

```bash
git add pages/4_Results.py
git commit -m "Predictions page: trait co-association ranking with supporting traits"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md` (benchmark table and the paragraph after it; "Interpreting predictions"; the Ranking score paragraph; the test count), `docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md` (Section 6: one amendment line pointing to the new spec).

- [ ] **Step 1: Run the CLI for the official rows**

Run: `python scripts/benchmark_disease_genes.py --graph full_s800_gwas --scorers enrichment,coassociation,coassociation-strict --json <scratchpad>/bench_coassociation.json`
Expected: `coassociation` recall@50 0.2602, `coassociation-strict` 0.1930 (the spec's numbers, same split); if they differ, the implementation deviates from the spike and the ledger gets a ruling before any copy is written.

- [ ] **Step 2: README** — two table rows (`Trait co-association (`coassociation`)` and the strict row) with the CLI's numbers; a paragraph after the table on what the rows mean and the alias-trait caveat; the "Interpreting predictions" paragraph gains one sentence on the option; the Ranking score paragraph lists *Trait co-association / Trait co-association, strict* as always-available options, names the new manifest scorers and the `supporting_traits` column; test count from `pytest -q`.

- [ ] **Step 3: Hetero spec Section 6** — append "Amendment 2026-11-08 (b): trait co-association (`docs/superpowers/specs/2026-11-08-trait-coassociation-scorer-design.md`) joins the Ranking score options and the benchmark; enrichment stays the default."

- [ ] **Step 4: Full suite**

Run: `pytest -q`
Expected: all passed, no new warnings

- [ ] **Step 5: Commit**

```bash
git add README.md docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md
git commit -m "Docs: trait co-association scorer"
```
