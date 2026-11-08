# Trait Co-association Scorer

**Date:** 2026-11-08
**Status:** Approved in chat (the user accepted the recommendation of the 2026-11-08 architecture investigation, "Yes, add it here"; the defaults below are the author's calls, marked as such; three builder-check questions were settled by the author and are marked in the text)
**Builds on:** `docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md` (benchmark, Ranking score option, export manifest, no-protein mode)

## Problem

Every disease-gene scorer MidnightStar ships ranks genes through the STRING protein network: hypergeometric enrichment of known disease genes among a gene's partners (the production default), and the heterogeneous gene–disease GNN alone or blended with enrichment. On the held-out benchmark (`src/utils/benchmark.py`, 219 diseases, 2,648 hidden genes, split of checkpoint `hetero_gnn_20261108_025450`) enrichment recovers 4.2 % of the hidden genes in the top 50 and no network-based variant tried on 2026-11-08 (random walk with restart, label propagation, three GNN decoders, four blends) gets past 5.7 %.

The 2026-11-08 investigation found a far stronger signal in the GWAS data itself: a gene's associations with *other* traits that share genes with the disease (pleiotropy). Ranking genes by their associations with the 20 traits most similar to the disease recovers 26 % of the hidden genes in the top 50, with no hub genes (no gene ranks first for more than 7 diseases), no training and under ten seconds of compute. Part of that gain comes from GWAS Catalog alias traits ("diabetes mellitus" next to "type 2 diabetes mellitus"); with every trait that shares a word with the disease name excluded it still recovers 19 %.

## Goal

Ship trait co-association as a first-class scorer:

1. a row in the benchmark CLI, computed exactly like production, so it can be compared with every other scorer on the same split;
2. a *Ranking score* option on the Predictions page (Predict Disease Genes tab and the export form), in both the protein-model and the no-protein modes, with the traits that put each candidate on the list shown next to it;
3. a strict variant that excludes traits sharing a word with the disease name, for users who want the pleiotropy signal without the alias traits;
4. the export bundle names the scorer and its parameters, carries the supporting traits per candidate and, in a third CSV, each exported disease's similar traits with their similarity, so Claude Science can inspect why a candidate ranks where it does (author's call after the builder check: the bundle holds no GWAS index, so without that file the ranking could not be examined outside the app).

Enrichment stays the default ranking (author's call: switching the default is a separate decision, best made after the user has looked at real candidate lists; the option's label states the benchmark gain so the choice is informed). **Amendment 2026-11-08:** after the scorer shipped the user made trait co-association the default ("Yes, make it default"); the page lists it first, enrichment remains the STRING-only alternative and the fallback when a gene–disease checkpoint cannot score the network, and the export manifest still records whichever scorer ranked.

## Non-goals

- No change to how enrichment, the hetero GNN or the blend score. No retraining.
- No combination of co-association with STRING signals: smoothing over partners, z-score sums and rank averages with enrichment were all tried and none beat co-association alone. Enrichment only breaks ties.
- No matrix-factorisation add-on in this round (it adds 5 % on top of co-association for six minutes of training; a possible follow-up once this ships).
- No gene → diseases direction, no new page.

## Anti-goals

- A ranking the user cannot explain: every candidate must show which traits put it there.
- Benchmark numbers that differ from what the page does: the CLI row must use the same weights, the same number of traits and the same enrichment tie-break as the page. (Author's call after the builder check: the embedding-similarity tie-break a loaded protein–protein model adds on the page is *not* applied in the benchmark rows, with or without `--checkpoint`; it only orders genes tied on both co-association and enrichment, which the benchmark averages. The README rows say they were measured without embeddings.)
- Hidden leakage in the benchmark: the scorer sees only the masked index.

## Constraints

- Pure NumPy/SciPy over the existing `index_disease_genes` index; no new dependencies.
- Scoring 363 diseases (the default export) must stay within a few seconds, excluding the enrichment tie-break the ranking already computes.
- Existing export columns keep their names and order; new columns are appended.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## The scorer

**Inputs.** The GWAS index `index: {trait: {"category", "genes": {gene: score}}}` from `index_disease_genes` (production: every association; benchmark: the masked index), `node_list` / `node_to_idx` (sorted graph nodes), and the diseases to score (each must be a key of the index; a disease that is not raises `ValueError`).

**Gene × trait matrix.** One column per trait of the index (sorted trait names), one row per node of `node_list`; entry = `log1p(score)` of the gene's association with the trait, 0 when none. Only genes present in `node_to_idx` contribute. (GWAS scores lie in [0.15, 1], so `log1p` is close to linear; it was the best of binary / raw / log1p on the benchmark, by 1–2 % recall@50 and 2–3 % recall@200.)

**Trait similarity.** Cosine similarity between the disease's column and every other trait's column of that matrix, with the disease's own column set to 0. The strict variant also sets to 0 every trait whose name shares a word with the disease name (see below).

**Score.** For disease *d*, keep its `top_traits = 20` most similar traits with similarity > 0 (fewer when fewer have positive similarity). The score of gene *g* is the sum over those traits *t* of `similarity(d, t) × log1p(score(g, t))`. Genes with no association to any of the 20 traits score 0; non-gene nodes score 0. The result is a `(len(node_list), len(diseases))` float matrix in `node_list` order, like every other scorer.

**Tie-break.** The ranking is `predict_gene_disease_links(..., scores=matrix)`, which orders by the matrix, then enrichment, then embedding similarity when a protein–protein model is loaded. So the many genes at 0 are ordered by enrichment, exactly as before this change.

**Name-sharing rule (strict variant).** Lower-case both names, split into alphanumeric words, drop words shorter than four characters and the stop words `and of the in to type mellitus disease disorder measurement level levels adjusted for or with response status count syndrome risk onset age at non`; two names share a word when the remaining sets intersect. Example: for "type 2 diabetes mellitus" this excludes "diabetes mellitus", "Drugs used in diabetes use measurement" and "diabetes mellitus, Drugs used in diabetes use measurement" but keeps "glucose measurement", "HbA1c measurement" and "diabetic retinopathy" (a different word). On the benchmark the rule removes 305 traits per disease on average. It is a word rule, not a synonym rule, and the documentation says so. A disease name with no qualifying word (every word shorter than four characters or a stop word) excludes nothing.

**Supporting traits.** For a (gene, disease) pair: the kept similar traits the gene is associated with, ordered by contribution `similarity × log1p(score)` descending, at most three in the UI column and the export column, formatted `trait (0.42); other trait (0.31)` where the number is the trait's similarity to the disease. The similar-trait list of a disease (all 20 with their similarity) is shown on the page and written to the export bundle as `similar_traits.csv` with the columns `disease, trait, similarity, rank` (rank 1 = most similar), one file per bundle, only when a co-association scorer produced it.

**Benchmark numbers** (same split as above, enrichment tie-break included, average rank for remaining ties):

| scorer | recall@50 | recall@200 | median rank | MRR | distinct top-10 | max rank-1 |
|---|---|---|---|---|---|---|
| enrichment | 0.0415 | 0.0827 | 10586 | 0.0854 | 1480 | 6 |
| trait co-association (log1p weights, 20 traits) | 0.2602 | 0.4033 | 681 | 0.4792 | 1234 | 7 |
| trait co-association, strict | 0.1930 | 0.2976 | 2411 | 0.3804 | 1186 | 9 |

(Numbers from `scripts/benchmark_disease_genes.py --scorers enrichment,coassociation,coassociation-strict` on the shipped implementation. The README's enrichment median rank of 5740 was measured with the similarity tie-break of a protein checkpoint; without embeddings ties get average ranks.)

## Where it lives

- **`src/utils/coassociation.py`** (new): the matrix, the similarity, the name rule, a result object that carries the score matrix plus the per-disease similar traits and can answer "which traits support gene g for disease d", and a helper that writes `supporting_traits` into ranking rows.
- **`src/utils/benchmark.py`**: an exact tie-break helper (primary order, ties by secondary, remaining ties to average rank; no epsilon arithmetic), and the scorer names `coassociation` and `coassociation-strict` in `reference_scorers`, computed from the masked index with enrichment as tie-break. The CLI runs them by default (they need no checkpoint).
- **`src/utils/discovery_export.py`**: scorer names `trait-coassociation` and `trait-coassociation-strict`; method texts with and without a protein–protein model; `supporting_traits` appended after `model_score` in `PREDICTED_COLUMNS` with its documentation; `model_score`'s documentation gains the co-association meaning; `build_manifest(..., scorer_parameters=None, similar_traits=None)` writes the parameters (`top_traits`, `weighting`, `exclude_name_sharing`) under `scorer.parameters` when given, leaves the scorer block unchanged otherwise, and describes `similar_traits.csv` in the manifest's `files` block (and the bundle README) only when the rows are given; `write_discovery_bundle(..., similar_traits=None)` writes that CSV when given.
- **`pages/4_Results.py`**: two always-available options after enrichment, *Trait co-association* and *Trait co-association, strict*, each labelled with its benchmark gain; `scorer_for` returns the co-association result alongside the matrix so the page can annotate rows; the candidates table shows a *Co-association score* column and a *Supporting traits* column, the caption explains the ranking, and an expander lists the disease's 20 most similar traits with their similarity; the export passes the scorer name, its parameters and the similar-trait rows to the manifest and the bundle. `SCORER_HELP` and the tab-1 intro mention the scorer.
- **`README.md`**: benchmark table rows, the Ranking score paragraph, the "Interpreting predictions" paragraph, the test count. The hetero spec's Section 6 gets one amendment line pointing here.

## What's left to the builder

- Exact function and class names, the dataclass layout, sparse vs dense intermediate shapes (the trait matrix must be sparse; the per-disease similarity may be dense).
- Test fixtures: a toy graph where one trait shares genes with the disease and another does not, so the expected scores can be written by hand; a name-rule case; a tie-break case; a `reference_scorers` smoke test without embeddings or hetero scores; export column and manifest tests.
- The wording of labels, captions and help texts, within the honesty rule above (state the benchmark gain and that part of it comes from alias traits).
