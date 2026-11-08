# tests/utils/test_benchmark.py
import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.utils.benchmark import (
    BenchmarkResult,
    HoldoutSplit,
    centroid_cosine_matrix,
    make_holdout,
    mask_index,
    run_benchmark,
    summarize_ranks,
)
from src.utils.discovery import enrichment_matrix, index_disease_genes, predict_gene_disease_links

GENES = [f"G{i}" for i in range(12)]
ASTHMA = ["G0", "G1", "G2", "G3", "G4"]          # disease, 5 known -> 1 hidden
HEIGHT = [f"G{i}" for i in range(10)]           # trait, 10 known -> 2 hidden
FLU = ["G5", "G6", "G7"]                        # disease, too few known


def make_graph():
    G = nx.Graph()
    for g in GENES:
        G.add_node(g, node_type="gene", name=f"{g} protein")
    G.add_node("DIS:1", node_type="disease")
    for g in ASTHMA:
        G.nodes[g].setdefault("gwas", {}).setdefault("disease", []).append({"trait": "Asthma", "score": 0.9})
    for g in FLU:
        G.nodes[g].setdefault("gwas", {}).setdefault("disease", []).append({"trait": "Flu", "score": 0.5})
    for g in HEIGHT:
        G.nodes[g].setdefault("gwas", {}).setdefault("trait", []).append({"trait": "Height", "score": 0.4})
    # a path through the genes plus a hub (G11) touching every asthma gene
    G.add_edges_from([(GENES[i], GENES[i + 1]) for i in range(len(GENES) - 1)])
    G.add_edges_from([("G11", g) for g in ASTHMA])
    G.add_edge("G0", "DIS:1")
    return G


def setup(min_known_genes=5, categories=("disease",)):
    G = make_graph()
    node_list = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    index = index_disease_genes(G, node_list)
    split = make_holdout(index, categories=categories, min_known_genes=min_known_genes, seed=0)
    return G, node_list, node_to_idx, index, split


def expected_protocol_split(index, diseases, seed=0):
    """The hold-out protocol used in the spike scripts, written out independently."""
    rng = np.random.default_rng(seed)
    visible, hidden = {}, {}
    for d in diseases:
        genes = sorted(index[d]["genes"])
        perm = rng.permutation(len(genes))
        h = max(1, len(genes) // 5)
        hidden[d] = [genes[i] for i in perm[:h]]
        visible[d] = [genes[i] for i in perm[h:]]
    return visible, hidden


# ---- hold-out split ----------------------------------------------------------

def test_make_holdout_hides_a_fifth_of_each_eligible_disease():
    _, _, _, index, split = setup(min_known_genes=5, categories=("disease", "trait"))
    assert split.diseases == ["Height", "Asthma"]  # most known genes first; Flu has too few
    assert len(split.hidden["Asthma"]) == 1 and len(split.visible["Asthma"]) == 4
    assert len(split.hidden["Height"]) == 2 and len(split.visible["Height"]) == 8
    for d in split.diseases:
        assert not set(split.hidden[d]) & set(split.visible[d])
        assert set(split.hidden[d]) | set(split.visible[d]) == set(index[d]["genes"])


def test_make_holdout_follows_the_spike_protocol_exactly():
    _, _, _, index, split = setup(min_known_genes=5, categories=("disease", "trait"))
    visible, hidden = expected_protocol_split(index, ["Height", "Asthma"])
    assert split.hidden == hidden
    assert split.visible == visible


def test_make_holdout_is_deterministic_per_seed():
    _, _, _, index, _ = setup()
    a = make_holdout(index, categories=("trait",), min_known_genes=10, seed=0)
    b = make_holdout(index, categories=("trait",), min_known_genes=10, seed=0)
    c = make_holdout(index, categories=("trait",), min_known_genes=10, seed=1)
    assert a == b
    assert a.hidden != c.hidden


def test_holdout_split_round_trips_through_dict():
    _, _, _, _, split = setup()
    restored = HoldoutSplit.from_dict(split.to_dict())
    assert restored == split
    assert restored.to_dict()["seed"] == 0


def test_mask_index_removes_only_hidden_genes_of_split_diseases():
    _, _, _, index, split = setup()
    masked = mask_index(index, split)
    hidden = split.hidden["Asthma"][0]
    assert hidden not in masked["Asthma"]["genes"]
    assert set(masked["Asthma"]["genes"]) == set(split.visible["Asthma"])
    assert masked["Asthma"]["genes"][split.visible["Asthma"][0]] == 0.9  # scores preserved
    assert masked["Asthma"]["category"] == "disease"
    assert masked["Height"] == index["Height"]  # not part of the split: untouched
    assert masked["Flu"] == index["Flu"]
    assert hidden in index["Asthma"]["genes"]  # original not mutated


# ---- ranking and summary -------------------------------------------------------

def test_summarize_ranks_computes_recall_median_and_per_disease_mrr():
    ranks = {"A": {"g1": 1.0, "g2": 300.0}, "B": {"g3": 4.0}, "C": {"g4": 60.0, "g5": 150.0}}
    result = summarize_ranks(ranks)
    assert isinstance(result, BenchmarkResult)
    assert result.n_diseases == 3 and result.n_hidden == 5
    assert result.recall[50] == pytest.approx(2 / 5)
    assert result.recall[200] == pytest.approx(4 / 5)
    assert result.median_rank == 60.0
    # MRR: per disease, reciprocal rank of its best-ranked hidden gene
    assert result.mrr == pytest.approx((1 / 1 + 1 / 4 + 1 / 60) / 3)
    assert result.to_dict()["recall@50"] == pytest.approx(0.4)


def test_run_benchmark_with_score_matrix_ranks_hidden_genes_among_candidates_only():
    G, node_list, node_to_idx, index, split = setup()
    hidden = split.hidden["Asthma"][0]
    scores = np.zeros((len(node_list), 1))
    scores[node_to_idx["DIS:1"], 0] = 10.0          # non-gene node: never a candidate
    for g in split.visible["Asthma"]:
        scores[node_to_idx[g], 0] = 9.0              # visible known genes: never candidates
    scores[node_to_idx["G11"], 0] = 2.0              # one real candidate ahead of the hidden gene
    scores[node_to_idx[hidden], 0] = 1.0
    result = run_benchmark(scores, G, node_list, node_to_idx, index, split)
    assert result.hidden_ranks == {"Asthma": {hidden: 2.0}}
    assert result.mrr == pytest.approx(0.5)


def test_run_benchmark_with_score_matrix_gives_tied_candidates_their_average_rank():
    G, node_list, node_to_idx, index, split = setup()
    hidden = split.hidden["Asthma"][0]
    scores = np.zeros((len(node_list), 1))          # every candidate tied
    result = run_benchmark(scores, G, node_list, node_to_idx, index, split)
    n_candidates = len(GENES) - len(split.visible["Asthma"])  # 12 genes minus 4 visible = 8
    assert result.hidden_ranks["Asthma"][hidden] == pytest.approx((1 + n_candidates) / 2)


def test_run_benchmark_accepts_torch_tensors_and_checks_shape():
    G, node_list, node_to_idx, index, split = setup()
    result = run_benchmark(torch.zeros(len(node_list), 1), G, node_list, node_to_idx, index, split)
    assert result.n_hidden == 1
    with pytest.raises(ValueError):
        run_benchmark(np.zeros((len(node_list), 2)), G, node_list, node_to_idx, index, split)


def test_run_benchmark_with_scorer_function_passes_masked_index_and_reads_ranks():
    G, node_list, node_to_idx, index, split = setup()
    hidden = split.hidden["Asthma"][0]
    calls = []

    def scorer(z_norm, nodes, n2i, graph, idx, diseases, top_k):
        calls.append((idx, list(diseases), top_k))
        candidates = [n for n in nodes if graph.nodes[n]["node_type"] == "gene" and n not in idx["Asthma"]["genes"]]
        order = sorted(candidates, key=lambda n: (n != "G11", n != hidden))  # G11 first, hidden second
        return [{"disease": "Asthma", "gene": g, "rank": r} for r, g in enumerate(order[:top_k], start=1)]

    result = run_benchmark(scorer, G, node_list, node_to_idx, index, split, z_norm=torch.zeros(len(node_list), 2))
    (idx, diseases, top_k), = calls
    assert hidden not in idx["Asthma"]["genes"] and diseases == ["Asthma"] and top_k == len(node_list)
    assert result.hidden_ranks == {"Asthma": {hidden: 2}}


def test_run_benchmark_rejects_a_scorer_that_drops_a_hidden_gene():
    G, node_list, node_to_idx, index, split = setup()

    def scorer(z_norm, nodes, n2i, graph, idx, diseases, top_k):
        return [{"disease": "Asthma", "gene": "G11", "rank": 1}]

    with pytest.raises(ValueError, match="hidden"):
        run_benchmark(scorer, G, node_list, node_to_idx, index, split, z_norm=torch.zeros(len(node_list), 2))


def test_run_benchmark_works_end_to_end_with_production_scorer():
    G, node_list, node_to_idx, index, split = setup()
    z = F.normalize(torch.randn(len(node_list), 4, generator=torch.Generator().manual_seed(0)), dim=-1)
    result = run_benchmark(predict_gene_disease_links, G, node_list, node_to_idx, index, split, z_norm=z)
    hidden = split.hidden["Asthma"][0]
    # G11 touches every known asthma gene, so it is the top candidate and the hidden gene is behind it
    assert result.hidden_ranks["Asthma"][hidden] >= 2
    assert result.n_diseases == 1 and result.n_hidden == 1


# ---- reference scorers -------------------------------------------------------

def test_enrichment_matrix_matches_production_enrichment_scores():
    G, node_list, node_to_idx, index, split = setup()
    masked = mask_index(index, split)
    matrix = enrichment_matrix(G, node_list, node_to_idx, masked, ["Asthma"])
    assert matrix.shape == (len(node_list), 1)
    z = torch.zeros(len(node_list), 2)
    rows = predict_gene_disease_links(z, node_list, node_to_idx, G, masked, ["Asthma"], top_k=len(node_list))
    for row in rows:
        assert matrix[node_to_idx[row["gene"]], 0] == pytest.approx(row["enrichment_score"], abs=1e-4)
    assert matrix[node_to_idx["G11"], 0] > 0


def test_centroid_cosine_matrix_scores_against_mean_of_visible_genes():
    G, node_list, node_to_idx, index, split = setup()
    masked = mask_index(index, split)
    z = torch.randn(len(node_list), 3, generator=torch.Generator().manual_seed(1))
    matrix = centroid_cosine_matrix(z, node_list, node_to_idx, masked, ["Asthma"])
    z_norm = F.normalize(z, dim=-1)
    sig = F.normalize(z_norm[[node_to_idx[g] for g in split.visible["Asthma"]]].mean(0), dim=0)
    expected = (z_norm @ sig).numpy()
    assert matrix.shape == (len(node_list), 1)
    assert np.allclose(matrix[:, 0], expected, atol=1e-6)


def test_centroid_cosine_matrix_can_centre_embeddings_first():
    G, node_list, node_to_idx, index, split = setup()
    masked = mask_index(index, split)
    z = torch.randn(len(node_list), 3, generator=torch.Generator().manual_seed(2)) + 5.0  # strong common offset
    raw = centroid_cosine_matrix(z, node_list, node_to_idx, masked, ["Asthma"])
    centred = centroid_cosine_matrix(z, node_list, node_to_idx, masked, ["Asthma"], centre=True)
    zc = F.normalize(z - z.mean(0, keepdim=True), dim=-1)
    sig = F.normalize(zc[[node_to_idx[g] for g in split.visible["Asthma"]]].mean(0), dim=0)
    assert np.allclose(centred[:, 0], (zc @ sig).numpy(), atol=1e-6)
    assert raw.min() > 0.9  # the offset makes everything look alike
    assert centred.min() < 0.0


# ---- named reference scorers -------------------------------------------------

def test_reference_scorers_runs_named_baselines_and_rejects_unknown_names():
    from src.utils.benchmark import reference_scorers

    G, node_list, node_to_idx, index, split = setup()
    z = torch.randn(len(node_list), 4, generator=torch.Generator().manual_seed(3))
    results = reference_scorers(["enrichment", "cosine", "centred-cosine"], G, node_list, node_to_idx, index, split, z=z)
    assert list(results) == ["enrichment", "cosine", "centred-cosine"]
    assert all(isinstance(r, BenchmarkResult) for r in results.values())
    with pytest.raises(ValueError, match="unknown scorer"):
        reference_scorers(["nope"], G, node_list, node_to_idx, index, split, z=z)
    with pytest.raises(ValueError, match="embeddings"):
        reference_scorers(["cosine"], G, node_list, node_to_idx, index, split)


# ---- combination scorers and hub concentration ------------------------------

def test_rank_average_is_symmetric_and_orders_by_mean_rank():
    from src.utils.benchmark import rank_average

    a = np.array([[3.0], [2.0], [1.0], [0.0]])    # ranks 1,2,3,4
    b = np.array([[0.0], [5.0], [1.0], [9.0]])    # ranks 4,2,3,1
    combined = rank_average(a, b)
    assert combined.shape == (4, 1)
    assert np.allclose(combined, rank_average(b, a))
    # mean ranks: 2.5, 2.0, 3.0, 2.5 -> row 1 is best, row 2 is worst, rows 0 and 3 tie
    order = np.argsort(-combined[:, 0], kind="stable")
    assert order[0] == 1 and order[-1] == 2
    assert combined[0, 0] == combined[3, 0]


def test_zscore_sum_standardises_each_column():
    from src.utils.benchmark import zscore_sum

    a = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    b = np.array([[30.0, 1.0], [20.0, 1.0], [10.0, 1.0]])    # second column constant: contributes 0
    combined = zscore_sum(a, b)
    assert np.allclose(combined[:, 0], 0.0)                    # a and b are exact opposites in column 0
    expected = (a[:, 1] - a[:, 1].mean()) / a[:, 1].std()
    assert np.allclose(combined[:, 1], expected)


def test_fallback_uses_secondary_only_where_primary_is_zero_and_keeps_it_below():
    from src.utils.benchmark import fallback

    primary = np.array([[5.0], [0.0], [0.5], [0.0]])
    secondary = np.array([[100.0], [3.0], [100.0], [-2.0]])
    combined = fallback(primary, secondary)
    assert combined[0, 0] == 5.0 and combined[2, 0] == 0.5     # primary kept where non-zero
    assert combined[1, 0] > combined[3, 0]                      # secondary order preserved
    assert combined[1, 0] < 0.5                                 # but every fallback score ranks below any primary hit


def test_hub_concentration_counts_distinct_top_genes_and_repeat_winners():
    from src.utils.benchmark import hub_concentration

    G, node_list, node_to_idx, index, split = setup(min_known_genes=5, categories=("disease", "trait"))
    scores = np.zeros((len(node_list), len(split.diseases)))
    scores[node_to_idx["G11"], :] = 10.0   # G11 wins every disease
    stats = hub_concentration(scores, G, node_list, node_to_idx, split, k=3)
    assert stats["max_rank1"] == len(split.diseases) == 2
    assert stats["distinct_top10"] <= 2 * 3
    assert stats["distinct_top10"] >= 3


def test_run_benchmark_reports_hub_concentration_in_extra():
    G, node_list, node_to_idx, index, split = setup()
    result = run_benchmark(np.zeros((len(node_list), 1)), G, node_list, node_to_idx, index, split)
    assert set(result.extra) >= {"distinct_top10", "max_rank1"}
    assert "max rank-1" in result.summary()
    assert result.to_dict()["max_rank1"] == 1


def test_reference_scorers_accepts_hetero_matrix_names():
    from src.utils.benchmark import HETERO_SCORERS, reference_scorers

    G, node_list, node_to_idx, index, split = setup()
    hetero = np.random.default_rng(0).normal(size=(len(node_list), 1))
    results = reference_scorers(list(HETERO_SCORERS), G, node_list, node_to_idx, index, split, hetero_scores=hetero)
    assert list(results) == list(HETERO_SCORERS)
    assert all(isinstance(r, BenchmarkResult) for r in results.values())
    with pytest.raises(ValueError, match="hetero"):
        reference_scorers(["hetero"], G, node_list, node_to_idx, index, split)
