# tests/utils/test_coassociation.py
import math

import numpy as np
import pytest

from src.utils.coassociation import (
    DEFAULT_TOP_TRAITS,
    CoassociationScores,
    coassociation_scores,
    name_tokens,
    shares_name_token,
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
    # Height (similarity 0) and Asthma itself are excluded
    assert [t for t, _ in co.similar_traits[0]] == ["Allergic rhinitis", "Asthma severity", "Eczema"]
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
