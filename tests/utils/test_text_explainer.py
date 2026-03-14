# tests/utils/test_text_explainer.py
from src.data.models import GeneNode, Association
from src.utils.text_explainer import TextExplainer


def _gene_full():
    return GeneNode("SP4", "ENSG00000105866", "SP4", "Sp4 transcription factor", "TF", "Homo sapiens")


def _gene_minimal():
    return GeneNode("SP4", "ENSG00000105866", "SP4", "", "", "Homo sapiens")


def _strong_assoc():
    return Association("SP4", "HSPD1", "protein-protein", 0.87, "experiments", "STRING")


def _weak_assoc():
    return Association("SP4", "EFO_0000249", "gene-disease", 0.42, "p<0.05", "GWAS")


def test_explain_gene():
    explainer = TextExplainer()
    text = explainer.explain_gene(_gene_full())
    assert "SP4" in text
    assert "Sp4 transcription factor" in text
    assert "TF" in text


def test_explain_gene_minimal_info():
    explainer = TextExplainer()
    text = explainer.explain_gene(_gene_minimal())
    assert "SP4" in text
    assert isinstance(text, str)
    assert len(text) > 0


def test_explain_association():
    explainer = TextExplainer()
    text = explainer.explain_association(_strong_assoc())
    assert "SP4" in text
    assert "HSPD1" in text
    assert "strong" in text.lower()
    assert "STRING" in text


def test_explain_weak_association():
    explainer = TextExplainer()
    text = explainer.explain_association(_weak_assoc())
    assert "SP4" in text
    assert "EFO_0000249" in text
    assert "weak" in text.lower() or "moderate" in text.lower()


def test_explain_prediction():
    explainer = TextExplainer()
    text = explainer.explain_prediction("SP4", "EFO_0000249", 0.91, ["HSPD1", "TP53"])
    assert "SP4" in text
    assert "EFO_0000249" in text
    assert "HSPD1" in text
    assert isinstance(text, str)
