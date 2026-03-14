from unittest.mock import patch, MagicMock
from src.data.gene_resolver import GeneResolver


def test_resolve_symbol_to_ensembl():
    mock_mg = MagicMock()
    mock_mg.query.return_value = {
        "hits": [{"ensembl": {"gene": "ENSG00000105866"}, "symbol": "SP4", "name": "Sp4 Transcription Factor"}]
    }
    resolver = GeneResolver(client=mock_mg)
    result = resolver.resolve("SP4")
    assert result.ensembl_id == "ENSG00000105866"
    assert result.symbol == "SP4"


def test_resolve_returns_none_for_unknown():
    mock_mg = MagicMock()
    mock_mg.query.return_value = {"hits": []}
    resolver = GeneResolver(client=mock_mg)
    result = resolver.resolve("FAKEGENE123")
    assert result is None


def test_resolve_batch():
    mock_mg = MagicMock()
    mock_mg.querymany.return_value = [
        {"query": "SP4", "ensembl": {"gene": "ENSG00000105866"}, "symbol": "SP4", "name": "Sp4 TF"},
        {"query": "TP53", "ensembl": {"gene": "ENSG00000141510"}, "symbol": "TP53", "name": "Tumor Protein P53"},
    ]
    resolver = GeneResolver(client=mock_mg)
    results = resolver.resolve_batch(["SP4", "TP53"])
    assert len(results) == 2
    assert results["SP4"].ensembl_id == "ENSG00000105866"
    assert results["TP53"].ensembl_id == "ENSG00000141510"
