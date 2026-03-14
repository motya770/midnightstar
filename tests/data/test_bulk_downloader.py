# tests/data/test_bulk_downloader.py
from unittest.mock import MagicMock
from src.data.models import GeneNode, Association, ExpressionProfile, DiseaseNode
from src.data.bulk_downloader import BulkDownloader


def _make_gene(symbol: str) -> GeneNode:
    return GeneNode(symbol, f"ENSG_{symbol}", symbol, symbol, "", "Homo sapiens")


def test_bulk_download_calls_all_sources():
    string_client = MagicMock()
    gwas_client = MagicMock()
    gtex_client = MagicMock()

    string_client.get_interactions.return_value = [
        Association("SP4", "HSPD1", "protein-protein", 0.87, "exp", "STRING")
    ]
    gwas_client.search_gene.return_value = (
        [Association("SP4", "EFO_001", "gene-disease", 0.6, "p<5e-8", "GWAS")],
        [DiseaseNode("EFO_001", "Alzheimer's", "", "neuro", "GWAS")],
    )
    gtex_client.get_expression.return_value = [
        ExpressionProfile("ENSG_SP4", "Brain", 12.5, 100)
    ]

    downloader = BulkDownloader(
        string_client=string_client,
        gwas_client=gwas_client,
        gtex_client=gtex_client,
    )
    genes = [_make_gene("SP4")]
    result = downloader.download(genes)

    string_client.get_interactions.assert_called_once_with("SP4")
    gwas_client.search_gene.assert_called_once_with("SP4")
    gtex_client.get_expression.assert_called_once_with("ENSG_SP4")

    assert len(result["associations"]) == 2
    assert len(result["diseases"]) == 1
    assert len(result["expression"]) == 1


def test_bulk_download_handles_partial_failure():
    string_client = MagicMock()
    gwas_client = MagicMock()
    gtex_client = MagicMock()

    string_client.get_interactions.side_effect = RuntimeError("STRING down")
    gwas_client.search_gene.return_value = (
        [Association("SP4", "EFO_001", "gene-disease", 0.6, "p<5e-8", "GWAS")],
        [],
    )
    gtex_client.get_expression.return_value = []

    downloader = BulkDownloader(
        string_client=string_client,
        gwas_client=gwas_client,
        gtex_client=gtex_client,
    )
    genes = [_make_gene("SP4")]
    result = downloader.download(genes)

    # Despite STRING failure, we still get GWAS results
    assert len(result["associations"]) == 1
    assert result["errors"]["SP4"]["string"] is not None
