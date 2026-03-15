# tests/data/test_bulk_downloader.py
from unittest.mock import MagicMock
from src.data.bulk_downloader import BulkDownloader
from src.data.models import Association, ExpressionProfile, GeneNode


def test_bulk_download_calls_all_sources():
    mock_gwas = MagicMock()
    mock_gwas.search_gene.return_value = (
        [Association("SP4", "EFO_1", "gene-disease", 0.8, "ev", "GWAS")],
        [],
    )
    mock_gtex = MagicMock()
    mock_gtex.get_expression.return_value = [
        ExpressionProfile("ENSG00000105866", "Brain", 10.0, 100)
    ]
    mock_hpa = MagicMock()
    mock_hpa.get_gene_info.return_value = {"gene_symbol": "SP4"}
    mock_string = MagicMock()
    mock_string.get_interactions.return_value = [
        Association("SP4", "HSP60", "protein-protein", 0.9, "exp", "STRING")
    ]

    gene = GeneNode("SP4", "ENSG00000105866", "SP4", "Sp4", "", "Homo sapiens")
    downloader = BulkDownloader(
        gwas_client=mock_gwas,
        gtex_client=mock_gtex,
        hpa_client=mock_hpa,
        string_client=mock_string,
    )
    result = downloader.download_all(gene)

    assert len(result["associations"]) == 2
    assert len(result["expression"]) == 1
    assert result["hpa_info"] is not None
    mock_gwas.search_gene.assert_called_once()
    mock_gtex.get_expression.assert_called_once()
    mock_string.get_interactions.assert_called_once()


def test_bulk_download_handles_partial_failure():
    mock_gwas = MagicMock()
    mock_gwas.search_gene.side_effect = Exception("API down")
    mock_gtex = MagicMock()
    mock_gtex.get_expression.return_value = []
    mock_hpa = MagicMock()
    mock_hpa.get_gene_info.return_value = None
    mock_string = MagicMock()
    mock_string.get_interactions.return_value = [
        Association("SP4", "HSP60", "protein-protein", 0.9, "exp", "STRING")
    ]

    gene = GeneNode("SP4", "ENSG00000105866", "SP4", "Sp4", "", "Homo sapiens")
    downloader = BulkDownloader(
        gwas_client=mock_gwas,
        gtex_client=mock_gtex,
        hpa_client=mock_hpa,
        string_client=mock_string,
    )
    result = downloader.download_all(gene)

    assert len(result["associations"]) == 1  # Only STRING succeeded
    assert "GWAS" in result["errors"]
