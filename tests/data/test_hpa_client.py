# tests/data/test_hpa_client.py
from unittest.mock import patch, MagicMock
from src.data.hpa_client import HPAClient

MOCK_HPA_GENE_RESPONSE = {
    "Gene": "SP4", "Gene synonym": "SPR-1", "Ensembl": "ENSG00000105866",
    "Gene description": "Sp4 transcription factor", "Subcellular location": "Nucleoplasm",
    "RNA tissue specificity": "Tissue enhanced (brain)", "Tissue expression cluster": "brain",
}

def test_hpa_get_gene_info():
    client = HPAClient()
    with patch("src.data.hpa_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [MOCK_HPA_GENE_RESPONSE]
        mock_get.return_value = mock_resp
        info = client.get_gene_info("ENSG00000105866")
        assert info is not None
        assert info["gene_symbol"] == "SP4"
        assert info["subcellular_location"] == "Nucleoplasm"
        assert "brain" in info["tissue_expression"].lower()

def test_hpa_handles_not_found():
    client = HPAClient()
    with patch("src.data.hpa_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp
        info = client.get_gene_info("ENSG_FAKE")
        assert info is None

def test_hpa_handles_api_error():
    client = HPAClient()
    with patch("src.data.hpa_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = Exception("Server error")
        mock_get.return_value = mock_resp
        info = client.get_gene_info("ENSG00000105866")
        assert info is None
