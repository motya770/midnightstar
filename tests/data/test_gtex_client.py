# tests/data/test_gtex_client.py
from unittest.mock import patch, MagicMock
from src.data.gtex_client import GTExClient
from src.data.models import ExpressionProfile

MOCK_GTEX_RESPONSE = {
    "medianGeneExpression": [
        {"gencodeId": "ENSG00000105866.10", "tissueSiteDetailId": "Brain_Cortex", "median": 15.3, "numSamples": 255},
        {"gencodeId": "ENSG00000105866.10", "tissueSiteDetailId": "Brain_Hippocampus", "median": 12.1, "numSamples": 197},
    ]
}

def test_gtex_get_expression():
    client = GTExClient()
    with patch("src.data.gtex_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_GTEX_RESPONSE
        mock_get.return_value = mock_resp
        profiles = client.get_expression("ENSG00000105866")
        assert len(profiles) == 2
        assert isinstance(profiles[0], ExpressionProfile)
        assert profiles[0].tissue == "Brain_Cortex"
        assert profiles[0].expression_level == 15.3

def test_gtex_handles_api_error():
    client = GTExClient()
    with patch("src.data.gtex_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.raise_for_status.side_effect = Exception("Service unavailable")
        mock_get.return_value = mock_resp
        profiles = client.get_expression("ENSG00000105866")
        assert profiles == []
