# tests/data/test_string_client.py
import time
from unittest.mock import patch, MagicMock
from src.data.string_client import STRINGClient
from src.data.models import Association

MOCK_INTERACTIONS = [
    {"preferredName_A": "SP4", "preferredName_B": "HSP60", "stringId_A": "9606.ENSP00000105866",
     "stringId_B": "9606.ENSP00000216015", "score": 870, "nscore": 0, "fscore": 0, "pscore": 0,
     "ascore": 0, "escore": 450, "dscore": 0, "tscore": 700},
]

MOCK_IDENTIFIERS = [
    {"queryItem": "SP4", "preferredName": "SP4", "stringId": "9606.ENSP00000105866",
     "annotation": "Sp4 transcription factor"},
]

def test_string_get_interactions():
    client = STRINGClient()
    with patch("src.data.string_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_INTERACTIONS
        mock_get.return_value = mock_resp
        associations = client.get_interactions("SP4")
        assert len(associations) == 1
        assert isinstance(associations[0], Association)
        assert associations[0].score == 0.87
        assert associations[0].data_source == "STRING"

def test_string_resolve_identifier():
    client = STRINGClient()
    with patch("src.data.string_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_IDENTIFIERS
        mock_get.return_value = mock_resp
        result = client.resolve_identifier("SP4")
        assert result["preferredName"] == "SP4"

def test_string_handles_api_error():
    client = STRINGClient()
    with patch("src.data.string_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = Exception("Error")
        mock_get.return_value = mock_resp
        associations = client.get_interactions("SP4")
        assert associations == []

def test_string_rate_limiting():
    client = STRINGClient()
    with patch("src.data.string_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_INTERACTIONS
        mock_get.return_value = mock_resp
        start = time.time()
        client.get_interactions("SP4")
        client.get_interactions("TP53")
        elapsed = time.time() - start
        assert elapsed >= 1.0, "Should wait at least 1 second between STRING requests"
