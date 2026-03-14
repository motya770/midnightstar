# tests/data/test_gwas_client.py
import json
from unittest.mock import patch, MagicMock
from src.data.gwas_client import GWASClient
from src.data.models import Association, DiseaseNode

MOCK_GWAS_RESPONSE = {
    "_embedded": {
        "associations": [
            {
                "riskFrequency": "0.35",
                "pvalue": 2.0e-12,
                "orPerCopyNum": 1.15,
                "loci": [{"strongestRiskAlleles": [{"riskAlleleName": "rs12345-A"}]}],
                "_links": {
                    "efoTraits": [{"href": "https://www.ebi.ac.uk/gwas/rest/api/efoTraits/EFO_0000249"}]
                },
            }
        ]
    }
}

MOCK_TRAIT_RESPONSE = {"trait": "Alzheimer's disease", "shortForm": "EFO_0000249"}

def test_gwas_search_gene():
    client = GWASClient()
    with patch("src.data.gwas_client.requests.get") as mock_get:
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = MOCK_GWAS_RESPONSE
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = MOCK_TRAIT_RESPONSE
        mock_get.side_effect = [resp1, resp2]
        associations, diseases = client.search_gene("SP4")
        assert len(associations) >= 1
        assert isinstance(associations[0], Association)
        assert associations[0].data_source == "GWAS"

def test_gwas_search_disease():
    client = GWASClient()
    with patch("src.data.gwas_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_GWAS_RESPONSE
        mock_get.return_value = mock_resp
        associations = client.search_disease("EFO_0000249")
        assert isinstance(associations, list)

def test_gwas_handles_api_error():
    client = GWASClient()
    with patch("src.data.gwas_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = Exception("Server error")
        mock_get.return_value = mock_resp
        associations, diseases = client.search_gene("SP4")
        assert associations == []
        assert diseases == []
