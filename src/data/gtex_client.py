# src/data/gtex_client.py
import logging
import requests
from src.data.models import ExpressionProfile

logger = logging.getLogger(__name__)
BASE_URL = "https://gtexportal.org/api/v2"

class GTExClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url

    def get_expression(self, ensembl_id: str) -> list[ExpressionProfile]:
        gencode_id = ensembl_id.split(".")[0]
        try:
            resp = requests.get(
                f"{self.base_url}/expression/medianGeneExpression",
                params={"gencodeId": gencode_id, "datasetId": "gtex_v8"}, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.warning("GTEx API error for %s", ensembl_id)
            return []
        return [
            ExpressionProfile(gene_id=ensembl_id, tissue=item.get("tissueSiteDetailId", "Unknown"),
                            expression_level=item.get("median", 0.0), sample_count=item.get("numSamples", 0))
            for item in data.get("medianGeneExpression", [])
        ]
