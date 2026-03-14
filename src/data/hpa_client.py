# src/data/hpa_client.py
import logging
import requests

logger = logging.getLogger(__name__)
GENE_URL = "https://www.proteinatlas.org/{ensembl_id}.json"

class HPAClient:
    def get_gene_info(self, ensembl_id: str) -> dict | None:
        try:
            resp = requests.get(GENE_URL.format(ensembl_id=ensembl_id), timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.warning("HPA API error for %s", ensembl_id)
            return None
        if not data:
            return None
        entry = data[0] if isinstance(data, list) else data
        return {
            "gene_symbol": entry.get("Gene", ""),
            "gene_description": entry.get("Gene description", ""),
            "ensembl_id": entry.get("Ensembl", ensembl_id),
            "subcellular_location": entry.get("Subcellular location", ""),
            "tissue_expression": entry.get("RNA tissue specificity", ""),
            "tissue_cluster": entry.get("Tissue expression cluster", ""),
            "synonyms": entry.get("Gene synonym", ""),
        }
