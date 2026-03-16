"""AlphaFold DB API client — fetch predicted protein structures and confidence scores."""
import logging
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://alphafold.ebi.ac.uk"
API_URL = f"{BASE_URL}/api"


class AlphaFoldClient:
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url

    def get_prediction(self, uniprot_id: str) -> dict | None:
        """Fetch prediction metadata for a UniProt accession."""
        try:
            resp = requests.get(f"{self.base_url}/prediction/{uniprot_id}", timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return None
            return data[0] if isinstance(data, list) else data
        except Exception as e:
            logger.warning("AlphaFold API error for %s: %s", uniprot_id, e)
            return None

    def get_plddt_scores(self, uniprot_id: str, version: int = 6) -> dict | None:
        """Fetch per-residue pLDDT confidence scores."""
        url = f"{BASE_URL}/files/AF-{uniprot_id}-F1-confidence_v{version}.json"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("AlphaFold pLDDT error for %s: %s", uniprot_id, e)
            return None

    def get_pdb_url(self, uniprot_id: str, version: int = 6) -> str:
        return f"{BASE_URL}/files/AF-{uniprot_id}-F1-model_v{version}.pdb"

    def get_cif_url(self, uniprot_id: str, version: int = 6) -> str:
        return f"{BASE_URL}/files/AF-{uniprot_id}-F1-model_v{version}.cif"

    def get_pae_url(self, uniprot_id: str, version: int = 6) -> str:
        return f"{BASE_URL}/files/AF-{uniprot_id}-F1-predicted_aligned_error_v{version}.json"

    def get_structure_pdb(self, uniprot_id: str, version: int = 6) -> str | None:
        """Download PDB file content as string."""
        url = self.get_pdb_url(uniprot_id, version)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.text
        except Exception:
            logger.warning("AlphaFold PDB download error for %s", uniprot_id)
            return None

    def get_structural_features(self, uniprot_id: str) -> dict | None:
        """Get summary structural features for use as ML node features."""
        pred = self.get_prediction(uniprot_id)
        if not pred:
            return None

        plddt_data = self.get_plddt_scores(uniprot_id)

        result = {
            "uniprot_id": uniprot_id,
            "gene": pred.get("gene", ""),
            "mean_plddt": pred.get("globalMetricValue", 0.0),
            "frac_very_high": pred.get("fractionPlddtVeryHigh", 0.0),
            "frac_confident": pred.get("fractionPlddtConfident", 0.0),
            "frac_low": pred.get("fractionPlddtLow", 0.0),
            "frac_very_low": pred.get("fractionPlddtVeryLow", 0.0),
            "sequence_length": pred.get("sequenceEnd", 0) - pred.get("sequenceStart", 0) + 1,
            "pdb_url": pred.get("pdbUrl", ""),
            "cif_url": pred.get("cifUrl", ""),
        }

        if plddt_data and "confidenceScore" in plddt_data:
            scores = plddt_data["confidenceScore"]
            result["plddt_scores"] = scores
            result["disordered_fraction"] = sum(1 for s in scores if s < 50) / max(len(scores), 1)
        else:
            result["plddt_scores"] = []
            result["disordered_fraction"] = 0.0

        return result

    def batch_predictions(self, uniprot_ids: list[str]) -> list[dict]:
        """Fetch predictions for multiple UniProt IDs via batch endpoint."""
        try:
            resp = requests.post(
                f"{self.base_url}/sequence/filtered-entries",
                json=uniprot_ids,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.warning("AlphaFold batch query failed")
            return []
