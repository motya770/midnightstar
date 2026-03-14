# src/data/gwas_client.py
import logging
import math
import requests
from src.data.models import Association, DiseaseNode

logger = logging.getLogger(__name__)
BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api"

class GWASClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url

    def search_gene(self, gene_symbol: str) -> tuple[list[Association], list[DiseaseNode]]:
        try:
            resp = requests.get(
                f"{self.base_url}/singleNucleotidePolymorphisms/search/findByDisOrGene",
                params={"gene": gene_symbol}, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.warning("GWAS API error for gene %s", gene_symbol)
            return [], []
        return self._parse_associations(data, gene_symbol)

    def search_disease(self, efo_trait_id: str) -> list[Association]:
        try:
            resp = requests.get(f"{self.base_url}/efoTraits/{efo_trait_id}/associations", timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.warning("GWAS API error for disease %s", efo_trait_id)
            return []
        embedded = data.get("_embedded", {})
        associations = []
        for item in embedded.get("associations", []):
            pvalue = item.get("pvalue", 1.0)
            score = self._pvalue_to_score(pvalue)
            associations.append(Association(
                source_id=efo_trait_id, target_id="", type="disease-gene",
                score=round(score, 4), evidence=f"p-value: {pvalue:.2e}", data_source="GWAS",
            ))
        return associations

    def _parse_associations(self, data: dict, gene_symbol: str) -> tuple[list[Association], list[DiseaseNode]]:
        embedded = data.get("_embedded", {})
        associations, diseases, seen_traits = [], [], set()
        for item in embedded.get("associations", []):
            pvalue = item.get("pvalue", 1.0)
            score = self._pvalue_to_score(pvalue)
            for trait_link in item.get("_links", {}).get("efoTraits", []):
                trait_href = trait_link.get("href", "")
                trait_id = trait_href.rstrip("/").split("/")[-1] if trait_href else ""
                associations.append(Association(
                    source_id=gene_symbol, target_id=trait_id, type="gene-disease",
                    score=round(score, 4), evidence=f"p-value: {pvalue:.2e}", data_source="GWAS",
                ))
                if trait_id and trait_id not in seen_traits:
                    seen_traits.add(trait_id)
                    disease = self._fetch_trait(trait_id)
                    if disease:
                        diseases.append(disease)
        return associations, diseases

    def _fetch_trait(self, trait_id: str) -> DiseaseNode | None:
        try:
            resp = requests.get(f"{self.base_url}/efoTraits/{trait_id}", timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return DiseaseNode(id=data.get("shortForm", trait_id), name=data.get("trait", "Unknown"),
                             description="", category="", source="GWAS")
        except Exception:
            return None

    @staticmethod
    def _pvalue_to_score(pvalue: float) -> float:
        if pvalue <= 0:
            return 1.0
        return min(1.0, -math.log10(pvalue) / 50)
