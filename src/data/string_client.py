# src/data/string_client.py
import logging
import time
import threading
import requests
from src.data.models import Association

logger = logging.getLogger(__name__)
BASE_URL = "https://string-db.org/api"
SPECIES_ID = "9606"

class STRINGClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def _rate_limit(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
            self._last_request_time = time.time()

    def get_interactions(self, gene_symbol: str, required_score: int = 400) -> list[Association]:
        self._rate_limit()
        try:
            resp = requests.get(
                f"{self.base_url}/json/network",
                params={"identifiers": gene_symbol, "species": SPECIES_ID,
                        "required_score": required_score, "caller_identity": "midnightstar"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.warning("STRING API error for %s", gene_symbol)
            return []
        return [
            Association(
                source_id=item.get("preferredName_A", ""), target_id=item.get("preferredName_B", ""),
                type="protein-protein", score=round(item.get("score", 0) / 1000.0, 4),
                evidence=self._build_evidence(item), data_source="STRING",
            ) for item in data
        ]

    def resolve_identifier(self, query: str) -> dict | None:
        self._rate_limit()
        try:
            resp = requests.get(
                f"{self.base_url}/json/get_string_ids",
                params={"identifiers": query, "species": SPECIES_ID, "caller_identity": "midnightstar"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return data[0] if data else None
        except Exception:
            return None

    @staticmethod
    def _build_evidence(item: dict) -> str:
        parts = []
        if item.get("escore", 0) > 0: parts.append("experiments")
        if item.get("dscore", 0) > 0: parts.append("databases")
        if item.get("tscore", 0) > 0: parts.append("textmining")
        if item.get("pscore", 0) > 0: parts.append("co-expression")
        return ", ".join(parts) if parts else "combined"
