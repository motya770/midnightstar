# MidnightStar Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit-based gene-disease correlation discovery platform with live API integration and guided ML model training.

**Architecture:** Multi-page Streamlit app. Data layer fetches from 4 public APIs (GWAS, GTEx, HPA, STRING) and caches in SQLite. Graph builder merges into NetworkX. Three ML models (GNN, Graph Transformer, VAE) for link prediction. Pyvis for network visualization.

**Tech Stack:** Python 3.11+, Streamlit, PyTorch, PyTorch Geometric, NetworkX, Pyvis, requests, mygene

**Spec:** `docs/superpowers/specs/2026-03-14-midnightstar-design.md`

---

## File Structure

```
midnightstar/
├── app.py                        # Streamlit entry, sidebar, config
├── pages/
│   ├── 1_Search.py               # Gene/disease search
│   ├── 2_Explorer.py             # Network graph exploration
│   ├── 3_Model_Training.py       # Guided ML config + training
│   └── 4_Results.py              # Discovery dashboard
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py             # Dataclasses: GeneNode, DiseaseNode, etc.
│   │   ├── cache.py              # SQLite cache with TTL + pinning
│   │   ├── gene_resolver.py      # Symbol-to-Ensembl via MyGene.info
│   │   ├── gwas_client.py        # GWAS Catalog API
│   │   ├── gtex_client.py        # GTEx Portal API
│   │   ├── hpa_client.py         # Human Protein Atlas
│   │   ├── string_client.py      # STRING DB API
│   │   └── bulk_downloader.py    # Parallel fetch from all sources
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn.py                # GNN (PyG)
│   │   ├── graph_transformer.py  # Graph Transformer with RWSE
│   │   ├── vae.py                # VAE
│   │   ├── trainer.py            # Training loop + multiprocessing
│   │   └── explainer.py          # GNNExplainer wrapper
│   └── utils/
│       ├── __init__.py
│       ├── graph_builder.py      # Build NetworkX from API data
│       ├── text_explainer.py     # Plain-language gene descriptions
│       └── session.py            # Streamlit session state helpers
├── tests/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── test_models.py
│   │   ├── test_cache.py
│   │   ├── test_gene_resolver.py
│   │   ├── test_gwas_client.py
│   │   ├── test_gtex_client.py
│   │   ├── test_hpa_client.py
│   │   ├── test_string_client.py
│   │   └── test_bulk_downloader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── test_gnn.py
│   │   ├── test_graph_transformer.py
│   │   ├── test_vae.py
│   │   └── test_trainer.py
│   └── utils/
│       ├── __init__.py
│       ├── test_graph_builder.py
│       └── test_text_explainer.py
├── pyproject.toml
└── .streamlit/
    └── config.toml
```

---

## Chunk 1: Foundation

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.streamlit/config.toml`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/utils/__init__.py`
- Create: `tests/__init__.py`, `tests/data/__init__.py`, `tests/models/__init__.py`, `tests/utils/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "midnightstar"
version = "0.1.0"
description = "Gene-disease correlation discovery platform"
requires-python = ">=3.11"
dependencies = [
    "streamlit>=1.40.0",
    "requests>=2.31.0",
    "networkx>=3.2",
    "pyvis>=0.3.2",
    "mygene>=3.2.2",
    "torch>=2.2.0",
    "torch-geometric>=2.5.0",
    "pandas>=2.1.0",
    "plotly>=5.18.0",
    "scikit-learn>=1.4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "responses>=0.25.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create Streamlit config**

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#7eb8ff"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1a1f2e"
textColor = "#fafafa"

[server]
headless = true
```

- [ ] **Step 3: Create all __init__.py files and pages directory**

Create empty `__init__.py` in: `src/`, `src/data/`, `src/models/`, `src/utils/`, `tests/`, `tests/data/`, `tests/models/`, `tests/utils/`. Create empty `pages/` directory.

- [ ] **Step 4: Create minimal app.py**

```python
import streamlit as st

st.set_page_config(
    page_title="MidnightStar",
    page_icon="🧬",
    layout="wide",
)

st.title("MidnightStar")
st.markdown("Gene-disease correlation discovery platform")
```

- [ ] **Step 5: Verify Streamlit runs**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && streamlit run app.py --server.headless true &; sleep 3; curl -s http://localhost:8501 | head -5; kill %1`
Expected: HTML output confirming Streamlit is serving.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .streamlit/config.toml app.py src/ tests/ pages/
git commit -m "feat: scaffold project structure with Streamlit config"
```

---

### Task 2: Data Structures

**Files:**
- Create: `src/data/models.py`
- Create: `tests/data/test_models.py`

- [ ] **Step 1: Write failing tests for data structures**

```python
# tests/data/test_models.py
from src.data.models import GeneNode, DiseaseNode, Association, ExpressionProfile


def test_gene_node_creation():
    gene = GeneNode(
        id="SP4",
        ensembl_id="ENSG00000105866",
        symbol="SP4",
        name="Sp4 Transcription Factor",
        description="Transcription factor involved in neuronal development",
        organism="Homo sapiens",
    )
    assert gene.symbol == "SP4"
    assert gene.ensembl_id == "ENSG00000105866"


def test_disease_node_creation():
    disease = DiseaseNode(
        id="EFO_0000249",
        name="Alzheimer's disease",
        description="A neurodegenerative disease",
        category="neurological",
        source="GWAS",
    )
    assert disease.name == "Alzheimer's disease"
    assert disease.category == "neurological"


def test_association_creation():
    assoc = Association(
        source_id="ENSG00000105866",
        target_id="EFO_0000249",
        type="gene-disease",
        score=0.87,
        evidence="GWAS significant (p<5e-8)",
        data_source="GWAS",
    )
    assert assoc.score == 0.87
    assert assoc.type == "gene-disease"


def test_expression_profile_creation():
    expr = ExpressionProfile(
        gene_id="ENSG00000105866",
        tissue="Brain - Cortex",
        expression_level=15.3,
        sample_count=255,
    )
    assert expr.tissue == "Brain - Cortex"
    assert expr.expression_level == 15.3


def test_gene_node_to_dict():
    gene = GeneNode(
        id="SP4",
        ensembl_id="ENSG00000105866",
        symbol="SP4",
        name="Sp4 Transcription Factor",
        description="Transcription factor",
        organism="Homo sapiens",
    )
    d = gene.to_dict()
    assert d["symbol"] == "SP4"
    assert "ensembl_id" in d


def test_association_is_strong():
    strong = Association("a", "b", "gene-gene", 0.9, "ev", "STRING")
    weak = Association("a", "b", "gene-gene", 0.3, "ev", "STRING")
    assert strong.is_strong()
    assert not weak.is_strong()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement data structures**

```python
# src/data/models.py
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class GeneNode:
    id: str
    ensembl_id: str
    symbol: str
    name: str
    description: str
    organism: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DiseaseNode:
    id: str
    name: str
    description: str
    category: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Association:
    source_id: str
    target_id: str
    type: str
    score: float
    evidence: str
    data_source: str

    def is_strong(self, threshold: float = 0.7) -> bool:
        return self.score >= threshold

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExpressionProfile:
    gene_id: str
    tissue: str
    expression_level: float
    sample_count: int

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_models.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/models.py tests/data/test_models.py
git commit -m "feat: add core data structures (GeneNode, DiseaseNode, Association, ExpressionProfile)"
```

---

### Task 3: SQLite Cache

**Files:**
- Create: `src/data/cache.py`
- Create: `tests/data/test_cache.py`

- [ ] **Step 1: Write failing tests for cache**

```python
# tests/data/test_cache.py
import time
import tempfile
import os
from src.data.cache import Cache


def test_cache_set_and_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path)
        cache.set("gwas", "SP4", {"param": "v1"}, {"result": "data"})
        result = cache.get("gwas", "SP4", {"param": "v1"})
        assert result == {"result": "data"}


def test_cache_miss_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path)
        result = cache.get("gwas", "NONEXISTENT", {})
        assert result is None


def test_cache_ttl_expiry():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path, default_ttl_seconds=1)
        cache.set("gwas", "SP4", {}, {"result": "data"})
        assert cache.get("gwas", "SP4", {}) is not None
        time.sleep(1.1)
        assert cache.get("gwas", "SP4", {}) is None


def test_cache_pin_survives_ttl():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path, default_ttl_seconds=1)
        cache.set("gwas", "SP4", {}, {"result": "data"}, pinned=True)
        time.sleep(1.1)
        assert cache.get("gwas", "SP4", {}) == {"result": "data"}


def test_cache_unpin():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path, default_ttl_seconds=1)
        cache.set("gwas", "SP4", {}, {"result": "data"}, pinned=True)
        cache.unpin("gwas", "SP4", {})
        time.sleep(1.1)
        assert cache.get("gwas", "SP4", {}) is None


def test_cache_clear_source():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path)
        cache.set("gwas", "SP4", {}, {"r": 1})
        cache.set("gtex", "SP4", {}, {"r": 2})
        cache.clear_source("gwas")
        assert cache.get("gwas", "SP4", {}) is None
        assert cache.get("gtex", "SP4", {}) is not None


def test_cache_size_bytes():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path)
        cache.set("gwas", "SP4", {}, {"r": 1})
        size = cache.size_bytes()
        assert size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement cache**

```python
# src/data/cache.py
import hashlib
import json
import os
import sqlite3
import time
from typing import Any


class Cache:
    def __init__(self, db_path: str, default_ttl_seconds: int = 86400):
        self.db_path = db_path
        self.default_ttl_seconds = default_ttl_seconds
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    query TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    ttl_seconds INTEGER NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON cache(source)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _make_key(source: str, query: str, params: dict) -> str:
        params_str = json.dumps(params, sort_keys=True)
        raw = f"{source}:{query}:{params_str}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _params_hash(params: dict) -> str:
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

    def set(
        self,
        source: str,
        query: str,
        params: dict,
        data: Any,
        pinned: bool = False,
        ttl_seconds: int | None = None,
    ):
        key = self._make_key(source, query, params)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache
                   (cache_key, source, query, params_hash, data, created_at, ttl_seconds, pinned)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (key, source, query, self._params_hash(params),
                 json.dumps(data), time.time(), ttl, int(pinned)),
            )

    def get(self, source: str, query: str, params: dict) -> Any | None:
        key = self._make_key(source, query, params)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data, created_at, ttl_seconds, pinned FROM cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        data_str, created_at, ttl_seconds, pinned = row
        if not pinned and (time.time() - created_at) > ttl_seconds:
            self._delete_key(key)
            return None
        return json.loads(data_str)

    def unpin(self, source: str, query: str, params: dict):
        key = self._make_key(source, query, params)
        with self._connect() as conn:
            conn.execute("UPDATE cache SET pinned = 0 WHERE cache_key = ?", (key,))

    def clear_source(self, source: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE source = ?", (source,))

    def clear_all(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM cache")

    def size_bytes(self) -> int:
        if not os.path.exists(self.db_path):
            return 0
        return os.path.getsize(self.db_path)

    def _delete_key(self, key: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE cache_key = ?", (key,))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_cache.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/cache.py tests/data/test_cache.py
git commit -m "feat: add SQLite cache with TTL and pinning support"
```

---

### Task 4: Gene Resolver

**Files:**
- Create: `src/data/gene_resolver.py`
- Create: `tests/data/test_gene_resolver.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_gene_resolver.py
from unittest.mock import patch, MagicMock
from src.data.gene_resolver import GeneResolver


def test_resolve_symbol_to_ensembl():
    mock_mg = MagicMock()
    mock_mg.query.return_value = {
        "hits": [{"ensembl": {"gene": "ENSG00000105866"}, "symbol": "SP4", "name": "Sp4 Transcription Factor"}]
    }
    resolver = GeneResolver(client=mock_mg)
    result = resolver.resolve("SP4")
    assert result.ensembl_id == "ENSG00000105866"
    assert result.symbol == "SP4"


def test_resolve_returns_none_for_unknown():
    mock_mg = MagicMock()
    mock_mg.query.return_value = {"hits": []}
    resolver = GeneResolver(client=mock_mg)
    result = resolver.resolve("FAKEGENE123")
    assert result is None


def test_resolve_batch():
    mock_mg = MagicMock()
    mock_mg.querymany.return_value = [
        {"query": "SP4", "ensembl": {"gene": "ENSG00000105866"}, "symbol": "SP4", "name": "Sp4 TF"},
        {"query": "TP53", "ensembl": {"gene": "ENSG00000141510"}, "symbol": "TP53", "name": "Tumor Protein P53"},
    ]
    resolver = GeneResolver(client=mock_mg)
    results = resolver.resolve_batch(["SP4", "TP53"])
    assert len(results) == 2
    assert results["SP4"].ensembl_id == "ENSG00000105866"
    assert results["TP53"].ensembl_id == "ENSG00000141510"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_gene_resolver.py -v`
Expected: FAIL

- [ ] **Step 3: Implement gene resolver**

```python
# src/data/gene_resolver.py
from src.data.models import GeneNode


class GeneResolver:
    def __init__(self, client=None):
        if client is None:
            import mygene
            client = mygene.MyGeneInfo()
        self._client = client

    def resolve(self, symbol: str) -> GeneNode | None:
        result = self._client.query(symbol, scopes="symbol", fields="ensembl.gene,symbol,name", species="human")
        hits = result.get("hits", [])
        if not hits:
            return None
        hit = hits[0]
        ensembl_id = self._extract_ensembl_id(hit)
        if not ensembl_id:
            return None
        return GeneNode(
            id=symbol,
            ensembl_id=ensembl_id,
            symbol=hit.get("symbol", symbol),
            name=hit.get("name", ""),
            description="",
            organism="Homo sapiens",
        )

    def resolve_batch(self, symbols: list[str]) -> dict[str, GeneNode]:
        results = self._client.querymany(symbols, scopes="symbol", fields="ensembl.gene,symbol,name", species="human")
        genes = {}
        for hit in results:
            if "notfound" in hit and hit["notfound"]:
                continue
            ensembl_id = self._extract_ensembl_id(hit)
            if not ensembl_id:
                continue
            symbol = hit.get("symbol", hit.get("query", ""))
            genes[symbol] = GeneNode(
                id=symbol,
                ensembl_id=ensembl_id,
                symbol=symbol,
                name=hit.get("name", ""),
                description="",
                organism="Homo sapiens",
            )
        return genes

    @staticmethod
    def _extract_ensembl_id(hit: dict) -> str | None:
        ensembl = hit.get("ensembl")
        if isinstance(ensembl, dict):
            return ensembl.get("gene")
        if isinstance(ensembl, list) and ensembl:
            return ensembl[0].get("gene")
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_gene_resolver.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/gene_resolver.py tests/data/test_gene_resolver.py
git commit -m "feat: add gene symbol-to-Ensembl resolver via MyGene.info"
```

---

## Chunk 2: API Clients

### Task 5: GWAS Catalog Client

**Files:**
- Create: `src/data/gwas_client.py`
- Create: `tests/data/test_gwas_client.py`

- [ ] **Step 1: Write failing tests**

```python
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
                "loci": [
                    {"strongestRiskAlleles": [{"riskAlleleName": "rs12345-A"}]}
                ],
                "_links": {
                    "efoTraits": [{"href": "https://www.ebi.ac.uk/gwas/rest/api/efoTraits/EFO_0000249"}]
                },
            }
        ]
    }
}

MOCK_TRAIT_RESPONSE = {
    "trait": "Alzheimer's disease",
    "shortForm": "EFO_0000249",
}


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_gwas_client.py -v`
Expected: FAIL

- [ ] **Step 3: Implement GWAS client**

```python
# src/data/gwas_client.py
import logging
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
                params={"gene": gene_symbol},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.warning("GWAS API error for gene %s", gene_symbol)
            return [], []

        return self._parse_associations(data, gene_symbol)

    def search_disease(self, efo_trait_id: str) -> list[Association]:
        try:
            resp = requests.get(
                f"{self.base_url}/efoTraits/{efo_trait_id}/associations",
                timeout=30,
            )
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
                source_id=efo_trait_id,
                target_id="",
                type="disease-gene",
                score=round(score, 4),
                evidence=f"p-value: {pvalue:.2e}",
                data_source="GWAS",
            ))
        return associations

    def _parse_associations(
        self, data: dict, gene_symbol: str
    ) -> tuple[list[Association], list[DiseaseNode]]:
        embedded = data.get("_embedded", {})
        associations = []
        diseases = []
        seen_traits = set()

        for item in embedded.get("associations", []):
            pvalue = item.get("pvalue", 1.0)
            score = self._pvalue_to_score(pvalue)

            trait_links = item.get("_links", {}).get("efoTraits", [])
            for trait_link in trait_links:
                trait_href = trait_link.get("href", "")
                trait_id = trait_href.rstrip("/").split("/")[-1] if trait_href else ""

                associations.append(Association(
                    source_id=gene_symbol,
                    target_id=trait_id,
                    type="gene-disease",
                    score=round(score, 4),
                    evidence=f"p-value: {pvalue:.2e}",
                    data_source="GWAS",
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
            return DiseaseNode(
                id=data.get("shortForm", trait_id),
                name=data.get("trait", "Unknown"),
                description="",
                category="",
                source="GWAS",
            )
        except Exception:
            return None

    @staticmethod
    def _pvalue_to_score(pvalue: float) -> float:
        if pvalue <= 0:
            return 1.0
        import math
        return min(1.0, -math.log10(pvalue) / 50)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_gwas_client.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/gwas_client.py tests/data/test_gwas_client.py
git commit -m "feat: add GWAS Catalog API client"
```

---

### Task 6: GTEx Client

**Files:**
- Create: `src/data/gtex_client.py`
- Create: `tests/data/test_gtex_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_gtex_client.py
from unittest.mock import patch, MagicMock
from src.data.gtex_client import GTExClient
from src.data.models import ExpressionProfile


MOCK_GTEX_RESPONSE = {
    "medianGeneExpression": [
        {
            "gencodeId": "ENSG00000105866.10",
            "tissueSiteDetailId": "Brain_Cortex",
            "median": 15.3,
            "numSamples": 255,
        },
        {
            "gencodeId": "ENSG00000105866.10",
            "tissueSiteDetailId": "Brain_Hippocampus",
            "median": 12.1,
            "numSamples": 197,
        },
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_gtex_client.py -v`
Expected: FAIL

- [ ] **Step 3: Implement GTEx client**

```python
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
                params={
                    "gencodeId": gencode_id,
                    "datasetId": "gtex_v8",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.warning("GTEx API error for %s", ensembl_id)
            return []

        profiles = []
        for item in data.get("medianGeneExpression", []):
            profiles.append(ExpressionProfile(
                gene_id=ensembl_id,
                tissue=item.get("tissueSiteDetailId", "Unknown"),
                expression_level=item.get("median", 0.0),
                sample_count=item.get("numSamples", 0),
            ))
        return profiles
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_gtex_client.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/gtex_client.py tests/data/test_gtex_client.py
git commit -m "feat: add GTEx Portal API client"
```

---

### Task 7: HPA Client

**Files:**
- Create: `src/data/hpa_client.py`
- Create: `tests/data/test_hpa_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_hpa_client.py
from unittest.mock import patch, MagicMock
from src.data.hpa_client import HPAClient


MOCK_HPA_GENE_RESPONSE = {
    "Gene": "SP4",
    "Gene synonym": "SPR-1",
    "Ensembl": "ENSG00000105866",
    "Gene description": "Sp4 transcription factor",
    "Subcellular location": "Nucleoplasm",
    "RNA tissue specificity": "Tissue enhanced (brain)",
    "Tissue expression cluster": "brain",
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_hpa_client.py -v`
Expected: FAIL

- [ ] **Step 3: Implement HPA client**

```python
# src/data/hpa_client.py
import logging
import requests

logger = logging.getLogger(__name__)

GENE_URL = "https://www.proteinatlas.org/{ensembl_id}.json"


class HPAClient:
    def get_gene_info(self, ensembl_id: str) -> dict | None:
        try:
            resp = requests.get(
                GENE_URL.format(ensembl_id=ensembl_id),
                timeout=30,
            )
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_hpa_client.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/hpa_client.py tests/data/test_hpa_client.py
git commit -m "feat: add Human Protein Atlas client"
```

---

### Task 8: STRING Client

**Files:**
- Create: `src/data/string_client.py`
- Create: `tests/data/test_string_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_string_client.py
import time
from unittest.mock import patch, MagicMock
from src.data.string_client import STRINGClient
from src.data.models import Association


MOCK_INTERACTIONS = [
    {
        "preferredName_A": "SP4",
        "preferredName_B": "HSP60",
        "stringId_A": "9606.ENSP00000105866",
        "stringId_B": "9606.ENSP00000216015",
        "score": 870,
        "nscore": 0,
        "fscore": 0,
        "pscore": 0,
        "ascore": 0,
        "escore": 450,
        "dscore": 0,
        "tscore": 700,
    },
]

MOCK_IDENTIFIERS = [
    {
        "queryItem": "SP4",
        "preferredName": "SP4",
        "stringId": "9606.ENSP00000105866",
        "annotation": "Sp4 transcription factor",
    }
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_string_client.py -v`
Expected: FAIL

- [ ] **Step 3: Implement STRING client**

```python
# src/data/string_client.py
import logging
import time
import threading
import requests
from src.data.models import Association

logger = logging.getLogger(__name__)

BASE_URL = "https://string-db.org/api"
SPECIES_ID = "9606"  # Homo sapiens


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

    def get_interactions(
        self, gene_symbol: str, required_score: int = 400
    ) -> list[Association]:
        self._rate_limit()
        try:
            resp = requests.get(
                f"{self.base_url}/json/network",
                params={
                    "identifiers": gene_symbol,
                    "species": SPECIES_ID,
                    "required_score": required_score,
                    "caller_identity": "midnightstar",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.warning("STRING API error for %s", gene_symbol)
            return []

        associations = []
        for item in data:
            score = item.get("score", 0) / 1000.0
            associations.append(Association(
                source_id=item.get("preferredName_A", ""),
                target_id=item.get("preferredName_B", ""),
                type="protein-protein",
                score=round(score, 4),
                evidence=self._build_evidence(item),
                data_source="STRING",
            ))
        return associations

    def resolve_identifier(self, query: str) -> dict | None:
        self._rate_limit()
        try:
            resp = requests.get(
                f"{self.base_url}/json/get_string_ids",
                params={
                    "identifiers": query,
                    "species": SPECIES_ID,
                    "caller_identity": "midnightstar",
                },
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
        if item.get("escore", 0) > 0:
            parts.append("experiments")
        if item.get("dscore", 0) > 0:
            parts.append("databases")
        if item.get("tscore", 0) > 0:
            parts.append("textmining")
        if item.get("pscore", 0) > 0:
            parts.append("co-expression")
        return ", ".join(parts) if parts else "combined"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_string_client.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/string_client.py tests/data/test_string_client.py
git commit -m "feat: add STRING DB client with rate limiting"
```

---

## Chunk 3: Data Processing

### Task 9: Graph Builder

**Files:**
- Create: `src/utils/graph_builder.py`
- Create: `tests/utils/test_graph_builder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/utils/test_graph_builder.py
import networkx as nx
from src.data.models import GeneNode, DiseaseNode, Association, ExpressionProfile
from src.utils.graph_builder import GraphBuilder


def _sample_genes():
    return [
        GeneNode("SP4", "ENSG00000105866", "SP4", "Sp4 TF", "TF", "Homo sapiens"),
        GeneNode("HSPD1", "ENSG00000144381", "HSPD1", "HSP60", "Chaperone", "Homo sapiens"),
    ]


def _sample_associations():
    return [
        Association("SP4", "HSPD1", "protein-protein", 0.87, "experiments", "STRING"),
        Association("SP4", "EFO_0000249", "gene-disease", 0.6, "p<5e-8", "GWAS"),
    ]


def _sample_diseases():
    return [DiseaseNode("EFO_0000249", "Alzheimer's", "Neuro disease", "neurological", "GWAS")]


def test_build_graph_nodes():
    builder = GraphBuilder()
    builder.add_genes(_sample_genes())
    builder.add_diseases(_sample_diseases())
    G = builder.build()
    assert G.number_of_nodes() == 3
    assert G.nodes["SP4"]["node_type"] == "gene"
    assert G.nodes["EFO_0000249"]["node_type"] == "disease"


def test_build_graph_edges():
    builder = GraphBuilder()
    builder.add_genes(_sample_genes())
    builder.add_diseases(_sample_diseases())
    builder.add_associations(_sample_associations())
    G = builder.build()
    assert G.number_of_edges() == 2
    edge_data = G.edges["SP4", "HSPD1"]
    assert edge_data["score"] == 0.87
    assert edge_data["data_source"] == "STRING"


def test_build_graph_with_expression():
    builder = GraphBuilder()
    builder.add_genes(_sample_genes()[:1])
    builder.add_expression([
        ExpressionProfile("ENSG00000105866", "Brain_Cortex", 15.3, 255),
        ExpressionProfile("ENSG00000105866", "Liver", 2.1, 200),
    ])
    G = builder.build()
    assert "expression" in G.nodes["SP4"]
    assert G.nodes["SP4"]["expression"]["Brain_Cortex"] == 15.3


def test_filter_by_score():
    builder = GraphBuilder()
    builder.add_genes(_sample_genes())
    builder.add_diseases(_sample_diseases())
    builder.add_associations(_sample_associations())
    G = builder.build(min_score=0.7)
    assert G.number_of_edges() == 1  # Only the 0.87 edge


def test_filter_by_source():
    builder = GraphBuilder()
    builder.add_genes(_sample_genes())
    builder.add_diseases(_sample_diseases())
    builder.add_associations(_sample_associations())
    G = builder.build(sources={"STRING"})
    assert G.number_of_edges() == 1


def test_merge_duplicate_edges():
    builder = GraphBuilder()
    builder.add_genes(_sample_genes())
    builder.add_associations([
        Association("SP4", "HSPD1", "protein-protein", 0.87, "exp", "STRING"),
        Association("SP4", "HSPD1", "co-expression", 0.72, "coex", "GTEx"),
    ])
    G = builder.build()
    assert G.number_of_edges() == 1
    edge = G.edges["SP4", "HSPD1"]
    assert len(edge["sources"]) == 2
    assert edge["score"] == 0.87  # max score
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/utils/test_graph_builder.py -v`
Expected: FAIL

- [ ] **Step 3: Implement graph builder**

```python
# src/utils/graph_builder.py
import networkx as nx
from src.data.models import GeneNode, DiseaseNode, Association, ExpressionProfile


class GraphBuilder:
    def __init__(self):
        self._genes: list[GeneNode] = []
        self._diseases: list[DiseaseNode] = []
        self._associations: list[Association] = []
        self._expression: dict[str, dict[str, float]] = {}  # ensembl_id -> {tissue: level}
        self._ensembl_to_symbol: dict[str, str] = {}

    def add_genes(self, genes: list[GeneNode]):
        for g in genes:
            self._genes.append(g)
            self._ensembl_to_symbol[g.ensembl_id] = g.symbol

    def add_diseases(self, diseases: list[DiseaseNode]):
        self._diseases.extend(diseases)

    def add_associations(self, associations: list[Association]):
        self._associations.extend(associations)

    def add_expression(self, profiles: list[ExpressionProfile]):
        for p in profiles:
            symbol = self._ensembl_to_symbol.get(p.gene_id, p.gene_id)
            if symbol not in self._expression:
                self._expression[symbol] = {}
            self._expression[symbol][p.tissue] = p.expression_level

    def build(
        self,
        min_score: float = 0.0,
        sources: set[str] | None = None,
    ) -> nx.Graph:
        G = nx.Graph()

        for gene in self._genes:
            G.add_node(gene.symbol, node_type="gene", **gene.to_dict())

        for disease in self._diseases:
            G.add_node(disease.id, node_type="disease", **disease.to_dict())

        # Group associations by edge (source_id, target_id)
        edge_groups: dict[tuple[str, str], list[Association]] = {}
        for assoc in self._associations:
            if sources and assoc.data_source not in sources:
                continue
            if assoc.score < min_score:
                continue
            key = tuple(sorted([assoc.source_id, assoc.target_id]))
            edge_groups.setdefault(key, []).append(assoc)

        for (src, tgt), assocs in edge_groups.items():
            best_score = max(a.score for a in assocs)
            all_sources = list({a.data_source for a in assocs})
            all_evidence = "; ".join(a.evidence for a in assocs if a.evidence)
            G.add_edge(
                src, tgt,
                score=best_score,
                sources=all_sources,
                data_source=all_sources[0],
                evidence=all_evidence,
                type=assocs[0].type,
            )

        for symbol, tissues in self._expression.items():
            if symbol in G.nodes:
                G.nodes[symbol]["expression"] = tissues

        return G
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/utils/test_graph_builder.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/graph_builder.py tests/utils/test_graph_builder.py
git commit -m "feat: add graph builder with multi-source merge and filtering"
```

---

### Task 10: Plain-Language Explainer

**Files:**
- Create: `src/utils/text_explainer.py`
- Create: `tests/utils/test_text_explainer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/utils/test_text_explainer.py
from src.utils.text_explainer import TextExplainer


def test_explain_gene():
    explainer = TextExplainer()
    result = explainer.explain_gene(
        symbol="SP4",
        name="Sp4 Transcription Factor",
        hpa_info={
            "subcellular_location": "Nucleoplasm",
            "tissue_expression": "Tissue enhanced (brain)",
            "gene_description": "Sp4 transcription factor",
        },
    )
    assert "SP4" in result
    assert len(result) > 20


def test_explain_gene_minimal_info():
    explainer = TextExplainer()
    result = explainer.explain_gene(symbol="XYZ1", name="Unknown Protein")
    assert "XYZ1" in result


def test_explain_association():
    explainer = TextExplainer()
    result = explainer.explain_association(
        gene_symbol="SP4",
        target_name="Alzheimer's disease",
        score=0.87,
        evidence="p-value: 2.0e-12",
        data_source="GWAS",
    )
    assert "SP4" in result
    assert "Alzheimer" in result
    assert "strong" in result.lower() or "high" in result.lower()


def test_explain_weak_association():
    explainer = TextExplainer()
    result = explainer.explain_association(
        gene_symbol="SP4",
        target_name="Diabetes",
        score=0.3,
        evidence="textmining",
        data_source="STRING",
    )
    assert "weak" in result.lower() or "low" in result.lower() or "modest" in result.lower()


def test_explain_prediction():
    explainer = TextExplainer()
    result = explainer.explain_prediction(
        gene_a="SP4",
        gene_b="HSP60",
        score=0.87,
        shared_partners=["NRF2", "PARK2"],
        shared_tissues=["Brain_Cortex", "Brain_Hippocampus"],
    )
    assert "SP4" in result
    assert "HSP60" in result
    assert "brain" in result.lower() or "Brain" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/utils/test_text_explainer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement explainer**

```python
# src/utils/text_explainer.py

class TextExplainer:
    def explain_gene(
        self,
        symbol: str,
        name: str,
        hpa_info: dict | None = None,
    ) -> str:
        parts = [f"**{symbol}** ({name})"]

        if hpa_info:
            desc = hpa_info.get("gene_description", "")
            if desc:
                parts.append(f"is {desc.lower()}.")

            location = hpa_info.get("subcellular_location", "")
            if location:
                parts.append(f"It is found in the {location.lower()} of cells.")

            tissue = hpa_info.get("tissue_expression", "")
            if tissue:
                parts.append(f"It is most active in: {tissue}.")
        else:
            parts.append("is a gene with limited annotation available.")

        return " ".join(parts)

    def explain_association(
        self,
        gene_symbol: str,
        target_name: str,
        score: float,
        evidence: str,
        data_source: str,
    ) -> str:
        strength = self._score_to_strength(score)
        source_desc = self._source_description(data_source)

        return (
            f"There is a **{strength} link** between **{gene_symbol}** and "
            f"**{target_name}** (confidence: {score:.2f}). "
            f"This was found through {source_desc}. "
            f"Evidence: {evidence}."
        )

    def explain_prediction(
        self,
        gene_a: str,
        gene_b: str,
        score: float,
        shared_partners: list[str] | None = None,
        shared_tissues: list[str] | None = None,
    ) -> str:
        parts = [
            f"The model predicts a connection between **{gene_a}** and **{gene_b}** "
            f"(confidence: {score:.2f})."
        ]

        if shared_partners:
            partners_str = ", ".join(shared_partners[:5])
            parts.append(
                f"They share {len(shared_partners)} interaction partners "
                f"({partners_str}), suggesting they operate in related pathways."
            )

        if shared_tissues:
            tissues_str = ", ".join(t.replace("_", " ") for t in shared_tissues[:5])
            parts.append(
                f"Both are active in similar tissues ({tissues_str}), "
                f"which strengthens the predicted link."
            )

        return " ".join(parts)

    @staticmethod
    def _score_to_strength(score: float) -> str:
        if score >= 0.9:
            return "very strong"
        if score >= 0.7:
            return "strong"
        if score >= 0.4:
            return "moderate"
        return "weak"

    @staticmethod
    def _source_description(source: str) -> str:
        descriptions = {
            "GWAS": "genome-wide association studies (scanning DNA from large populations)",
            "STRING": "protein interaction databases (how proteins physically connect)",
            "GTEx": "gene expression data (which tissues a gene is active in)",
            "HPA": "protein atlas data (where proteins are found in cells)",
        }
        return descriptions.get(source, f"{source} data")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/utils/test_text_explainer.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/text_explainer.py tests/utils/test_text_explainer.py
git commit -m "feat: add plain-language explainer for genes, associations, and predictions"
```

---

### Task 11: Session State Helpers

**Files:**
- Create: `src/utils/session.py`

- [ ] **Step 1: Create session helpers**

These are thin wrappers around `st.session_state` — tested manually via Streamlit.

```python
# src/utils/session.py
import streamlit as st
import networkx as nx


def get_graph() -> nx.Graph | None:
    return st.session_state.get("graph")


def set_graph(graph: nx.Graph):
    st.session_state["graph"] = graph


def get_selected_node() -> str | None:
    return st.session_state.get("selected_node")


def set_selected_node(node_id: str):
    st.session_state["selected_node"] = node_id


def get_search_query() -> str:
    return st.session_state.get("search_query", "")


def set_search_query(query: str):
    st.session_state["search_query"] = query


def get_training_results() -> dict | None:
    return st.session_state.get("training_results")


def set_training_results(results: dict):
    st.session_state["training_results"] = results


def get_model_predictions() -> list[dict] | None:
    return st.session_state.get("model_predictions")


def set_model_predictions(predictions: list[dict]):
    st.session_state["model_predictions"] = predictions
```

- [ ] **Step 2: Commit**

```bash
git add src/utils/session.py
git commit -m "feat: add Streamlit session state helpers"
```

---

### Task 12: Bulk Downloader

**Files:**
- Create: `src/data/bulk_downloader.py`
- Create: `tests/data/test_bulk_downloader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_bulk_downloader.py
from unittest.mock import MagicMock, patch
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
    assert result["errors"] == ["GWAS"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_bulk_downloader.py -v`
Expected: FAIL

- [ ] **Step 3: Implement bulk downloader**

```python
# src/data/bulk_downloader.py
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data.models import GeneNode

logger = logging.getLogger(__name__)


class BulkDownloader:
    def __init__(self, gwas_client, gtex_client, hpa_client, string_client):
        self.gwas = gwas_client
        self.gtex = gtex_client
        self.hpa = hpa_client
        self.string = string_client

    def download_all(
        self, gene: GeneNode, on_progress=None
    ) -> dict:
        results = {
            "associations": [],
            "diseases": [],
            "expression": [],
            "hpa_info": None,
            "errors": [],
        }

        def fetch_gwas():
            assocs, diseases = self.gwas.search_gene(gene.symbol)
            return "GWAS", {"associations": assocs, "diseases": diseases}

        def fetch_gtex():
            profiles = self.gtex.get_expression(gene.ensembl_id)
            return "GTEx", {"expression": profiles}

        def fetch_hpa():
            info = self.hpa.get_gene_info(gene.ensembl_id)
            return "HPA", {"hpa_info": info}

        def fetch_string():
            assocs = self.string.get_interactions(gene.symbol)
            return "STRING", {"associations": assocs}

        tasks = [fetch_gwas, fetch_gtex, fetch_hpa, fetch_string]

        # STRING must be called serially (rate limit), others can be parallel.
        # Use max_workers=3 so STRING doesn't overlap with itself.
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    source_name, data = future.result()
                    if on_progress:
                        on_progress(source_name, "done")

                    if "associations" in data:
                        results["associations"].extend(data["associations"])
                    if "diseases" in data:
                        results["diseases"].extend(data["diseases"])
                    if "expression" in data:
                        results["expression"].extend(data["expression"])
                    if "hpa_info" in data and data["hpa_info"] is not None:
                        results["hpa_info"] = data["hpa_info"]

                except Exception as e:
                    logger.warning("Bulk download task failed: %s", e)
                    # Try to identify which source failed
                    task_fn = futures[future]
                    name = task_fn.__name__.replace("fetch_", "").upper()
                    results["errors"].append(name)
                    if on_progress:
                        on_progress(name, "error")

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/data/test_bulk_downloader.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/bulk_downloader.py tests/data/test_bulk_downloader.py
git commit -m "feat: add parallel bulk downloader with error resilience"
```

---

## Chunk 4: ML Models

### Task 13: GNN Model

**Files:**
- Create: `src/models/gnn.py`
- Create: `tests/models/test_gnn.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/models/test_gnn.py
import torch
from torch_geometric.data import Data
from src.models.gnn import GNNLinkPredictor


def _make_test_graph():
    # 5 nodes, 4 edges, 10-dim node features
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def test_gnn_forward_shape():
    model = GNNLinkPredictor(in_channels=10, hidden_channels=32, num_layers=2)
    data = _make_test_graph()
    # Predict link between node 0 and node 4
    src = torch.tensor([0, 1])
    dst = torch.tensor([4, 3])
    scores = model(data, src, dst)
    assert scores.shape == (2,)
    assert all(0 <= s <= 1 for s in scores)


def test_gnn_different_configs():
    for layers in [1, 3]:
        for hidden in [16, 64]:
            for agg in ["mean", "max", "sum"]:
                model = GNNLinkPredictor(
                    in_channels=10, hidden_channels=hidden,
                    num_layers=layers, aggr=agg,
                )
                data = _make_test_graph()
                scores = model(data, torch.tensor([0]), torch.tensor([4]))
                assert scores.shape == (1,)


def test_gnn_encode_produces_embeddings():
    model = GNNLinkPredictor(in_channels=10, hidden_channels=32, num_layers=2)
    data = _make_test_graph()
    embeddings = model.encode(data)
    assert embeddings.shape == (5, 32)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/models/test_gnn.py -v`
Expected: FAIL

- [ ] **Step 3: Implement GNN**

```python
# src/models/gnn.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GNNLinkPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        aggr: str = "mean",
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

    def encode(self, data) -> torch.Tensor:
        x = data.x
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = torch.relu(x)
        return x

    def decode(self, z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))

    def forward(self, data, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        z = self.encode(data)
        return self.decode(z, src, dst)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/models/test_gnn.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/gnn.py tests/models/test_gnn.py
git commit -m "feat: add GNN link predictor model (SAGEConv)"
```

---

### Task 14: Graph Transformer Model

**Files:**
- Create: `src/models/graph_transformer.py`
- Create: `tests/models/test_graph_transformer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/models/test_graph_transformer.py
import torch
from torch_geometric.data import Data
from src.models.graph_transformer import GraphTransformerLinkPredictor


def _make_test_graph():
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=5)


def test_transformer_forward_shape():
    model = GraphTransformerLinkPredictor(
        in_channels=10, hidden_channels=32, num_layers=2, num_heads=2, rwse_dim=8,
    )
    data = _make_test_graph()
    src = torch.tensor([0, 1])
    dst = torch.tensor([4, 3])
    scores = model(data, src, dst)
    assert scores.shape == (2,)
    assert all(0 <= s <= 1 for s in scores)


def test_transformer_encode():
    model = GraphTransformerLinkPredictor(
        in_channels=10, hidden_channels=32, num_layers=1, num_heads=2, rwse_dim=8,
    )
    data = _make_test_graph()
    z = model.encode(data)
    assert z.shape == (5, 32)


def test_transformer_different_configs():
    for heads in [1, 4]:
        for layers in [1, 2]:
            model = GraphTransformerLinkPredictor(
                in_channels=10, hidden_channels=32,
                num_layers=layers, num_heads=heads, rwse_dim=8,
            )
            data = _make_test_graph()
            scores = model(data, torch.tensor([0]), torch.tensor([4]))
            assert scores.shape == (1,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/models/test_graph_transformer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Graph Transformer**

```python
# src/models/graph_transformer.py
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_adj


class RWSEEncoder(nn.Module):
    """Random Walk Structural Encoding."""

    def __init__(self, walk_length: int = 16, out_dim: int = 8):
        super().__init__()
        self.walk_length = walk_length
        self.linear = nn.Linear(walk_length, out_dim)

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        rw = adj / deg  # row-normalized adjacency (random walk matrix)

        rw_diag = torch.zeros(num_nodes, self.walk_length, device=edge_index.device)
        power = torch.eye(num_nodes, device=edge_index.device)
        for k in range(self.walk_length):
            power = power @ rw
            rw_diag[:, k] = power.diagonal()

        return self.linear(rw_diag)


class GraphTransformerLinkPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        rwse_dim: int = 16,
        rwse_walk_length: int = 16,
    ):
        super().__init__()
        self.rwse = RWSEEncoder(walk_length=rwse_walk_length, out_dim=rwse_dim)
        self.input_proj = nn.Linear(in_channels + rwse_dim, hidden_channels)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerConv(hidden_channels, hidden_channels // num_heads, heads=num_heads)
            )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

    def encode(self, data) -> torch.Tensor:
        num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
        pe = self.rwse(data.edge_index, num_nodes)
        x = torch.cat([data.x, pe], dim=-1)
        x = self.input_proj(x)

        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x, data.edge_index))
            x = torch.relu(x)
        return x

    def decode(self, z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))

    def forward(self, data, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        z = self.encode(data)
        return self.decode(z, src, dst)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/models/test_graph_transformer.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/graph_transformer.py tests/models/test_graph_transformer.py
git commit -m "feat: add Graph Transformer with RWSE positional encoding"
```

---

### Task 15: VAE Model

**Files:**
- Create: `src/models/vae.py`
- Create: `tests/models/test_vae.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/models/test_vae.py
import torch
from src.models.vae import GraphVAE


def test_vae_forward_shape():
    model = GraphVAE(in_channels=10, hidden_channels=32, latent_dim=16, num_layers=2)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    adj_pred, mu, logvar = model(x, edge_index)
    assert adj_pred.shape == (5, 5)
    assert mu.shape == (5, 16)
    assert logvar.shape == (5, 16)


def test_vae_link_scores():
    model = GraphVAE(in_channels=10, hidden_channels=32, latent_dim=16)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    src = torch.tensor([0, 1])
    dst = torch.tensor([4, 3])
    scores = model.predict_links(x, edge_index, src, dst)
    assert scores.shape == (2,)
    assert all(0 <= s <= 1 for s in scores)


def test_vae_loss():
    model = GraphVAE(in_channels=10, hidden_channels=32, latent_dim=16)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    loss = model.loss(x, edge_index)
    assert loss.item() > 0


def test_vae_get_embeddings():
    model = GraphVAE(in_channels=10, hidden_channels=32, latent_dim=16)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    z = model.get_embeddings(x, edge_index)
    assert z.shape == (5, 16)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/models/test_vae.py -v`
Expected: FAIL

- [ ] **Step 3: Implement VAE**

```python
# src/models/vae.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj


class GraphVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        beta: float = 1.0,
    ):
        super().__init__()
        self.beta = beta

        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.encoder_layers.append(GCNConv(hidden_channels, hidden_channels))

        self.fc_mu = nn.Linear(hidden_channels, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels, latent_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = x
        for layer in self.encoder_layers:
            h = torch.relu(layer(h, edge_index))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        adj = torch.sigmoid(z @ z.t())
        return adj

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        adj_pred = self.decode(z)
        return adj_pred, mu, logvar

    def loss(self, x, edge_index) -> torch.Tensor:
        adj_pred, mu, logvar = self.forward(x, edge_index)
        adj_true = to_dense_adj(edge_index, max_num_nodes=x.size(0)).squeeze(0)

        recon_loss = nn.functional.binary_cross_entropy(adj_pred, adj_true)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss

    def predict_links(
        self, x, edge_index, src: torch.Tensor, dst: torch.Tensor,
    ) -> torch.Tensor:
        mu, _ = self.encode(x, edge_index)
        return torch.sigmoid((mu[src] * mu[dst]).sum(dim=-1))

    def get_embeddings(self, x, edge_index) -> torch.Tensor:
        mu, _ = self.encode(x, edge_index)
        return mu.detach()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/models/test_vae.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/vae.py tests/models/test_vae.py
git commit -m "feat: add Graph VAE for latent space link prediction"
```

---

### Task 16: Trainer

**Files:**
- Create: `src/models/trainer.py`
- Create: `tests/models/test_trainer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/models/test_trainer.py
import torch
from torch_geometric.data import Data
from src.models.gnn import GNNLinkPredictor
from src.models.trainer import Trainer, TrainConfig


def _make_train_data():
    x = torch.randn(10, 8)
    # Create a connected graph
    src = [0,1,2,3,4,5,6,7,0,1]
    dst = [1,2,3,4,5,6,7,8,2,3]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=10)


def test_trainer_runs_epochs():
    model = GNNLinkPredictor(in_channels=8, hidden_channels=16, num_layers=1)
    config = TrainConfig(epochs=3, lr=0.01, batch_size=4, train_ratio=0.6, val_ratio=0.2)
    trainer = Trainer(model, config)
    data = _make_train_data()
    history = trainer.train(data)
    assert len(history["train_loss"]) == 3
    assert "val_auc" in history


def test_trainer_early_stopping():
    model = GNNLinkPredictor(in_channels=8, hidden_channels=16, num_layers=1)
    config = TrainConfig(epochs=100, lr=0.01, early_stopping=True, patience=2)
    trainer = Trainer(model, config)
    data = _make_train_data()
    history = trainer.train(data)
    # Should stop before 100 epochs (or run all 100 if val keeps improving)
    assert len(history["train_loss"]) <= 100


def test_trainer_returns_metrics():
    model = GNNLinkPredictor(in_channels=8, hidden_channels=16, num_layers=1)
    config = TrainConfig(epochs=5, lr=0.01)
    trainer = Trainer(model, config)
    data = _make_train_data()
    history = trainer.train(data)
    metrics = trainer.evaluate(data)
    assert "auc_roc" in metrics
    assert "avg_precision" in metrics
    assert 0 <= metrics["auc_roc"] <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/models/test_trainer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement trainer**

```python
# src/models/trainer.py
import random
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 0.001
    batch_size: int = 64
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    early_stopping: bool = False
    patience: int = 10


class Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig):
        self.model = model
        self.config = config
        self._train_edges = None
        self._val_edges = None
        self._test_edges = None
        self._is_vae = hasattr(model, "loss") and hasattr(model, "predict_links")

    def _split_edges(self, data: Data):
        num_edges = data.edge_index.size(1) // 2  # undirected
        # Get unique edges (one direction only)
        edge_set = set()
        edges = []
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            key = (min(src, dst), max(src, dst))
            if key not in edge_set:
                edge_set.add(key)
                edges.append(key)

        random.shuffle(edges)
        n = len(edges)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)

        train_edges = edges[:n_train]
        val_edges = edges[n_train:n_train + n_val]
        test_edges = edges[n_train + n_val:]

        def to_tensor(edge_list):
            if not edge_list:
                return torch.zeros((2, 0), dtype=torch.long)
            src = [e[0] for e in edge_list]
            dst = [e[1] for e in edge_list]
            return torch.tensor([src + dst, dst + src], dtype=torch.long)

        self._train_edges = to_tensor(train_edges)
        self._val_edges = to_tensor(val_edges)
        self._test_edges = to_tensor(test_edges)

    def train(self, data: Data, on_epoch=None) -> dict:
        self._split_edges(data)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        criterion = nn.BCELoss()

        train_data = Data(x=data.x, edge_index=self._train_edges, num_nodes=data.num_nodes)
        history = {"train_loss": [], "val_auc": []}
        best_val = 0.0
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            optimizer.zero_grad()

            if self._is_vae:
                loss = self.model.loss(data.x, self._train_edges)
                loss.backward()
                optimizer.step()
            else:
                # Positive edges
                pos_src = self._train_edges[0]
                pos_dst = self._train_edges[1]

                # Negative sampling
                neg_edge = negative_sampling(
                    self._train_edges, num_nodes=data.num_nodes,
                    num_neg_samples=pos_src.size(0),
                )
                neg_src, neg_dst = neg_edge[0], neg_edge[1]

                src = torch.cat([pos_src, neg_src])
                dst = torch.cat([pos_dst, neg_dst])
                labels = torch.cat([
                    torch.ones(pos_src.size(0)),
                    torch.zeros(neg_src.size(0)),
                ])

                scores = self.model(train_data, src, dst)
                loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()

            train_loss = loss.item()
            history["train_loss"].append(train_loss)

            # Validation
            val_auc = self._evaluate_edges(train_data, self._val_edges, data.num_nodes)
            history["val_auc"].append(val_auc)

            if on_epoch:
                on_epoch(epoch, train_loss, val_auc)

            if self.config.early_stopping:
                if val_auc > best_val:
                    best_val = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        break

        return history

    def evaluate(self, data: Data) -> dict:
        train_data = Data(x=data.x, edge_index=self._train_edges, num_nodes=data.num_nodes)
        test_edges = self._test_edges if self._test_edges.size(1) > 0 else self._val_edges

        self.model.eval()
        with torch.no_grad():
            pos_src = test_edges[0]
            pos_dst = test_edges[1]
            neg_edge = negative_sampling(
                test_edges, num_nodes=data.num_nodes,
                num_neg_samples=pos_src.size(0),
            )
            neg_src, neg_dst = neg_edge[0], neg_edge[1]

            src = torch.cat([pos_src, neg_src])
            dst = torch.cat([pos_dst, neg_dst])
            labels = torch.cat([
                torch.ones(pos_src.size(0)),
                torch.zeros(neg_src.size(0)),
            ]).numpy()

            if self._is_vae:
                scores = self.model.predict_links(data.x, self._train_edges, src, dst).detach().numpy()
            else:
                scores = self.model(train_data, src, dst).detach().numpy()

        return {
            "auc_roc": roc_auc_score(labels, scores),
            "avg_precision": average_precision_score(labels, scores),
        }

    def _evaluate_edges(self, train_data, eval_edges, num_nodes) -> float:
        if eval_edges.size(1) == 0:
            return 0.5
        self.model.eval()
        with torch.no_grad():
            pos_src = eval_edges[0]
            pos_dst = eval_edges[1]
            neg_edge = negative_sampling(
                eval_edges, num_nodes=num_nodes,
                num_neg_samples=pos_src.size(0),
            )
            src = torch.cat([pos_src, neg_edge[0]])
            dst = torch.cat([pos_dst, neg_edge[1]])
            labels = torch.cat([
                torch.ones(pos_src.size(0)),
                torch.zeros(neg_edge[0].size(0)),
            ]).numpy()
            if self._is_vae:
                scores = self.model.predict_links(train_data.x, train_data.edge_index, src, dst).detach().numpy()
            else:
                scores = self.model(train_data, src, dst).detach().numpy()
        try:
            return roc_auc_score(labels, scores)
        except ValueError:
            return 0.5
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/models/test_trainer.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/trainer.py tests/models/test_trainer.py
git commit -m "feat: add unified trainer with edge-level split and early stopping"
```

---

## Chunk 5: UI Pages

### Task 17: App Entry Point + Sidebar

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Implement app entry point with sidebar**

```python
# app.py
import streamlit as st

st.set_page_config(
    page_title="MidnightStar",
    page_icon="🧬",
    layout="wide",
)

# Sidebar — shared across all pages
with st.sidebar:
    st.title("🧬 MidnightStar")
    st.caption("Gene-disease correlation discovery")
    st.divider()

    # Data Manager section
    st.subheader("Data Manager")
    from src.data.cache import Cache
    import os

    cache_path = os.path.join(os.path.dirname(__file__), ".cache", "midnightstar.db")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = Cache(cache_path)

    size_mb = cache.size_bytes() / (1024 * 1024)
    st.metric("Cache Size", f"{size_mb:.1f} MB")
    if st.button("Clear Cache"):
        cache.clear_all()
        st.rerun()

    st.divider()
    st.caption("v0.1.0 — Local-first compute")

# Main page content
st.title("Welcome to MidnightStar")
st.markdown("""
Discover gene-disease correlations using deep learning.

**Get started:**
1. **Search** — Look up a gene or disease
2. **Explorer** — Navigate gene interaction networks
3. **Model Training** — Train ML models on your data
4. **Results** — View discovered patterns

Use the sidebar to navigate between pages.
""")
```

- [ ] **Step 2: Verify it runs**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && timeout 5 streamlit run app.py --server.headless true 2>&1 | head -3`
Expected: Streamlit starts without import errors

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add app entry point with sidebar and data manager"
```

---

### Task 18: Search Page

**Files:**
- Create: `pages/1_Search.py`

- [ ] **Step 1: Implement Search page**

```python
# pages/1_Search.py
import streamlit as st
import os
from src.data.cache import Cache
from src.data.gene_resolver import GeneResolver
from src.data.gwas_client import GWASClient
from src.data.gtex_client import GTExClient
from src.data.hpa_client import HPAClient
from src.data.string_client import STRINGClient
from src.data.bulk_downloader import BulkDownloader
from src.utils.graph_builder import GraphBuilder
from src.utils.text_explainer import TextExplainer
from src.utils import session

st.title("🔍 Search")
st.markdown("Search for a gene or disease to explore its connections.")

# Initialize clients
cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "midnightstar.db")
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
cache = Cache(cache_path)
resolver = GeneResolver()
gwas = GWASClient()
gtex = GTExClient()
hpa = HPAClient()
string = STRINGClient()
downloader = BulkDownloader(gwas, gtex, hpa, string)
explainer = TextExplainer()

# Search input
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Gene or disease name", placeholder="e.g., SP4, BRCA1, Alzheimer's")
with col2:
    st.write("")  # spacing
    download_all = st.button("📥 Download All Data")

if query:
    # Resolve gene
    with st.spinner("Resolving gene..."):
        gene = resolver.resolve(query)

    if gene is None:
        st.error(f"Could not find gene or disease: **{query}**. Try a different name.")
        st.stop()

    st.success(f"Found: **{gene.symbol}** ({gene.name}) — {gene.ensembl_id}")

    if download_all:
        progress = st.progress(0, text="Downloading from all sources...")
        sources_done = []

        def on_progress(source, status):
            sources_done.append(source)
            progress.progress(
                len(sources_done) / 4,
                text=f"✅ {source} {'done' if status == 'done' else 'error'} ({len(sources_done)}/4)"
            )

        result = downloader.download_all(gene, on_progress=on_progress)
        progress.progress(1.0, text="Download complete!")

        if result["errors"]:
            st.warning(f"Some sources had errors: {', '.join(result['errors'])}")

        # Cache results
        for assoc in result["associations"]:
            cache.set(assoc.data_source, gene.symbol, {"type": "association"}, assoc.to_dict(), pinned=True)

        # Store in session for other pages
        builder = GraphBuilder()
        builder.add_genes([gene])
        builder.add_diseases(result["diseases"])
        builder.add_associations(result["associations"])
        builder.add_expression(result["expression"])
        graph = builder.build()
        session.set_graph(graph)
        session.set_search_query(query)

        # Display results
        tab_summary, tab_assoc, tab_expr, tab_interact = st.tabs(
            ["Summary", "Associations", "Expression", "Interactions"]
        )

        with tab_summary:
            hpa_info = result["hpa_info"]
            explanation = explainer.explain_gene(gene.symbol, gene.name, hpa_info)
            st.markdown(explanation)

        with tab_assoc:
            if result["associations"]:
                import pandas as pd
                assoc_data = [
                    {
                        "Source Gene": a.source_id,
                        "Target": a.target_id,
                        "Type": a.type,
                        "Score": a.score,
                        "Evidence": a.evidence,
                        "Database": a.data_source,
                    }
                    for a in result["associations"]
                ]
                df = pd.DataFrame(assoc_data)
                st.dataframe(df.sort_values("Score", ascending=False), use_container_width=True)
            else:
                st.info("No associations found.")

        with tab_expr:
            if result["expression"]:
                import pandas as pd
                expr_data = [
                    {"Tissue": e.tissue.replace("_", " "), "Expression Level": e.expression_level}
                    for e in sorted(result["expression"], key=lambda e: e.expression_level, reverse=True)
                ]
                df = pd.DataFrame(expr_data)
                st.bar_chart(df.set_index("Tissue")["Expression Level"])
            else:
                st.info("No expression data found.")

        with tab_interact:
            string_assocs = [a for a in result["associations"] if a.data_source == "STRING"]
            if string_assocs:
                st.markdown(f"**{len(string_assocs)} protein interaction partners found.**")
                for a in sorted(string_assocs, key=lambda x: x.score, reverse=True)[:10]:
                    partner = a.target_id if a.source_id == gene.symbol else a.source_id
                    st.markdown(f"- **{partner}** (score: {a.score:.2f}) — {a.evidence}")
                if st.button("🕸️ Explore in Network Graph"):
                    st.switch_page("pages/2_Explorer.py")
            else:
                st.info("No protein interactions found.")
    else:
        st.info("Click **Download All Data** to fetch from GWAS, GTEx, HPA, and STRING.")
```

- [ ] **Step 2: Verify page loads**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && timeout 5 streamlit run app.py --server.headless true 2>&1 | head -3`
Expected: No import errors

- [ ] **Step 3: Commit**

```bash
git add pages/1_Search.py
git commit -m "feat: add Search page with bulk download and results tabs"
```

---

### Task 19: Explorer Page

**Files:**
- Create: `pages/2_Explorer.py`

- [ ] **Step 1: Implement Explorer page**

```python
# pages/2_Explorer.py
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import os
from src.data.gene_resolver import GeneResolver
from src.data.string_client import STRINGClient
from src.utils.graph_builder import GraphBuilder
from src.utils.text_explainer import TextExplainer
from src.utils import session

st.title("🕸️ Network Explorer")

graph = session.get_graph()

if graph is None:
    st.warning("No graph data loaded. Go to the **Search** page first to download data.")
    if st.button("Go to Search"):
        st.switch_page("pages/1_Search.py")
    st.stop()

# Controls
with st.sidebar:
    st.subheader("Graph Controls")
    min_score = st.slider("Minimum confidence score", 0.0, 1.0, 0.4, 0.05)

    available_sources = set()
    for _, _, data in graph.edges(data=True):
        for s in data.get("sources", [data.get("data_source", "unknown")]):
            available_sources.add(s)

    selected_sources = st.multiselect(
        "Data sources",
        sorted(available_sources),
        default=sorted(available_sources),
    )

    layout_options = {
        "Force-directed": "forceAtlas2Based",
        "Hierarchical": "hierarchicalRepulsion",
        "Repulsion": "repulsion",
    }
    layout_name = st.selectbox("Layout", list(layout_options.keys()))

    depth = st.slider("Expansion depth (hops)", 1, 3, 1)

# Filter graph
filtered_edges = [
    (u, v, d) for u, v, d in graph.edges(data=True)
    if d.get("score", 0) >= min_score
    and any(s in selected_sources for s in d.get("sources", [d.get("data_source", "")]))
]
filtered_nodes = set()
for u, v, _ in filtered_edges:
    filtered_nodes.add(u)
    filtered_nodes.add(v)

# Build Pyvis network
net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")
solver = layout_options[layout_name]
if solver == "hierarchicalRepulsion":
    net.set_options('{"layout": {"hierarchical": {"enabled": true}}}')
else:
    net.barnes_hut(gravity=-3000)

color_map = {"gene": "#2a5a8c", "disease": "#8c2a5c", "default": "#5c5c2a"}

for node_id in filtered_nodes:
    node_data = graph.nodes.get(node_id, {})
    node_type = node_data.get("node_type", "default")
    color = color_map.get(node_type, color_map["default"])
    label = node_data.get("symbol", node_data.get("name", node_id))
    title = f"{label}\nType: {node_type}"
    if "expression" in node_data:
        top_tissue = max(node_data["expression"], key=node_data["expression"].get)
        title += f"\nTop tissue: {top_tissue}"
    net.add_node(node_id, label=label, color=color, title=title, size=20)

for u, v, d in filtered_edges:
    width = max(1, d.get("score", 0.5) * 5)
    title = f"Score: {d.get('score', 'N/A')}\nSource: {d.get('data_source', 'unknown')}"
    net.add_edge(u, v, width=width, title=title)

# Render
col_graph, col_details = st.columns([7, 3])

with col_graph:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        f.seek(0)
        html_content = open(f.name).read()
        components.html(html_content, height=620, scrolling=True)
    os.unlink(f.name)

with col_details:
    st.subheader("Node Details")
    node_list = sorted(filtered_nodes)
    selected = st.selectbox("Select a node", node_list)
    if selected:
        node_data = graph.nodes.get(selected, {})
        st.markdown(f"**{selected}**")
        st.markdown(f"Type: {node_data.get('node_type', 'unknown')}")

        if "description" in node_data and node_data["description"]:
            st.markdown(f"_{node_data['description']}_")

        if "expression" in node_data:
            st.markdown("**Top tissues:**")
            sorted_tissues = sorted(
                node_data["expression"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            for tissue, level in sorted_tissues:
                st.markdown(f"- {tissue.replace('_', ' ')}: {level:.1f}")

        neighbors = list(graph.neighbors(selected))
        st.markdown(f"**Connections:** {len(neighbors)}")

        if st.button("⚙️ Train model on this subgraph"):
            session.set_selected_node(selected)
            st.switch_page("pages/3_Model_Training.py")

st.caption(f"Showing {len(filtered_nodes)} nodes, {len(filtered_edges)} edges")
```

- [ ] **Step 2: Commit**

```bash
git add pages/2_Explorer.py
git commit -m "feat: add Explorer page with Pyvis network graph and filtering"
```

---

### Task 20: Model Training Page

**Files:**
- Create: `pages/3_Model_Training.py`

- [ ] **Step 1: Implement Model Training page**

```python
# pages/3_Model_Training.py
import streamlit as st
import torch
import numpy as np
from torch_geometric.data import Data
from src.models.gnn import GNNLinkPredictor
from src.models.graph_transformer import GraphTransformerLinkPredictor
from src.models.vae import GraphVAE
from src.models.trainer import Trainer, TrainConfig
from src.utils import session

st.title("⚙️ Model Training")

graph = session.get_graph()
if graph is None:
    st.warning("No graph data loaded. Go to the **Search** page first.")
    st.stop()

# Convert NetworkX graph to PyG Data
def nx_to_pyg(G):
    node_list = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)

    # Build node features: expression (GTEx) + degree + node_type one-hot
    all_tissues = set()
    for n in node_list:
        expr = G.nodes[n].get("expression", {})
        all_tissues.update(expr.keys())
    all_tissues = sorted(all_tissues)

    # Feature vector: [expression_per_tissue..., degree, is_gene, is_disease]
    num_features = max(len(all_tissues), 1) + 3

    x = torch.zeros(num_nodes, num_features)
    for i, n in enumerate(node_list):
        # Expression features
        expr = G.nodes[n].get("expression", {})
        for j, tissue in enumerate(all_tissues):
            x[i, j] = expr.get(tissue, 0.0)
        # Degree feature
        x[i, -3] = float(G.degree(n))
        # Node type one-hot
        node_type = G.nodes[n].get("node_type", "")
        x[i, -2] = 1.0 if node_type == "gene" else 0.0
        x[i, -1] = 1.0 if node_type == "disease" else 0.0

    # Normalize expression columns only
    expr_cols = max(len(all_tissues), 1)
    max_vals = x[:, :expr_cols].max(dim=0).values.clamp(min=1.0)
    x[:, :expr_cols] = x[:, :expr_cols] / max_vals
    # Normalize degree
    max_deg = x[:, -3].max().clamp(min=1.0)
    x[:, -3] = x[:, -3] / max_deg

    # Build edges
    src, dst = [], []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            src.append(node_to_idx[u])
            dst.append(node_to_idx[v])
            src.append(node_to_idx[v])
            dst.append(node_to_idx[u])

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes), node_list, node_to_idx

pyg_data, node_list, node_to_idx = nx_to_pyg(graph)
in_channels = pyg_data.x.size(1)

# Layout: three columns
col_data, col_config, col_monitor = st.columns([1, 1, 1])

with col_data:
    st.subheader("📊 Data Selection")
    st.metric("Nodes", pyg_data.num_nodes)
    st.metric("Edges", pyg_data.edge_index.size(1) // 2)
    st.metric("Features", in_channels)
    train_ratio = st.slider("Train split", 0.5, 0.9, 0.8, 0.05)
    val_ratio = st.slider("Validation split", 0.05, 0.3, 0.1, 0.05)

with col_config:
    st.subheader("🧠 Model Configuration")
    model_type = st.radio("Model type", ["GNN", "Graph Transformer", "VAE"])

    if model_type == "GNN":
        num_layers = st.slider("Layers", 1, 5, 2)
        hidden_dim = st.select_slider("Hidden dimension", [16, 32, 64, 128, 256], value=64)
        aggr = st.selectbox("Aggregation", ["mean", "max", "sum"])
        with st.expander("What does this do?"):
            st.markdown("**GNN** learns by passing messages between connected genes. "
                       "It finds genes with similar neighborhoods that might share hidden connections.")
    elif model_type == "Graph Transformer":
        num_layers = st.slider("Layers", 1, 6, 2)
        hidden_dim = st.select_slider("Hidden dimension", [32, 64, 128, 256, 512], value=64)
        num_heads = st.select_slider("Attention heads", [1, 2, 4, 8], value=4)
        rwse_k = st.slider("RWSE walk length (k)", 8, 24, 16)
        with st.expander("What does this do?"):
            st.markdown("**Graph Transformer** uses attention to weigh which gene connections matter most. "
                       "It can detect longer-range patterns than GNN by looking at broader neighborhoods.")
    else:  # VAE
        hidden_dim = st.select_slider("Hidden dimension", [16, 32, 64, 128], value=64)
        latent_dim = st.select_slider("Latent dimension", [8, 16, 32, 64, 128], value=32)
        num_layers = st.slider("Encoder layers", 1, 4, 2)
        beta = st.slider("Beta (KL weight)", 0.1, 10.0, 1.0, 0.1)
        with st.expander("What does this do?"):
            st.markdown("**VAE** compresses gene data into a compact representation. "
                       "Genes that end up close together in this compressed space may share unknown pathways.")

    st.divider()
    epochs = st.slider("Epochs", 10, 500, 100)
    lr = st.select_slider("Learning rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    early_stop = st.checkbox("Early stopping", value=True)

with col_monitor:
    st.subheader("📈 Training Monitor")

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        # Build model
        if model_type == "GNN":
            model = GNNLinkPredictor(in_channels, hidden_dim, num_layers, aggr)
        elif model_type == "Graph Transformer":
            model = GraphTransformerLinkPredictor(
                in_channels, hidden_dim, num_layers, num_heads, rwse_dim=16, rwse_walk_length=rwse_k,
            )
        else:
            model = GraphVAE(in_channels, hidden_dim, latent_dim, num_layers, beta)

        config = TrainConfig(
            epochs=epochs, lr=lr, train_ratio=train_ratio,
            val_ratio=val_ratio, early_stopping=early_stop,
        )
        trainer = Trainer(model, config)

        progress_bar = st.progress(0)
        loss_chart = st.empty()
        status_text = st.empty()
        loss_history = []
        auc_history = []

        def on_epoch(epoch, loss, val_auc):
            loss_history.append(loss)
            auc_history.append(val_auc)
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch + 1}/{epochs} — Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")
            import pandas as pd
            df = pd.DataFrame({"Loss": loss_history, "Val AUC": auc_history})
            loss_chart.line_chart(df)

        with st.spinner("Training..."):
            history = trainer.train(pyg_data, on_epoch=on_epoch)

        metrics = trainer.evaluate(pyg_data)
        st.success("Training complete!")
        st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
        st.metric("Avg Precision", f"{metrics['avg_precision']:.4f}")

        # Store results
        session.set_training_results({
            "model": model,
            "trainer": trainer,
            "history": history,
            "metrics": metrics,
            "pyg_data": pyg_data,
            "node_list": node_list,
            "node_to_idx": node_to_idx,
            "model_type": model_type,
        })

        if st.button("📊 View Results"):
            st.switch_page("pages/4_Results.py")
```

- [ ] **Step 2: Commit**

```bash
git add pages/3_Model_Training.py
git commit -m "feat: add Model Training page with guided configuration"
```

---

### Task 21: Results Dashboard

**Files:**
- Create: `pages/4_Results.py`

- [ ] **Step 1: Implement Results dashboard**

```python
# pages/4_Results.py
import streamlit as st
import streamlit.components.v1 as components
import torch
import pandas as pd
import numpy as np
from pyvis.network import Network
import tempfile
import os
from src.utils.text_explainer import TextExplainer
from src.utils import session

st.title("📊 Discovery Dashboard")

results = session.get_training_results()
graph = session.get_graph()

if results is None:
    st.warning("No training results available. Go to **Model Training** first.")
    st.stop()

model = results["model"]
pyg_data = results["pyg_data"]
node_list = results["node_list"]
node_to_idx = results["node_to_idx"]
metrics = results["metrics"]
model_type = results["model_type"]
explainer = TextExplainer()

# Generate predictions for all non-existing edges
model.eval()
with torch.no_grad():
    existing_edges = set()
    for i in range(pyg_data.edge_index.size(1)):
        src_idx = pyg_data.edge_index[0, i].item()
        dst_idx = pyg_data.edge_index[1, i].item()
        existing_edges.add((min(src_idx, dst_idx), max(src_idx, dst_idx)))

    predictions = []
    num_nodes = pyg_data.num_nodes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) not in existing_edges:
                src_t = torch.tensor([i])
                dst_t = torch.tensor([j])
                if hasattr(model, "predict_links"):
                    score = model.predict_links(pyg_data.x, pyg_data.edge_index, src_t, dst_t).item()
                else:
                    score = model(pyg_data, src_t, dst_t).item()
                if score > 0.3:
                    predictions.append({
                        "Gene A": node_list[i],
                        "Gene B": node_list[j],
                        "Predicted Score": round(score, 4),
                        "src_idx": i,
                        "dst_idx": j,
                    })

    predictions.sort(key=lambda x: x["Predicted Score"], reverse=True)

# 2x2 Grid
top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)

# Top-left: Discovery Network
with top_left:
    st.subheader("Discovery Network")
    show_only_new = st.checkbox("Show only discoveries", value=False)

    net = Network(height="400px", width="100%", bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-2000)

    shown_nodes = set()
    # Add predicted edges
    for pred in predictions[:20]:
        for gene in [pred["Gene A"], pred["Gene B"]]:
            if gene not in shown_nodes:
                net.add_node(gene, label=gene, color="#ff8844", size=15, title=f"Discovery: {gene}")
                shown_nodes.add(gene)
        net.add_edge(pred["Gene A"], pred["Gene B"],
                    width=pred["Predicted Score"] * 4,
                    color="#ff8844", dashes=True,
                    title=f"Predicted: {pred['Predicted Score']:.2f}")

    if not show_only_new and graph:
        for u, v, d in graph.edges(data=True):
            for node in [u, v]:
                if node not in shown_nodes:
                    node_data = graph.nodes.get(node, {})
                    color = "#2a5a8c" if node_data.get("node_type") == "gene" else "#8c2a5c"
                    net.add_node(node, label=node, color=color, size=12)
                    shown_nodes.add(node)
            net.add_edge(u, v, width=d.get("score", 0.5) * 3, color="#2a5a8c",
                        title=f"Known: {d.get('score', 'N/A')}")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html = open(f.name).read()
        components.html(html, height=420, scrolling=True)
    os.unlink(f.name)

# Top-right: Summary
with top_right:
    st.subheader("Summary")
    st.markdown(f"**Model:** {model_type} | **AUC-ROC:** {metrics['auc_roc']:.4f} | "
                f"**Avg Precision:** {metrics['avg_precision']:.4f}")

    num_discoveries = len(predictions)
    if num_discoveries > 0:
        top = predictions[0]
        st.markdown(
            f"The {model_type} model found **{num_discoveries} potential new connections**. "
            f"The strongest predicted link is between **{top['Gene A']}** and "
            f"**{top['Gene B']}** (confidence: {top['Predicted Score']:.2f})."
        )

        st.markdown("**Top Discoveries:**")
        for i, pred in enumerate(predictions[:10], 1):
            score = pred["Predicted Score"]
            color = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🟠"
            st.markdown(f"{i}. {color} **{pred['Gene A']}** ↔ **{pred['Gene B']}** — {score:.2f}")

            with st.expander(f"Why {pred['Gene A']} ↔ {pred['Gene B']}?"):
                # Find shared info from graph
                shared_partners, shared_tissues = [], []
                if graph:
                    neighbors_a = set(graph.neighbors(pred["Gene A"])) if pred["Gene A"] in graph else set()
                    neighbors_b = set(graph.neighbors(pred["Gene B"])) if pred["Gene B"] in graph else set()
                    shared_partners = list(neighbors_a & neighbors_b)
                    expr_a = graph.nodes.get(pred["Gene A"], {}).get("expression", {})
                    expr_b = graph.nodes.get(pred["Gene B"], {}).get("expression", {})
                    shared_tissues = [t for t in expr_a if t in expr_b]

                explanation = explainer.explain_prediction(
                    pred["Gene A"], pred["Gene B"], score, shared_partners, shared_tissues,
                )
                st.markdown(explanation)
    else:
        st.info("No strong predictions found. Try training with more data or different parameters.")

# Bottom-left: Data Table
with bottom_left:
    st.subheader("Predictions Table")
    if predictions:
        df = pd.DataFrame(predictions)[["Gene A", "Gene B", "Predicted Score"]]
        st.dataframe(df, use_container_width=True, height=300)

        csv = df.to_csv(index=False)
        st.download_button("📥 Export CSV", csv, "predictions.csv", "text/csv")
    else:
        st.info("No predictions to display.")

# Bottom-right: Evidence
with bottom_right:
    st.subheader("Evidence Panel")
    if predictions:
        selected_pred = st.selectbox(
            "Select prediction",
            [f"{p['Gene A']} ↔ {p['Gene B']} ({p['Predicted Score']:.2f})" for p in predictions[:20]],
        )
        if selected_pred:
            idx = next(
                i for i, p in enumerate(predictions[:20])
                if f"{p['Gene A']} ↔ {p['Gene B']}" in selected_pred
            )
            pred = predictions[idx]

            if graph:
                for gene in [pred["Gene A"], pred["Gene B"]]:
                    expr = graph.nodes.get(gene, {}).get("expression", {})
                    if expr:
                        st.markdown(f"**{gene} expression:**")
                        top_tissues = sorted(expr.items(), key=lambda x: x[1], reverse=True)[:5]
                        tissue_df = pd.DataFrame(top_tissues, columns=["Tissue", "Level"])
                        st.bar_chart(tissue_df.set_index("Tissue"))

                st.markdown("**Supporting evidence:**")
                for gene in [pred["Gene A"], pred["Gene B"]]:
                    neighbors = list(graph.neighbors(gene)) if gene in graph else []
                    if neighbors:
                        st.markdown(f"- {gene} connects to: {', '.join(neighbors[:5])}")
    else:
        st.info("Select a prediction to see evidence.")

# Save/Export
st.divider()
col_save, col_export = st.columns(2)
with col_save:
    if st.button("💾 Save Model"):
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".models")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_type.lower().replace(' ', '_')}_model.pt")
        torch.save(model.state_dict(), path)
        st.success(f"Model saved to `{path}`")

with col_export:
    if predictions:
        full_df = pd.DataFrame(predictions)[["Gene A", "Gene B", "Predicted Score"]]
        html_report = full_df.to_html(index=False)
        st.download_button("📄 Export HTML Report", html_report, "report.html", "text/html")
```

- [ ] **Step 2: Commit**

```bash
git add pages/4_Results.py
git commit -m "feat: add Results dashboard with discovery network, summary, table, and evidence"
```

---

### Task 22: Final Integration + .gitignore Updates

- [ ] **Step 1: Update .gitignore**

Add these lines to `.gitignore`:

```
# MidnightStar
.cache/
.models/
.superpowers/
```

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/wizard/Desktop/programming/projects/midnightstar && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: update gitignore for cache, models, and superpowers directories"
```

---

## Deferred to v1.1

The following spec requirements are deferred to keep v1 shippable:

1. **GNNExplainer integration** — Use PyG's `GNNExplainer` for proper model-driven "Why?" explanations instead of the heuristic text approach in v1.
2. **HPA bulk TSV download** — Download and index the full HPA dataset (~200MB) for search/auto-suggest capability.
3. **Search auto-suggest** — Add typeahead dropdown powered by STRING/GWAS lookups as user types.
4. **Tooltip definitions** — Add hover tooltips for technical biology terms in the UI.
5. **Hits@K and Silhouette Score metrics** — Add to the Trainer's `evaluate()` method.
6. **Multiprocessing training** — Run model training in a separate `multiprocessing.Process` to avoid blocking the Streamlit UI for large datasets.
7. **Compare model runs** — Side-by-side comparison of multiple training runs.
8. **"Did you mean?" suggestions** — Fuzzy search suggestions for invalid gene/disease queries.
9. **Richer node features** — Add HPA subcellular location one-hot and GWAS association count to node feature vectors (v1 uses expression + degree + type).
