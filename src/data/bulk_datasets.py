"""Download and index full bulk datasets from all 4 sources."""
import csv
import gzip
import io
import logging
import os
import sqlite3
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DATA_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".datasets")

URLS = {
    "gwas": "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/gwas-catalog-associations_ontology-annotated-full.zip",
    "gtex": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz",
    "hpa": "https://www.proteinatlas.org/download/proteinatlas.tsv.zip",
    "string": "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/9606.protein.links.detailed.v12.0.txt.gz",
}


class BulkDatasetManager:
    def __init__(self, data_dir: str = DATA_DIR_DEFAULT):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "datasets.db")
        os.makedirs(data_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gwas (
                    snp TEXT,
                    mapped_gene TEXT,
                    disease_trait TEXT,
                    pvalue REAL,
                    pvalue_mlog REAL,
                    risk_allele_freq TEXT,
                    reported_genes TEXT,
                    study TEXT,
                    pubmedid TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gwas_gene ON gwas(mapped_gene)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gwas_trait ON gwas(disease_trait)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS gtex (
                    ensembl_id TEXT,
                    gene_symbol TEXT,
                    tissue TEXT,
                    median_tpm REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gtex_gene ON gtex(gene_symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gtex_ensembl ON gtex(ensembl_id)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS hpa (
                    gene TEXT,
                    ensembl TEXT,
                    gene_description TEXT,
                    subcellular_location TEXT,
                    rna_tissue_specificity TEXT,
                    tissue_expression_cluster TEXT,
                    protein_class TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hpa_gene ON hpa(gene)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hpa_ensembl ON hpa(ensembl)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS string (
                    protein1 TEXT,
                    protein2 TEXT,
                    neighborhood INTEGER,
                    fusion INTEGER,
                    cooccurence INTEGER,
                    coexpression INTEGER,
                    experimental INTEGER,
                    database_score INTEGER,
                    textmining INTEGER,
                    combined_score INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_string_p1 ON string(protein1)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_string_p2 ON string(protein2)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS download_status (
                    source TEXT PRIMARY KEY,
                    status TEXT,
                    downloaded_at TEXT,
                    row_count INTEGER
                )
            """)

    def get_status(self) -> dict[str, dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT source, status, downloaded_at, row_count FROM download_status").fetchall()
        return {r[0]: {"status": r[1], "downloaded_at": r[2], "row_count": r[3]} for r in rows}

    def is_downloaded(self, source: str) -> bool:
        status = self.get_status()
        return source in status and status[source]["status"] == "complete"

    def download_gwas(self, on_progress=None) -> int:
        if on_progress:
            on_progress("Downloading GWAS Catalog (~58 MB)...")
        raw = self._download_file(URLS["gwas"], on_progress)
        if on_progress:
            on_progress("Extracting and indexing GWAS data...")

        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            tsv_name = [n for n in zf.namelist() if n.endswith(".tsv")][0]
            with zf.open(tsv_name) as f:
                text = io.TextIOWrapper(f, encoding="utf-8")
                reader = csv.DictReader(text, delimiter="\t")
                rows = []
                for row in reader:
                    pvalue_str = row.get("P-VALUE", "")
                    pvalue_mlog_str = row.get("PVALUE_MLOG", "")
                    try:
                        pvalue = float(pvalue_str) if pvalue_str else None
                    except ValueError:
                        pvalue = None
                    try:
                        pvalue_mlog = float(pvalue_mlog_str) if pvalue_mlog_str else None
                    except ValueError:
                        pvalue_mlog = None

                    rows.append((
                        row.get("SNPS", ""),
                        row.get("MAPPED_GENE", ""),
                        row.get("DISEASE/TRAIT", ""),
                        pvalue,
                        pvalue_mlog,
                        row.get("RISK ALLELE FREQUENCY", ""),
                        row.get("REPORTED GENE(S)", ""),
                        row.get("STUDY", ""),
                        row.get("PUBMEDID", ""),
                    ))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM gwas")
            conn.executemany(
                "INSERT INTO gwas VALUES (?,?,?,?,?,?,?,?,?)", rows
            )
        self._set_status("gwas", "complete", len(rows))
        return len(rows)

    def download_gtex(self, on_progress=None) -> int:
        if on_progress:
            on_progress("Downloading GTEx expression data (~7 MB)...")
        raw = self._download_file(URLS["gtex"], on_progress)
        if on_progress:
            on_progress("Extracting and indexing GTEx data...")

        text = gzip.decompress(raw).decode("utf-8")
        lines = text.strip().split("\n")
        # GCT format: line 0 = #1.2, line 1 = dimensions, line 2 = header, line 3+ = data
        header = lines[2].split("\t")
        tissues = header[2:]  # first two cols are Name, Description

        rows = []
        for line in lines[3:]:
            parts = line.split("\t")
            ensembl_id = parts[0].split(".")[0]  # strip version
            gene_symbol = parts[1]
            for i, tissue in enumerate(tissues):
                tpm = float(parts[i + 2]) if parts[i + 2] else 0.0
                if tpm > 0:
                    rows.append((ensembl_id, gene_symbol, tissue, tpm))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM gtex")
            conn.executemany(
                "INSERT INTO gtex VALUES (?,?,?,?)", rows
            )
        self._set_status("gtex", "complete", len(rows))
        return len(rows)

    def download_hpa(self, on_progress=None) -> int:
        if on_progress:
            on_progress("Downloading Human Protein Atlas (~6 MB)...")
        raw = self._download_file(URLS["hpa"], on_progress)
        if on_progress:
            on_progress("Extracting and indexing HPA data...")

        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            tsv_name = [n for n in zf.namelist() if n.endswith(".tsv")][0]
            with zf.open(tsv_name) as f:
                text = io.TextIOWrapper(f, encoding="utf-8")
                reader = csv.DictReader(text, delimiter="\t")
                rows = []
                for row in reader:
                    rows.append((
                        row.get("Gene", ""),
                        row.get("Ensembl", ""),
                        row.get("Gene description", ""),
                        row.get("Subcellular location", ""),
                        row.get("RNA tissue specificity", ""),
                        row.get("Tissue expression cluster", ""),
                        row.get("Protein class", ""),
                    ))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM hpa")
            conn.executemany(
                "INSERT INTO hpa VALUES (?,?,?,?,?,?,?)", rows
            )
        self._set_status("hpa", "complete", len(rows))
        return len(rows)

    def download_string(self, on_progress=None) -> int:
        if on_progress:
            on_progress("Downloading STRING interactions (~133 MB)...")
        raw = self._download_file(URLS["string"], on_progress)
        if on_progress:
            on_progress("Extracting and indexing STRING data...")

        text = gzip.decompress(raw).decode("utf-8")
        lines = text.strip().split("\n")

        rows = []
        for line in lines[1:]:  # skip header
            parts = line.split(" ")
            if len(parts) < 10:
                continue
            rows.append((
                parts[0],  # protein1
                parts[1],  # protein2
                int(parts[2]),  # neighborhood
                int(parts[3]),  # fusion
                int(parts[4]),  # cooccurence
                int(parts[5]),  # coexpression
                int(parts[6]),  # experimental
                int(parts[7]),  # database
                int(parts[8]),  # textmining
                int(parts[9]),  # combined_score
            ))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM string")
            conn.executemany(
                "INSERT INTO string VALUES (?,?,?,?,?,?,?,?,?,?)", rows
            )
        self._set_status("string", "complete", len(rows))
        return len(rows)

    def download_all(self, on_progress=None) -> dict[str, int]:
        results = {}
        for source, method in [
            ("gwas", self.download_gwas),
            ("gtex", self.download_gtex),
            ("hpa", self.download_hpa),
            ("string", self.download_string),
        ]:
            try:
                count = method(on_progress=on_progress)
                results[source] = count
            except Exception as e:
                logger.error("Failed to download %s: %s", source, e)
                results[source] = -1
                self._set_status(source, f"error: {e}", 0)
        return results

    def query_gene(self, gene_symbol: str) -> dict:
        """Query all downloaded datasets for a gene."""
        result = {"gwas": [], "gtex": [], "hpa": None, "string": []}
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # GWAS
            rows = conn.execute(
                "SELECT * FROM gwas WHERE mapped_gene LIKE ? OR reported_genes LIKE ?",
                (f"%{gene_symbol}%", f"%{gene_symbol}%")
            ).fetchall()
            result["gwas"] = [dict(r) for r in rows]

            # GTEx
            rows = conn.execute(
                "SELECT tissue, median_tpm FROM gtex WHERE gene_symbol = ?",
                (gene_symbol,)
            ).fetchall()
            result["gtex"] = [dict(r) for r in rows]

            # HPA
            row = conn.execute(
                "SELECT * FROM hpa WHERE gene = ?", (gene_symbol,)
            ).fetchone()
            result["hpa"] = dict(row) if row else None

            # STRING — map protein IDs back
            rows = conn.execute(
                """SELECT * FROM string
                   WHERE protein1 LIKE ? OR protein2 LIKE ?""",
                (f"%{gene_symbol}%", f"%{gene_symbol}%")
            ).fetchall()
            result["string"] = [dict(r) for r in rows]

        return result

    def _download_file(self, url: str, on_progress=None) -> bytes:
        resp = requests.get(url, timeout=600, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        chunks = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            chunks.append(chunk)
            downloaded += len(chunk)
            if on_progress and total:
                mb_done = downloaded / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                on_progress(f"Downloaded {mb_done:.1f} / {mb_total:.1f} MB")
        return b"".join(chunks)

    def _set_status(self, source: str, status: str, row_count: int):
        import datetime
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO download_status (source, status, downloaded_at, row_count)
                   VALUES (?, ?, ?, ?)""",
                (source, status, datetime.datetime.now().isoformat(), row_count),
            )

    def db_size_mb(self) -> float:
        if not os.path.exists(self.db_path):
            return 0.0
        return os.path.getsize(self.db_path) / (1024 * 1024)
