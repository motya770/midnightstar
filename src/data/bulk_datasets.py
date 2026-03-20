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
                    pubmedid TEXT,
                    mapped_trait TEXT,
                    mapped_trait_uri TEXT
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
                CREATE TABLE IF NOT EXISTS alphafold (
                    uniprot_id TEXT PRIMARY KEY,
                    gene TEXT,
                    mean_plddt REAL,
                    frac_very_high REAL,
                    frac_confident REAL,
                    frac_low REAL,
                    frac_very_low REAL,
                    sequence_length INTEGER,
                    disordered_fraction REAL,
                    pdb_url TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_af_gene ON alphafold(gene)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS download_status (
                    source TEXT PRIMARY KEY,
                    status TEXT,
                    downloaded_at TEXT,
                    row_count INTEGER
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    graph_name TEXT,
                    nodes INTEGER,
                    edges INTEGER,
                    features INTEGER,
                    epochs_run INTEGER,
                    auc_roc REAL,
                    avg_precision REAL,
                    model_path TEXT
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
                        row.get("MAPPED_TRAIT", ""),
                        row.get("MAPPED_TRAIT_URI", ""),
                    ))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM gwas")
            conn.executemany(
                "INSERT INTO gwas VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
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

    def download_alphafold(self, on_progress=None) -> int:
        """Download AlphaFold structural features for all human genes.

        Strategy:
        1. Download UniProt human ID mapping file (gene name → UniProt ID)
        2. Query AlphaFold prediction API per UniProt ID
        """
        if on_progress:
            on_progress("Step 1/3: Downloading UniProt ID mapping for human proteome...")

        # Download the UniProt human gene-to-accession mapping via the tab file
        # This is a small file (~2MB) that maps gene names to UniProt accessions
        try:
            resp = requests.get(
                "https://rest.uniprot.org/uniprotkb/stream",
                params={
                    "query": "organism_id:9606 AND reviewed:true",
                    "format": "tsv",
                    "fields": "accession,gene_primary",
                },
                timeout=120,
            )
            resp.raise_for_status()
        except Exception as e:
            if on_progress:
                on_progress(f"Failed to download UniProt mapping: {e}")
            return 0

        # Parse TSV: columns are "Entry" and "Gene Names (primary)"
        lines = resp.text.strip().split("\n")
        gene_to_uniprot = {}
        for line in lines[1:]:  # skip header
            parts = line.split("\t")
            if len(parts) >= 2:
                uniprot_id = parts[0].strip()
                gene_name = parts[1].strip()
                if gene_name and uniprot_id:
                    gene_to_uniprot[gene_name] = uniprot_id

        if on_progress:
            on_progress(f"Mapped {len(gene_to_uniprot)} human genes to UniProt IDs")

        # Get our HPA gene list
        with sqlite3.connect(self.db_path) as conn:
            hpa_genes = conn.execute("SELECT DISTINCT gene FROM hpa WHERE gene != ''").fetchall()
        hpa_gene_set = {r[0] for r in hpa_genes}

        # Find overlap
        genes_with_uniprot = []
        for gene in hpa_gene_set:
            uid = gene_to_uniprot.get(gene)
            if uid:
                genes_with_uniprot.append((gene, uid))

        if on_progress:
            on_progress(f"Found UniProt IDs for {len(genes_with_uniprot)} / {len(hpa_gene_set)} HPA genes")

        # Step 2: Check what's already downloaded (resume support)
        already_done = set()
        with sqlite3.connect(self.db_path) as conn:
            tables = [t[0] for t in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "alphafold" in tables:
                existing = conn.execute("SELECT gene FROM alphafold").fetchall()
                already_done = {r[0] for r in existing}

        # Filter out already-downloaded genes
        todo = [(g, u) for g, u in genes_with_uniprot if g not in already_done]

        if on_progress:
            on_progress(f"Step 2/3: Fetching AlphaFold predictions... "
                        f"({len(already_done)} already done, {len(todo)} remaining)")

        if not todo:
            total_count = len(already_done)
            self._set_status("alphafold", "complete", total_count)
            if on_progress:
                on_progress(f"AlphaFold already complete: {total_count} genes")
            return total_count

        # Fetch in batches, saving immediately to DB after each batch
        batch = []
        batch_size = 100
        fetched = 0
        failed = 0
        total = len(todo)

        for i, (gene_name, uniprot_id) in enumerate(todo):
            if on_progress and i % 100 == 0:
                on_progress(f"AlphaFold: {i}/{total} remaining "
                            f"({fetched + len(already_done)} total, {failed} failed)")

            try:
                resp = requests.get(
                    f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}",
                    timeout=15,
                )
                if resp.status_code != 200:
                    failed += 1
                    continue
                data = resp.json()
                pred = data[0] if isinstance(data, list) and data else data
                if not pred:
                    failed += 1
                    continue

                batch.append((
                    uniprot_id,
                    gene_name,
                    pred.get("globalMetricValue", 0.0),
                    pred.get("fractionPlddtVeryHigh", 0.0),
                    pred.get("fractionPlddtConfident", 0.0),
                    pred.get("fractionPlddtLow", 0.0),
                    pred.get("fractionPlddtVeryLow", 0.0),
                    (pred.get("sequenceEnd", 0) or 0) - (pred.get("sequenceStart", 0) or 0) + 1,
                    pred.get("fractionPlddtVeryLow", 0.0) + pred.get("fractionPlddtLow", 0.0),
                    pred.get("pdbUrl", ""),
                ))
                fetched += 1
            except Exception:
                failed += 1

            # Flush batch to DB immediately
            if len(batch) >= batch_size:
                with sqlite3.connect(self.db_path) as conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO alphafold VALUES (?,?,?,?,?,?,?,?,?,?)",
                        batch
                    )
                batch = []
                # Update status so progress survives crashes
                total_so_far = fetched + len(already_done)
                self._set_status("alphafold", f"in_progress ({total_so_far})", total_so_far)

        # Flush remaining
        if batch:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO alphafold VALUES (?,?,?,?,?,?,?,?,?,?)",
                    batch
                )

        total_count = fetched + len(already_done)
        self._set_status("alphafold", "complete", total_count)
        if on_progress:
            on_progress(f"AlphaFold done: {total_count} genes indexed ({fetched} new, {failed} failed)")
        return total_count

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
        result = {"gwas": [], "gtex": [], "hpa": None, "string": [], "alphafold": None}
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

            # STRING — resolve gene symbol to protein ID via aliases
            protein_ids = []
            tables = [t[0] for t in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "string_aliases" in tables:
                alias_rows = conn.execute(
                    "SELECT DISTINCT protein_id FROM string_aliases WHERE gene_symbol = ?",
                    (gene_symbol,)
                ).fetchall()
                protein_ids = [r[0] for r in alias_rows]

            if protein_ids:
                placeholders = ",".join(["?"] * len(protein_ids))
                rows = conn.execute(
                    f"""SELECT * FROM string
                        WHERE protein1 IN ({placeholders}) OR protein2 IN ({placeholders})""",
                    protein_ids + protein_ids
                ).fetchall()
                result["string"] = [dict(r) for r in rows]
            else:
                # Fallback: LIKE search (slow but works without aliases)
                rows = conn.execute(
                    "SELECT * FROM string WHERE protein1 LIKE ? OR protein2 LIKE ?",
                    (f"%{gene_symbol}%", f"%{gene_symbol}%")
                ).fetchall()
                result["string"] = [dict(r) for r in rows]

            # AlphaFold
            if "alphafold" in tables:
                af_row = conn.execute(
                    "SELECT * FROM alphafold WHERE gene = ?", (gene_symbol,)
                ).fetchone()
                result["alphafold"] = dict(af_row) if af_row else None

        return result

    def get_all_gene_symbols(self) -> list[str]:
        """Return all gene symbols present in the HPA table."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT DISTINCT gene FROM hpa WHERE gene != '' ORDER BY gene").fetchall()
        return [r[0] for r in rows]

    def get_string_protein_map(self) -> dict[str, str]:
        """Build a mapping from STRING protein ID (9606.ENSPXXX) to gene symbol via HPA ensembl IDs."""
        mapping = {}
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT gene, ensembl FROM hpa WHERE ensembl != ''").fetchall()
        for gene, ensembl in rows:
            # STRING uses 9606.ENSPXXX but we have ENSG IDs — build what we can
            mapping[ensembl] = gene
        return mapping

    def build_graph(self, gene_symbol: str, depth: int = 1, min_score: int = 400):
        """Build a NetworkX graph centered on a gene using local bulk data.

        Args:
            gene_symbol: Starting gene symbol
            depth: How many hops of STRING interactions to include (1-3)
            min_score: Minimum STRING combined_score (0-1000)
        """
        import networkx as nx

        G = nx.Graph()
        visited_genes = set()
        genes_to_expand = {gene_symbol}

        for hop in range(depth):
            next_genes = set()
            for gene in genes_to_expand:
                if gene in visited_genes:
                    continue
                visited_genes.add(gene)

                data = self.query_gene(gene)

                # Add gene node
                hpa = data["hpa"]
                G.add_node(gene, node_type="gene", symbol=gene,
                           name=hpa.get("gene_description", gene) if hpa else gene,
                           description=hpa.get("gene_description", "") if hpa else "")

                # Add expression data
                if data["gtex"]:
                    expr = {r["tissue"]: r["median_tpm"] for r in data["gtex"]}
                    G.nodes[gene]["expression"] = expr

                # Add HPA info
                if hpa:
                    G.nodes[gene]["subcellular_location"] = hpa.get("subcellular_location", "")
                    G.nodes[gene]["tissue_specificity"] = hpa.get("rna_tissue_specificity", "")

                # Add GWAS disease associations
                seen_diseases = set()
                for assoc in data["gwas"]:
                    disease = assoc.get("disease_trait", "")
                    if not disease or disease in seen_diseases:
                        continue
                    seen_diseases.add(disease)
                    disease_id = disease.replace(" ", "_")[:50]
                    if disease_id not in G.nodes:
                        G.add_node(disease_id, node_type="disease", name=disease,
                                   description="", id=disease_id)
                    pvalue = assoc.get("pvalue")
                    score = 0.5
                    if pvalue and pvalue > 0:
                        import math
                        score = min(1.0, -math.log10(pvalue) / 50)
                    if not G.has_edge(gene, disease_id):
                        G.add_edge(gene, disease_id, score=round(score, 4),
                                   data_source="GWAS", sources=["GWAS"],
                                   type="gene-disease",
                                   evidence=f"p-value: {pvalue:.2e}" if pvalue else "GWAS")

                # Add STRING interactions
                for interaction in data["string"]:
                    combined = interaction.get("combined_score", 0)
                    if combined < min_score:
                        continue
                    p1 = interaction["protein1"]
                    p2 = interaction["protein2"]
                    # Resolve protein IDs to gene symbols via HPA
                    partner_protein = p2 if gene.upper() in p1.upper() else p1
                    partner_gene = self._resolve_protein_to_gene(partner_protein)
                    if not partner_gene or partner_gene == gene:
                        continue

                    if partner_gene not in G.nodes:
                        G.add_node(partner_gene, node_type="gene", symbol=partner_gene,
                                   name=partner_gene, description="")
                    score = combined / 1000.0
                    evidence_parts = []
                    if interaction.get("experimental", 0) > 0:
                        evidence_parts.append("experiments")
                    if interaction.get("database_score", 0) > 0:
                        evidence_parts.append("databases")
                    if interaction.get("textmining", 0) > 0:
                        evidence_parts.append("textmining")
                    if interaction.get("coexpression", 0) > 0:
                        evidence_parts.append("co-expression")

                    if not G.has_edge(gene, partner_gene):
                        G.add_edge(gene, partner_gene,
                                   score=round(score, 4), data_source="STRING",
                                   sources=["STRING"], type="protein-protein",
                                   evidence=", ".join(evidence_parts) or "combined")
                    next_genes.add(partner_gene)

            genes_to_expand = next_genes - visited_genes

        return G

    def build_full_graph(self, min_string_score: int = 700, include_diseases: bool = True,
                         max_disease_pvalue: float = 5e-8, on_progress=None):
        """Build a graph from ALL downloaded data across all sources.

        Args:
            min_string_score: Minimum STRING combined_score (0-1000). Higher = fewer but stronger edges.
                              700 = high confidence (~2-3M edges), 900 = highest (~500K edges).
            include_diseases: Whether to add GWAS disease nodes and gene-disease edges.
            max_disease_pvalue: Only include GWAS associations with p-value below this threshold.
            on_progress: Callback for progress updates.
        """
        import math
        import networkx as nx

        G = nx.Graph()

        # Step 1: Add all genes from HPA as nodes
        if on_progress:
            on_progress("Loading gene nodes from HPA...")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            hpa_rows = conn.execute(
                "SELECT gene, ensembl, gene_description, subcellular_location, "
                "rna_tissue_specificity, tissue_expression_cluster FROM hpa"
            ).fetchall()

        gene_set = set()
        for row in hpa_rows:
            gene = row["gene"]
            if not gene:
                continue
            gene_set.add(gene)
            G.add_node(gene, node_type="gene", symbol=gene,
                       name=row["gene_description"] or gene,
                       description=row["gene_description"] or "",
                       subcellular_location=row["subcellular_location"] or "",
                       tissue_specificity=row["rna_tissue_specificity"] or "")

        if on_progress:
            on_progress(f"Added {len(gene_set)} gene nodes")

        # Build Ensembl ID -> HPA symbol mapping for cross-referencing
        ensembl_to_hpa = {}
        for row in hpa_rows:
            gene = row["gene"]
            ensembl = row["ensembl"]
            if gene and ensembl:
                ensembl_to_hpa[ensembl] = gene

        # Step 2: Add GTEx expression as node features
        if on_progress:
            on_progress("Loading GTEx expression data...")
        with sqlite3.connect(self.db_path) as conn:
            gtex_rows = conn.execute(
                "SELECT ensembl_id, gene_symbol, tissue, median_tpm FROM gtex WHERE median_tpm > 0"
            ).fetchall()

        expr_data = {}
        for ensembl_id, gene, tissue, tpm in gtex_rows:
            # Resolve to HPA symbol: try direct match first, then Ensembl ID lookup
            resolved = gene if gene in gene_set else ensembl_to_hpa.get(ensembl_id)
            if not resolved:
                continue
            if resolved not in expr_data:
                expr_data[resolved] = {}
            expr_data[resolved][tissue] = tpm

        for gene, tissues in expr_data.items():
            if gene in G.nodes:
                G.nodes[gene]["expression"] = tissues

        if on_progress:
            on_progress(f"Added expression for {len(expr_data)} genes across {len(set(t for ts in expr_data.values() for t in ts))} tissues")

        # Step 2b: Add AlphaFold structural features
        with sqlite3.connect(self.db_path) as conn:
            tables = [t[0] for t in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "alphafold" in tables:
                if on_progress:
                    on_progress("Loading AlphaFold structural features...")
                af_rows = conn.execute(
                    "SELECT gene, mean_plddt, frac_very_high, frac_very_low, "
                    "sequence_length, disordered_fraction FROM alphafold"
                ).fetchall()
                af_count = 0
                for gene, plddt, fvh, fvl, seqlen, disorder in af_rows:
                    if gene in G.nodes:
                        G.nodes[gene]["mean_plddt"] = plddt
                        G.nodes[gene]["frac_plddt_high"] = fvh
                        G.nodes[gene]["frac_plddt_low"] = fvl
                        G.nodes[gene]["sequence_length"] = seqlen
                        G.nodes[gene]["disordered_fraction"] = disorder
                        af_count += 1
                if on_progress:
                    on_progress(f"Added AlphaFold features for {af_count} genes")

        # Step 3: Build protein-to-gene mapping
        if on_progress:
            on_progress("Building protein-to-gene mapping...")
        protein_to_gene = {}
        with sqlite3.connect(self.db_path) as conn:
            tables = [t[0] for t in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "string_aliases" in tables:
                alias_rows = conn.execute(
                    "SELECT protein_id, gene_symbol FROM string_aliases"
                ).fetchall()
                for pid, gene in alias_rows:
                    protein_to_gene[pid] = gene

        if on_progress:
            on_progress(f"Mapped {len(protein_to_gene)} protein IDs to gene symbols")

        # Step 4: Add STRING interactions as edges
        if on_progress:
            on_progress(f"Loading STRING interactions (score >= {min_string_score})...")
        with sqlite3.connect(self.db_path) as conn:
            string_rows = conn.execute(
                "SELECT protein1, protein2, combined_score, experimental, "
                "database_score, textmining, coexpression "
                "FROM string WHERE combined_score >= ?",
                (min_string_score,)
            ).fetchall()

        edge_count = 0
        for p1, p2, score, exp, db, tm, coex in string_rows:
            gene1 = protein_to_gene.get(p1)
            gene2 = protein_to_gene.get(p2)
            if not gene1 or not gene2 or gene1 == gene2:
                continue

            # Add nodes if not already present (some genes may not be in HPA)
            for g in (gene1, gene2):
                if g not in G.nodes:
                    G.add_node(g, node_type="gene", symbol=g, name=g, description="")

            if not G.has_edge(gene1, gene2):
                evidence = []
                if exp > 0: evidence.append("experiments")
                if db > 0: evidence.append("databases")
                if tm > 0: evidence.append("textmining")
                if coex > 0: evidence.append("co-expression")

                G.add_edge(gene1, gene2,
                           score=round(score / 1000.0, 4),
                           data_source="STRING", sources=["STRING"],
                           type="protein-protein",
                           evidence=", ".join(evidence) or "combined")
                edge_count += 1

        if on_progress:
            on_progress(f"Added {edge_count} STRING edges")

        # Step 5: Add GWAS associations as gene node features
        if include_diseases:
            if on_progress:
                on_progress("Loading GWAS associations as gene features...")
            with sqlite3.connect(self.db_path) as conn:
                gwas_rows = conn.execute(
                    "SELECT mapped_gene, disease_trait, pvalue, mapped_trait, mapped_trait_uri "
                    "FROM gwas WHERE pvalue IS NOT NULL AND pvalue <= ? AND mapped_gene != ''",
                    (max_disease_pvalue,)
                ).fetchall()

            # Classify traits by ontology URI prefix
            def _classify_trait(uri):
                if not uri:
                    return "other"
                for u in uri.split(", "):
                    tag = u.rsplit("/", 1)[-1] if "/" in u else ""
                    if tag.startswith("MONDO") or tag.startswith("Orphanet"):
                        return "disease"
                    if tag.startswith("HP"):
                        return "phenotype"
                    if tag.startswith("OBA"):
                        return "measurement"
                    if tag.startswith("GO"):
                        return "biological_process"
                return "trait"  # EFO and others

            # Collect per-gene GWAS associations grouped by category
            gwas_by_gene = {}
            for mapped_gene, disease_trait, pvalue, mapped_trait, uri in gwas_rows:
                category = _classify_trait(uri)
                normalized = mapped_gene.replace(" - ", ",").replace(";", ",")
                genes = {g.strip() for g in normalized.split(",") if g.strip()}
                trait_name = mapped_trait or disease_trait
                score = min(1.0, -math.log10(max(pvalue, 1e-300)) / 50)

                for gene in genes:
                    if gene not in G.nodes:
                        continue
                    if gene not in gwas_by_gene:
                        gwas_by_gene[gene] = {}
                    key = (category, trait_name)
                    # Keep strongest association (lowest p-value = highest score)
                    if key not in gwas_by_gene[gene] or score > gwas_by_gene[gene][key]:
                        gwas_by_gene[gene][key] = score

            # Apply to graph nodes
            gwas_gene_count = 0
            for gene, associations in gwas_by_gene.items():
                gwas_features = {}
                for (category, trait_name), score in associations.items():
                    if category not in gwas_features:
                        gwas_features[category] = []
                    gwas_features[category].append({"trait": trait_name, "score": round(score, 4)})
                G.nodes[gene]["gwas"] = gwas_features
                gwas_gene_count += 1

            trait_count = sum(len(v) for assoc in gwas_by_gene.values() for v in assoc)
            if on_progress:
                on_progress(f"Added GWAS features to {gwas_gene_count} genes ({trait_count} trait associations)")

        if on_progress:
            on_progress(f"Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        return G

    def _resolve_protein_to_gene(self, protein_id: str) -> str | None:
        """Resolve a STRING protein ID (9606.ENSPXXX) to a gene symbol."""
        if not hasattr(self, "_protein_cache"):
            self._protein_cache = {}
            with sqlite3.connect(self.db_path) as conn:
                tables = [t[0] for t in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]
                if "string_aliases" in tables:
                    rows = conn.execute(
                        "SELECT protein_id, gene_symbol FROM string_aliases"
                    ).fetchall()
                    for pid, gene in rows:
                        self._protein_cache[pid] = gene

        return self._protein_cache.get(protein_id)

    def build_string_alias_table(self, on_progress=None):
        """Download and index STRING aliases to map protein IDs to gene symbols."""
        url = "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz"
        if on_progress:
            on_progress("Downloading STRING aliases (~25 MB)...")
        raw = self._download_file(url, on_progress)
        if on_progress:
            on_progress("Indexing STRING aliases...")

        text = gzip.decompress(raw).decode("utf-8")
        lines = text.strip().split("\n")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS string_aliases (
                    protein_id TEXT,
                    gene_symbol TEXT,
                    source TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alias_protein ON string_aliases(protein_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alias_gene ON string_aliases(gene_symbol)")
            conn.execute("DELETE FROM string_aliases")

            rows = []
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                protein_id = parts[0]
                alias = parts[1]
                source = parts[2]
                # Keep only canonical gene symbol mappings
                if source == "Ensembl_HGNC":
                    rows.append((protein_id, alias, source))

            conn.executemany("INSERT INTO string_aliases VALUES (?,?,?)", rows)

        self._set_status("string_aliases", "complete", len(rows))
        return len(rows)

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

    def save_training_run(self, model_type: str, parameters: str, graph_name: str,
                          nodes: int, edges: int, features: int, epochs_run: int,
                          auc_roc: float, avg_precision: float, model, model_name: str) -> int:
        """Save a training run with metrics and model weights."""
        import datetime
        import torch

        # Save model weights
        models_dir = os.path.join(self.data_dir, "trained_models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO training_runs
                   (created_at, model_type, parameters, graph_name, nodes, edges,
                    features, epochs_run, auc_roc, avg_precision, model_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (datetime.datetime.now().isoformat(), model_type, parameters,
                 graph_name, nodes, edges, features, epochs_run,
                 round(auc_roc, 6), round(avg_precision, 6), model_path),
            )
            return cursor.lastrowid

    def list_training_runs(self) -> list[dict]:
        """List all training runs with metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            tables = [t[0] for t in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "training_runs" not in tables:
                return []
            rows = conn.execute(
                "SELECT * FROM training_runs ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_training_run(self, run_id: int):
        """Delete a training run and its model file."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT model_path FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row and row[0] and os.path.exists(row[0]):
                os.remove(row[0])
            conn.execute("DELETE FROM training_runs WHERE id = ?", (run_id,))

    def save_graph(self, graph, name: str):
        """Save a NetworkX graph to the data directory."""
        import pickle
        graphs_dir = os.path.join(self.data_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        path = os.path.join(graphs_dir, f"{name}.gpickle")
        with open(path, "wb") as f:
            pickle.dump(graph, f)
        # Store metadata
        import datetime
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS saved_graphs (
                    name TEXT PRIMARY KEY,
                    nodes INTEGER,
                    edges INTEGER,
                    created_at TEXT,
                    file_path TEXT
                )
            """)
            conn.execute(
                "INSERT OR REPLACE INTO saved_graphs VALUES (?,?,?,?,?)",
                (name, graph.number_of_nodes(), graph.number_of_edges(),
                 datetime.datetime.now().isoformat(), path),
            )

    def load_graph(self, name: str):
        """Load a saved NetworkX graph by name."""
        import pickle
        path = os.path.join(self.data_dir, "graphs", f"{name}.gpickle")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def list_saved_graphs(self) -> list[dict]:
        """List all saved graphs with metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            tables = [t[0] for t in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "saved_graphs" not in tables:
                return []
            rows = conn.execute(
                "SELECT name, nodes, edges, created_at FROM saved_graphs ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_graph(self, name: str):
        """Delete a saved graph."""
        path = os.path.join(self.data_dir, "graphs", f"{name}.gpickle")
        if os.path.exists(path):
            os.remove(path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM saved_graphs WHERE name = ?", (name,))

    def db_size_mb(self) -> float:
        if not os.path.exists(self.db_path):
            return 0.0
        return os.path.getsize(self.db_path) / (1024 * 1024)
