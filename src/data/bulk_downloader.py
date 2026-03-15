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

    def download_all(self, gene: GeneNode, on_progress=None) -> dict:
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
                    task_fn = futures[future]
                    name = task_fn.__name__.replace("fetch_", "").upper()
                    results["errors"].append(name)
                    if on_progress:
                        on_progress(name, "error")

        return results
