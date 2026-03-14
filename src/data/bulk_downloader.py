import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data.models import GeneNode, Association, DiseaseNode, ExpressionProfile

logger = logging.getLogger(__name__)


class BulkDownloader:
    """Downloads data for a list of genes from all sources in parallel."""

    def __init__(self, string_client, gwas_client, gtex_client):
        self._string = string_client
        self._gwas = gwas_client
        self._gtex = gtex_client

    def download(self, genes: list[GeneNode]) -> dict:
        """Download data for all genes. Returns dict with associations, diseases,
        expression, and errors keys."""
        all_associations: list[Association] = []
        all_diseases: list[DiseaseNode] = []
        all_expression: list[ExpressionProfile] = []
        errors: dict[str, dict] = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for gene in genes:
                futures[executor.submit(self._fetch_gene, gene)] = gene

            for future in as_completed(futures):
                gene = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("Unexpected error for gene %s: %s", gene.symbol, exc)
                    errors[gene.symbol] = {"string": exc, "gwas": exc, "gtex": exc}
                    continue

                all_associations.extend(result["associations"])
                all_diseases.extend(result["diseases"])
                all_expression.extend(result["expression"])
                if result["errors"]:
                    errors[gene.symbol] = result["errors"]

        return {
            "associations": all_associations,
            "diseases": all_diseases,
            "expression": all_expression,
            "errors": errors,
        }

    def _fetch_gene(self, gene: GeneNode) -> dict:
        associations: list[Association] = []
        diseases: list[DiseaseNode] = []
        expression: list[ExpressionProfile] = []
        gene_errors: dict[str, Exception | None] = {}

        # STRING interactions
        try:
            string_assocs = self._string.get_interactions(gene.symbol)
            associations.extend(string_assocs)
            gene_errors["string"] = None
        except Exception as exc:
            logger.warning("STRING error for %s: %s", gene.symbol, exc)
            gene_errors["string"] = exc

        # GWAS associations and diseases
        try:
            gwas_assocs, gwas_diseases = self._gwas.search_gene(gene.symbol)
            associations.extend(gwas_assocs)
            diseases.extend(gwas_diseases)
            gene_errors["gwas"] = None
        except Exception as exc:
            logger.warning("GWAS error for %s: %s", gene.symbol, exc)
            gene_errors["gwas"] = exc

        # GTEx expression
        try:
            expr_profiles = self._gtex.get_expression(gene.ensembl_id)
            expression.extend(expr_profiles)
            gene_errors["gtex"] = None
        except Exception as exc:
            logger.warning("GTEx error for %s: %s", gene.symbol, exc)
            gene_errors["gtex"] = exc

        # Only include errors dict if there were actual errors
        has_errors = any(v is not None for v in gene_errors.values())
        return {
            "associations": associations,
            "diseases": diseases,
            "expression": expression,
            "errors": gene_errors if has_errors else {},
        }
