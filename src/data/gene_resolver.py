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
