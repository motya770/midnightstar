from src.data.models import GeneNode, Association


class TextExplainer:
    """Generates plain-language explanations for genes, associations, and predictions."""

    def explain_gene(self, gene: GeneNode) -> str:
        parts = [f"{gene.symbol}"]
        if gene.name:
            parts.append(f"({gene.name})")
        if gene.description:
            parts.append(f"is a {gene.description}")
        parts.append(f"in {gene.organism}.")
        return " ".join(parts)

    def explain_association(self, assoc: Association) -> str:
        strength = self._score_to_strength(assoc.score)
        source_desc = self._source_description(assoc.data_source)
        return (
            f"{assoc.source_id} and {assoc.target_id} have a {strength} "
            f"{assoc.type} association (score: {assoc.score:.2f}) "
            f"supported by {source_desc} evidence."
        )

    def explain_prediction(
        self,
        gene: str,
        disease: str,
        confidence: float,
        top_features: list[str],
    ) -> str:
        strength = self._score_to_strength(confidence)
        feature_str = ", ".join(top_features[:3]) if top_features else "no key features"
        return (
            f"The model predicts a {strength} link between {gene} and {disease} "
            f"(confidence: {confidence:.2f}). "
            f"Top contributing features: {feature_str}."
        )

    def _score_to_strength(self, score: float) -> str:
        if score >= 0.7:
            return "strong"
        if score >= 0.5:
            return "moderate"
        return "weak"

    def _source_description(self, source: str) -> str:
        descriptions = {
            "STRING": "protein interaction database (STRING)",
            "GWAS": "genome-wide association study (GWAS)",
            "GTEx": "tissue expression (GTEx)",
            "HPA": "Human Protein Atlas (HPA)",
        }
        return descriptions.get(source, source)
