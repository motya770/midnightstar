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
