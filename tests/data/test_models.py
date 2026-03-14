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
