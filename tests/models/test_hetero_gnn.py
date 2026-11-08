# tests/models/test_hetero_gnn.py
import pytest
import torch
from torch_geometric.data import HeteroData

from src.models.hetero_gnn import HeteroGNNLinkPredictor

N_GENES, N_TRAITS, GENE_IN, TRAIT_IN, H = 8, 3, 5, 6, 16


def make_data(with_trait_edges=True):
    torch.manual_seed(0)
    data = HeteroData()
    data["gene"].x = torch.randn(N_GENES, GENE_IN)
    data["trait"].x = torch.eye(TRAIT_IN)[:N_TRAITS]
    data["gene", "interacts", "gene"].edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    if with_trait_edges:
        gt = torch.tensor([[0, 1, 2, 5], [0, 0, 1, 2]])
    else:
        gt = torch.zeros((2, 0), dtype=torch.long)
    data["gene", "has", "trait"].edge_index = gt
    data["trait", "rev_has", "gene"].edge_index = gt.flip(0)
    return data


def make_model(**kwargs):
    torch.manual_seed(1)
    return HeteroGNNLinkPredictor(gene_in=GENE_IN, num_traits=N_TRAITS, trait_in=TRAIT_IN, hidden_channels=H, **kwargs)


def test_encode_returns_gene_and_trait_embeddings_of_hidden_size():
    z_gene, z_trait = make_model().encode(make_data())
    assert z_gene.shape == (N_GENES, H)
    assert z_trait.shape == (N_TRAITS, H)
    assert torch.isfinite(z_gene).all() and torch.isfinite(z_trait).all()


def test_decode_scores_are_probabilities_and_match_logits():
    model = make_model()
    z_gene, z_trait = model.encode(make_data())
    gene_idx, trait_idx = torch.tensor([0, 3, 7]), torch.tensor([0, 1, 2])
    logits = model.decode_logits(z_gene, z_trait, gene_idx, trait_idx)
    probs = model.decode(z_gene, z_trait, gene_idx, trait_idx)
    assert logits.shape == (3,)
    assert torch.allclose(probs, torch.sigmoid(logits))
    assert ((probs >= 0) & (probs <= 1)).all()


@pytest.mark.parametrize("decoder", ["dot", "bilinear", "mlp"])
def test_score_matrix_equals_pairwise_decode_logits(decoder):
    model = make_model(decoder=decoder)
    model.eval()
    z_gene, z_trait = model.encode(make_data())
    trait_idx = torch.tensor([2, 0])
    matrix = model.score_matrix(z_gene, z_trait, trait_idx)
    assert matrix.shape == (N_GENES, 2)
    for j, t in enumerate(trait_idx.tolist()):
        expected = model.decode_logits(z_gene, z_trait, torch.arange(N_GENES), torch.full((N_GENES,), t))
        assert torch.allclose(matrix[:, j], expected, atol=1e-6)


def test_unknown_decoder_raises():
    with pytest.raises(ValueError, match="decoder"):
        make_model(decoder="magic")


def test_gradients_reach_trait_embeddings_and_bias():
    model = make_model()
    z_gene, z_trait = model.encode(make_data())
    logits = model.decode_logits(z_gene, z_trait, torch.tensor([0, 5]), torch.tensor([0, 2]))
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1.0, 0.0])).backward()
    assert model.trait_embedding.weight.grad is not None and model.trait_embedding.weight.grad.abs().sum() > 0
    assert model.trait_bias.grad is not None and model.trait_bias.grad.abs().sum() > 0


def test_model_handles_graph_with_no_gene_trait_edges():
    z_gene, z_trait = make_model().encode(make_data(with_trait_edges=False))
    assert z_gene.shape == (N_GENES, H)
    assert z_trait.shape == (N_TRAITS, H)
    assert torch.isfinite(z_trait).all()


def test_dropout_only_active_in_training_mode():
    model = make_model(dropout=0.5)
    data = make_data()
    model.eval()
    a, _ = model.encode(data)
    b, _ = model.encode(data)
    assert torch.equal(a, b)
    model.train()
    c, _ = model.encode(data)
    d, _ = model.encode(data)
    assert not torch.equal(c, d)
