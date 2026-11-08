# tests/models/test_hetero_pipeline.py
import os

import pytest

from src.data.bulk_datasets import BulkDatasetManager
from src.models.hetero_pipeline import (
    benchmark_hetero,
    describe_hetero_params,
    hetero_model_kwargs,
    prepare_hetero_training,
    save_hetero_run,
    train_hetero,
)
from src.models.hetero_trainer import HeteroTrainConfig
from src.utils.benchmark import HETERO_SCORERS, BenchmarkResult
from src.utils.hetero_scorer import hetero_score_matrix, is_hetero_checkpoint_path, load_hetero_checkpoint
from tests.utils.test_hetero_scorer import GENES, make_graph

CONFIG = dict(epochs=3, early_stopping=False)


def trained_toy(G=None, holdout=True):
    G = G or make_graph()
    prepared = prepare_hetero_training(G, holdout=holdout)
    kwargs = hetero_model_kwargs(prepared.bundle, hidden=8)
    config = HeteroTrainConfig(**CONFIG)
    trained = train_hetero(prepared.bundle, prepared.index, prepared.split, kwargs, config)
    return G, prepared, kwargs, config, trained


def test_prepare_builds_split_and_bundle_without_hidden_pairs():
    prepared = prepare_hetero_training(make_graph())
    assert prepared.split.diseases == ["Asthma"]
    assert prepared.n_hidden == sum(len(g) for g in prepared.split.hidden.values()) > 0
    assert prepared.n_assoc == 20 - prepared.n_hidden
    assert prepared.n_string == 24
    assert prepared.node_list == sorted(make_graph().nodes())
    production = prepare_hetero_training(make_graph(), holdout=False)
    assert production.split is None and production.n_hidden == 0 and production.n_assoc == 20


def test_prepare_refuses_graph_without_gwas():
    G = make_graph()
    for n in G.nodes:
        G.nodes[n].pop("gwas", None)
    with pytest.raises(ValueError, match="GWAS"):
        prepare_hetero_training(G)


def test_prepare_refuses_holdout_when_no_disease_qualifies():
    G = make_graph()
    for g in GENES[6:12]:          # Asthma keeps 6 genes, below the 10 the protocol needs
        G.nodes[g].pop("gwas")
    with pytest.raises(ValueError, match="10"):
        prepare_hetero_training(G)
    assert prepare_hetero_training(G, holdout=False).split is None


def test_train_hetero_returns_metrics_and_cpu_model():
    _, _, _, _, trained = trained_toy()
    assert set(trained.metrics) == {"epochs_run", "best_epoch", "val_mrr", "val_recall50", "val_auc", "train_minutes"}
    assert trained.metrics["epochs_run"] == 3 == len(trained.history["train_loss"])
    assert all(p.device.type == "cpu" for p in trained.model.parameters())


def test_benchmark_hetero_reports_enrichment_and_hetero_scorers():
    G, prepared, _, _, trained = trained_toy()
    results = benchmark_hetero(trained.model, prepared, G)
    assert list(results) == ["enrichment", *HETERO_SCORERS]
    assert all(isinstance(r, BenchmarkResult) and r.n_diseases == 1 for r in results.values())
    assert benchmark_hetero(trained.model, prepare_hetero_training(G, holdout=False), G) == {}


def test_save_hetero_run_writes_loadable_checkpoint_and_records_run(tmp_path):
    manager = BulkDatasetManager(str(tmp_path))
    G, prepared, kwargs, config, trained = trained_toy()
    bench = benchmark_hetero(trained.model, prepared, G)
    params = describe_hetero_params(kwargs, config, holdout=True)
    saved = save_hetero_run(manager, model=trained.model, model_kwargs=kwargs, prepared=prepared, config=config,
                            metrics=trained.metrics, benchmark=bench, graph_name="toy", min_trait_genes=1, params=params)
    assert os.path.dirname(saved.checkpoint_path) == str(tmp_path / "trained_models")
    assert is_hetero_checkpoint_path(saved.checkpoint_path)
    assert saved.model_name == os.path.basename(saved.checkpoint_path)[: -len("_checkpoint.pt")]
    checkpoint = load_hetero_checkpoint(saved.checkpoint_path)
    assert checkpoint["split"]["hidden"] == prepared.split.hidden
    assert checkpoint["metrics"]["benchmark"]["hetero"]["recall@50"] == bench["hetero"].recall[50]
    assert hetero_score_matrix(checkpoint, G, ["Asthma"], honest=True).shape == (len(prepared.node_list), 1)
    run = manager.list_training_runs()[0]
    assert run["id"] == saved.run_id and run["model_type"] == "Hetero GNN" and run["parameters"] == params
    assert run["avg_precision"] == round(bench["hetero"].recall[50], 6)
    assert run["auc_roc"] == round(trained.metrics["val_auc"], 6)
    plain = save_hetero_run(manager, model=trained.model, model_kwargs=kwargs, prepared=prepared, config=config,
                            metrics=trained.metrics, benchmark=None, graph_name="toy", min_trait_genes=1,
                            params=params, out=str(tmp_path / "x.pt"), save_run=False)
    assert plain.run_id is None and (tmp_path / "x.pt").exists()
    assert "benchmark" not in load_hetero_checkpoint(tmp_path / "x.pt")["metrics"]
    assert len(manager.list_training_runs()) == 1


def test_describe_hetero_params_matches_cli_format():
    kwargs = dict(gene_in=3, num_traits=2, trait_in=6, hidden_channels=128, num_layers=2, dropout=0.2, decoder="dot")
    config = HeteroTrainConfig()
    assert describe_hetero_params(kwargs, config, holdout=True) == (
        "hidden=128, layers=2, decoder=dot, dropout=0.2, neg=4g+1t/uniform, lr=0.001, wd=1e-05, "
        "epochs=300, seed=42, holdout=benchmark")
    assert describe_hetero_params(kwargs, config, holdout=False, gpu="A100").endswith("holdout=no, gpu=A100")
