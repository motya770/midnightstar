# tests/scripts/test_train_hetero_cli.py
import importlib.util
import json
import os

from src.data.bulk_datasets import BulkDatasetManager
from src.utils.hetero_scorer import load_hetero_checkpoint
from tests.utils.test_hetero_scorer import GENES, make_graph

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "train_hetero_cli", os.path.join(ROOT, "scripts", "train_hetero_disease_model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_trains_benchmarks_and_records_a_run_on_a_saved_graph(tmp_path):
    BulkDatasetManager(str(tmp_path)).save_graph(make_graph(), "toy")
    out = tmp_path / "run.json"
    cli = load_cli()
    assert cli.main(["--data-dir", str(tmp_path), "--graph", "toy", "--hidden", "8", "--epochs", "2",
                     "--json", str(out)]) == 0
    report = json.loads(out.read_text())
    assert os.path.exists(report["checkpoint"]) and report["run_id"] == 1
    assert "hetero" in report["metrics"]["benchmark"] and report["metrics"]["epochs_run"] == 2
    assert len(report["history"]["train_loss"]) == 2
    assert BulkDatasetManager(str(tmp_path)).list_training_runs()[0]["model_type"] == "Hetero GNN"


def test_cli_no_holdout_skips_the_benchmark_and_no_save_run_skips_the_record(tmp_path):
    BulkDatasetManager(str(tmp_path)).save_graph(make_graph(), "toy")
    out = tmp_path / "run.json"
    cli = load_cli()
    assert cli.main(["--data-dir", str(tmp_path), "--graph", "toy", "--hidden", "8", "--epochs", "2",
                     "--no-holdout", "--no-save-run", "--out", str(tmp_path / "prod.pt"), "--json", str(out)]) == 0
    report = json.loads(out.read_text())
    assert "benchmark" not in report["metrics"] and report["run_id"] is None
    assert report["checkpoint"] == str(tmp_path / "prod.pt")
    assert load_hetero_checkpoint(tmp_path / "prod.pt")["split"] is None
    assert BulkDatasetManager(str(tmp_path)).list_training_runs() == []


def test_cli_refuses_an_empty_validation_split(tmp_path, capsys):
    G = make_graph()
    for g in GENES[10:12]:          # Asthma keeps 10 genes: fine for the benchmark, nothing left to validate on
        G.nodes[g].pop("gwas")
    BulkDatasetManager(str(tmp_path)).save_graph(G, "tiny")
    assert load_cli().main(["--data-dir", str(tmp_path), "--graph", "tiny", "--hidden", "8", "--epochs", "2"]) == 1
    assert "early stopping" in capsys.readouterr().err


def test_cli_reports_a_missing_graph(tmp_path, capsys):
    assert load_cli().main(["--data-dir", str(tmp_path), "--graph", "nope"]) == 1
    assert "not found" in capsys.readouterr().err
