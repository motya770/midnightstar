#!/usr/bin/env python
"""Train the heterogeneous gene–disease model on a saved MidnightStar graph.

By default the benchmark's hidden gene–disease pairs (``make_holdout`` with the
standard protocol) are removed from the graph before training, the model is
benchmarked against the enrichment baseline on those pairs, and the split is
stored in the checkpoint. ``--no-holdout`` trains on every known association for
a production model; its benchmark numbers are then meaningless and are not run.

Example (from the repo root):

    python scripts/train_hetero_disease_model.py --graph full_s800_gwas --json run.json
"""
import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.data.bulk_datasets import DATA_DIR_DEFAULT, BulkDatasetManager  # noqa: E402
from src.models.hetero_gnn import DECODERS, HeteroGNNLinkPredictor  # noqa: E402
from src.models.hetero_trainer import NEG_SAMPLING, HeteroTrainConfig, HeteroTrainer  # noqa: E402
from src.utils.benchmark import make_holdout, reference_scorers  # noqa: E402
from src.utils.discovery import index_disease_genes  # noqa: E402
from src.utils.hetero_graph_features import TRAIT_CATEGORIES, GENE_GENE, GENE_TRAIT, nx_to_hetero_data  # noqa: E402
from src.utils.hetero_scorer import HETERO_MODEL_TYPE, save_hetero_checkpoint  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DATA_DIR_DEFAULT)
    parser.add_argument("--graph", default="full_s800_gwas")
    parser.add_argument("--out", help="checkpoint path (default: <data-dir>/trained_models/hetero_gnn_<stamp>_checkpoint.pt)")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--decoder", choices=DECODERS, default="dot")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--supervision-ratio", type=float, default=0.3)
    parser.add_argument("--neg-gene", type=int, default=4)
    parser.add_argument("--neg-trait", type=int, default=1)
    parser.add_argument("--neg-sampling", choices=NEG_SAMPLING, default="uniform")
    parser.add_argument("--min-trait-genes", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu or mps")
    parser.add_argument("--no-holdout", action="store_true", help="train on all pairs (production model, no benchmark)")
    parser.add_argument("--no-save-run", action="store_true", help="do not record the run in the training-runs table")
    parser.add_argument("--json", help="write history, metrics and paths to this file")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manager = BulkDatasetManager(args.data_dir)
    graph = manager.load_graph(args.graph)
    if graph is None:
        print(f"Graph {args.graph!r} not found under {args.data_dir}", file=sys.stderr)
        return 1
    node_list = sorted(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    index = index_disease_genes(graph, node_list)

    split = None if args.no_holdout else make_holdout(index)
    hidden = {(g, d) for d, genes in split.hidden.items() for g in genes} if split else set()
    started = time.time()
    bundle = nx_to_hetero_data(graph, exclude_pairs=hidden, min_trait_genes=args.min_trait_genes)
    n_string = bundle.data[GENE_GENE].edge_index.size(1) // 2
    n_assoc = bundle.data[GENE_TRAIT].edge_index.size(1)
    print(f"{args.graph}: {len(bundle.node_list)} genes, {len(bundle.trait_list)} traits, {n_string} STRING edges, "
          f"{n_assoc} gene–trait edges after removing {len(hidden)} hidden pairs "
          f"({len(split.diseases) if split else 0} benchmark diseases); built in {time.time() - started:.0f}s")

    torch.manual_seed(args.seed)
    model_kwargs = dict(gene_in=bundle.data["gene"].x.size(1), num_traits=len(bundle.trait_list),
                        trait_in=len(TRAIT_CATEGORIES), hidden_channels=args.hidden, num_layers=args.layers,
                        dropout=args.dropout, decoder=args.decoder)
    model = HeteroGNNLinkPredictor(**model_kwargs)
    config = HeteroTrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                               supervision_ratio=args.supervision_ratio, neg_gene=args.neg_gene,
                               neg_trait=args.neg_trait, neg_sampling=args.neg_sampling,
                               val_fraction=args.val_fraction, patience=args.patience, seed=args.seed,
                               device=args.device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters, decoder={args.decoder}; config: {config}")

    started = time.time()
    trainer = HeteroTrainer(model, config)
    history = trainer.train(bundle, index, split, graph)
    best = history["best_epoch"]
    metrics = {
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best,
        "val_mrr": history["val_mrr"][best],
        "val_recall50": history["val_recall50"][best],
        "val_auc": history["val_auc"][best],
        "train_minutes": round((time.time() - started) / 60, 2),
    }
    print(f"trained {metrics['epochs_run']} epochs in {metrics['train_minutes']} min; best epoch {best}: "
          f"val MRR {metrics['val_mrr']:.4f}, val recall@50 {metrics['val_recall50']:.4f}, val AUC {metrics['val_auc']:.4f}")

    results = {}
    if split is not None:
        results["hetero"] = trainer.evaluate(split)
        results.update(reference_scorers(["enrichment"], graph, node_list, node_to_idx, index, split))
        for name, result in results.items():
            print(f"{name:12s} {result.summary()}")
        metrics["benchmark"] = {name: result.to_dict() for name, result in results.items()}

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"hetero_gnn_{stamp}"
    out = args.out or os.path.join(args.data_dir, "trained_models", f"{model_name}_checkpoint.pt")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_hetero_checkpoint(out, model=model, model_kwargs=model_kwargs, bundle=bundle, split=split, config=config,
                           metrics=metrics, graph_name=args.graph, min_trait_genes=args.min_trait_genes)
    print(f"checkpoint: {out}")

    run_id = None
    if not args.no_save_run:
        params = (f"hidden={args.hidden}, layers={args.layers}, decoder={args.decoder}, dropout={args.dropout}, "
                  f"neg={args.neg_gene}g+{args.neg_trait}t/{args.neg_sampling}, lr={args.lr}, wd={args.weight_decay}, "
                  f"epochs={args.epochs}, seed={args.seed}, holdout={'no' if args.no_holdout else 'benchmark'}")
        bench = metrics.get("benchmark", {}).get("hetero", {})
        run_id = manager.save_training_run(
            model_type=HETERO_MODEL_TYPE, parameters=params, graph_name=args.graph,
            nodes=len(bundle.node_list), edges=n_string, features=bundle.data["gene"].x.size(1),
            epochs_run=metrics["epochs_run"], auc_roc=metrics["val_auc"],
            avg_precision=bench.get("recall@50", metrics["val_recall50"]),
            model=model.cpu(), model_name=model_name,
        )
        print(f"training run #{run_id} recorded (AUC column = validation AUC, precision column = benchmark recall@50)")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"args": vars(args), "checkpoint": out, "run_id": run_id, "metrics": metrics,
                       "history": history}, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
