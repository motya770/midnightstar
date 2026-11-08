#!/usr/bin/env python
"""Train the heterogeneous gene–disease model on a saved MidnightStar graph.

By default the benchmark's hidden gene–disease pairs (``make_holdout`` with the
standard protocol) are removed from the graph before training, the model is
benchmarked against the enrichment baseline on those pairs, and the split is
stored in the checkpoint. ``--no-holdout`` skips that benchmark hold-out for a
production model (a seeded validation hold-out of ``--val-fraction`` of each
disease's genes still drives early stopping, and those pairs are kept as
message-passing context in the saved model); no benchmark is run for it.

The Model Training page's *Hetero GNN* mode runs the same pipeline
(``src/models/hetero_pipeline.py``).

Example (from the repo root):

    python scripts/train_hetero_disease_model.py --graph full_s800_gwas --json run.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.bulk_datasets import DATA_DIR_DEFAULT, BulkDatasetManager  # noqa: E402
from src.models.hetero_gnn import DECODERS  # noqa: E402
from src.models.hetero_pipeline import (  # noqa: E402
    benchmark_hetero,
    describe_hetero_params,
    hetero_model_kwargs,
    prepare_hetero_training,
    save_hetero_run,
    train_hetero,
)
from src.models.hetero_trainer import NEG_SAMPLING, HeteroTrainConfig  # noqa: E402


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
    parser.add_argument("--patience", type=int, default=20, help="evaluations without improvement before stopping")
    parser.add_argument("--eval-every", type=int, default=1, help="validate every N epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu or mps")
    parser.add_argument("--no-holdout", action="store_true",
                        help="skip the benchmark hold-out (production model, no benchmark; validation hold-out still applies)")
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

    started = time.time()
    try:
        prepared = prepare_hetero_training(graph, holdout=not args.no_holdout, min_trait_genes=args.min_trait_genes)
    except ValueError as exc:
        print(f"cannot train on {args.graph!r}: {exc}", file=sys.stderr)
        return 1
    split = prepared.split
    print(f"{args.graph}: {len(prepared.bundle.node_list)} genes, {len(prepared.bundle.trait_list)} traits, "
          f"{prepared.n_string} STRING edges, {prepared.n_assoc} gene–trait edges after removing "
          f"{prepared.n_hidden} hidden pairs ({len(split.diseases) if split else 0} benchmark diseases); "
          f"built in {time.time() - started:.0f}s")

    model_kwargs = hetero_model_kwargs(prepared.bundle, hidden=args.hidden, layers=args.layers,
                                       dropout=args.dropout, decoder=args.decoder)
    config = HeteroTrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                               supervision_ratio=args.supervision_ratio, neg_gene=args.neg_gene,
                               neg_trait=args.neg_trait, neg_sampling=args.neg_sampling,
                               val_fraction=args.val_fraction, patience=args.patience, eval_every=args.eval_every,
                               seed=args.seed, device=args.device)
    print(f"decoder={args.decoder}; config: {config}")

    trained = train_hetero(prepared.bundle, prepared.index, split, model_kwargs, config)
    metrics = trained.metrics
    print(f"model: {sum(p.numel() for p in trained.model.parameters()):,} parameters; trained {metrics['epochs_run']} "
          f"epochs in {metrics['train_minutes']} min; best epoch {metrics['best_epoch']}: val MRR {metrics['val_mrr']:.4f}, "
          f"val recall@50 {metrics['val_recall50']:.4f}, val AUC {metrics['val_auc']:.4f}")

    results = benchmark_hetero(trained.model, prepared, graph)
    metrics_out = dict(metrics)
    if results:
        width = max(len(n) for n in results)
        for name, result in results.items():
            print(f"{name:{width}s}  {result.summary()}")
        metrics_out["benchmark"] = {name: result.to_dict() for name, result in results.items()}

    params = describe_hetero_params(model_kwargs, config, holdout=not args.no_holdout)
    saved = save_hetero_run(manager, model=trained.model, model_kwargs=model_kwargs, prepared=prepared, config=config,
                            metrics=metrics, benchmark=results, graph_name=args.graph,
                            min_trait_genes=args.min_trait_genes, params=params, out=args.out,
                            save_run=not args.no_save_run)
    print(f"checkpoint: {saved.checkpoint_path}")
    if saved.run_id is not None:
        print(f"training run #{saved.run_id} recorded (AUC column = validation AUC, "
              f"precision column = benchmark recall@50)")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"args": vars(args), "checkpoint": saved.checkpoint_path, "run_id": saved.run_id,
                       "metrics": metrics_out, "history": trained.history}, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
