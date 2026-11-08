#!/usr/bin/env python
"""Held-out disease-gene benchmark for MidnightStar scorers.

Hides 20% of each disease's known GWAS genes (seeded), scores every gene from
the rest and reports how well the hidden genes are recovered. See
``src/utils/benchmark.py`` for the protocol.

Examples (run from the repo root):

    python scripts/benchmark_disease_genes.py --graph full_s800_gwas
    python scripts/benchmark_disease_genes.py --graph full_s800_gwas \
        --scorers enrichment,coassociation,coassociation-strict
    python scripts/benchmark_disease_genes.py --graph full_s800_gwas \
        --checkpoint .datasets/trained_models/gnn_20261030_012253_checkpoint.pt \
        --scorers enrichment,cosine,centred-cosine --json benchmark.json
    python scripts/benchmark_disease_genes.py --graph full_s800_gwas \
        --hetero-checkpoint .datasets/trained_models/hetero_gnn_<stamp>_checkpoint.pt

Without options the benchmark runs enrichment (the production default) and the two
trait co-association rows, which need no model. With a hetero checkpoint the benchmark uses the split stored in it, so the model
is scored on exactly the pairs it never saw; a checkpoint trained with
--no-holdout has no such split and is refused.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.data.bulk_datasets import DATA_DIR_DEFAULT, BulkDatasetManager  # noqa: E402
from src.utils.benchmark import (  # noqa: E402
    COASSOCIATION_SCORERS, EMBEDDING_SCORERS, HETERO_SCORERS, REFERENCE_SCORERS, HoldoutSplit, make_holdout,
    reference_scorers,
)
from src.utils.discovery import index_disease_genes  # noqa: E402
from src.utils.hetero_scorer import hetero_score_matrix, load_hetero_checkpoint  # noqa: E402


def load_embeddings(checkpoint_path: str, graph):
    """Encode the graph with a saved protein–protein checkpoint; returns (z, model_type)."""
    from src.models.gnn import GNNLinkPredictor
    from src.models.graph_transformer import GraphTransformerLinkPredictor
    from src.models.vae import GraphVAE
    from src.utils.graph_features import nx_to_pyg_data

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = {
        "GNNLinkPredictor": GNNLinkPredictor,
        "GraphTransformerLinkPredictor": GraphTransformerLinkPredictor,
        "GraphVAE": GraphVAE,
    }
    model = classes[checkpoint["model_class"]](**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    data, _, _ = nx_to_pyg_data(graph)
    with torch.no_grad():
        z = model.encode(data)
    return z, checkpoint.get("model_type", checkpoint["model_class"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DATA_DIR_DEFAULT, help="MidnightStar .datasets folder")
    parser.add_argument("--graph", default="full_s800_gwas", help="saved graph name")
    parser.add_argument("--checkpoint", help="protein–protein checkpoint whose embeddings the cosine scorers use")
    parser.add_argument("--hetero-checkpoint", help="hetero gene–disease checkpoint for the hetero scorers")
    parser.add_argument("--scorers", default=None,
                        help=f"comma-separated subset of {','.join(REFERENCE_SCORERS)} "
                             "(default: all that the inputs allow)")
    parser.add_argument("--categories", default="disease", help="comma-separated trait categories")
    parser.add_argument("--min-known", type=int, default=10, help="minimum known genes per trait")
    parser.add_argument("--hidden-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="write results and split provenance to this file")
    args = parser.parse_args(argv)

    manager = BulkDatasetManager(args.data_dir)
    graph = manager.load_graph(args.graph)
    if graph is None:
        print(f"Graph {args.graph!r} not found under {args.data_dir}", file=sys.stderr)
        return 1
    node_list = sorted(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    index = index_disease_genes(graph, node_list)
    split = make_holdout(index, categories=args.categories.split(","), min_known_genes=args.min_known,
                         hidden_fraction=args.hidden_fraction, seed=args.seed)
    hetero_checkpoint = None
    if args.hetero_checkpoint:
        hetero_checkpoint = load_hetero_checkpoint(args.hetero_checkpoint)
        if not hetero_checkpoint.get("split"):
            print(f"{args.hetero_checkpoint} was trained without a hold-out (--no-holdout); its benchmark "
                  f"numbers would be leaked, so it cannot be benchmarked", file=sys.stderr)
            return 1
        stored = HoldoutSplit.from_dict(hetero_checkpoint["split"])
        if stored != split:
            print(f"note: using the split stored in the hetero checkpoint (seed {stored.seed}, "
                  f"{stored.min_known_genes}+ known genes, {stored.hidden_fraction:.0%} hidden) instead of the "
                  f"command-line parameters, so the model is scored on pairs it never saw")
            split = stored
    n_hidden = sum(len(g) for g in split.hidden.values())
    n_genes = sum(1 for n in node_list if graph.nodes[n].get("node_type") == "gene")
    print(f"{args.graph}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges; "
          f"{len(split.diseases)} traits with >= {args.min_known} known genes, {n_hidden} hidden genes; "
          f"chance recall@50 ~ {50 / n_genes:.4f}")

    z, model_type = (None, None)
    if args.checkpoint:
        z, model_type = load_embeddings(args.checkpoint, graph)
        print(f"embeddings: {model_type} {tuple(z.shape)} from {args.checkpoint}")

    hetero_scores = None
    if hetero_checkpoint is not None:
        started = time.time()
        hetero_scores = hetero_score_matrix(hetero_checkpoint, graph, split.diseases, honest=True)
        print(f"hetero scores: {hetero_scores.shape} from {args.hetero_checkpoint} ({time.time() - started:.0f}s)")

    if args.scorers:
        names = args.scorers.split(",")
    else:
        names = ["enrichment"] + list(COASSOCIATION_SCORERS)
        if z is not None:
            names += list(EMBEDDING_SCORERS)
        if hetero_scores is not None:
            names += list(HETERO_SCORERS)

    started = time.time()
    results = reference_scorers(names, graph, node_list, node_to_idx, index, split, z=z, hetero_scores=hetero_scores)
    width = max(len(n) for n in names)
    for name, result in results.items():
        print(f"{name:{width}s}  {result.summary()}")
    print(f"({time.time() - started:.0f}s)")

    if args.json:
        payload = {
            "graph": args.graph,
            "checkpoint": args.checkpoint,
            "hetero_checkpoint": args.hetero_checkpoint,
            "model_type": model_type,
            "split": {k: v for k, v in split.to_dict().items() if k not in ("visible", "hidden")},
            "n_hidden": n_hidden,
            "results": {name: result.to_dict() for name, result in results.items()},
        }
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
