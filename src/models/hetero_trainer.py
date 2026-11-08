"""Train a heterogeneous gene–trait link predictor on gene–trait edges.

STRING edges always carry messages and are never targets. Gene–trait edges are
the targets: each epoch a fresh random share of them is supervised while the
rest carry messages, so the model never scores an edge it can see. Validation
is a seeded hold-out of known disease genes scored the benchmark way (MRR), and
the benchmark's own hidden pairs were removed from the bundle before training.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.utils.benchmark import BenchmarkResult, HoldoutSplit, make_holdout, mask_index, run_benchmark
from src.utils.hetero_graph_features import GENE_GENE, GENE_TRAIT, TRAIT_GENE, HeteroBundle

NEG_SAMPLING = ("uniform", "degree")


@dataclass
class HeteroTrainConfig:
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-5
    supervision_ratio: float = 0.3
    neg_gene: int = 4
    neg_trait: int = 1
    neg_sampling: str = "uniform"
    val_fraction: float = 0.1
    val_seed: int = 1
    early_stopping: bool = True
    patience: int = 20
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self):
        if not 0 < self.supervision_ratio < 1:
            raise ValueError("supervision_ratio must be strictly between 0 and 1")
        if self.neg_sampling not in NEG_SAMPLING:
            raise ValueError(f"neg_sampling must be one of {NEG_SAMPLING}")
        if self.neg_gene < 0 or self.neg_trait < 0:
            raise ValueError("negative counts cannot be negative")


class HeteroTrainer:
    def __init__(self, model: nn.Module, config: HeteroTrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        optimizer_cls = torch.optim.AdamW if config.weight_decay > 0 else torch.optim.Adam
        self.optimizer = optimizer_cls(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.val_split: HoldoutSplit | None = None
        self._bundle: HeteroBundle | None = None
        self._data: HeteroData | None = None
        self._graph = None
        self._index = None
        self._train_edges: torch.Tensor | None = None   # [2, E] (gene idx, trait idx), on device
        self._all_edges: torch.Tensor | None = None     # every gene–trait edge of the bundle
        self._known_keys: torch.Tensor | None = None    # sorted gene*T+trait keys, for negative filtering
        self._gene_weights: torch.Tensor | None = None  # degree-aware negative weights
        self._val_pairs: torch.Tensor | None = None
        self._gen = torch.Generator()

    # ---- setup -----------------------------------------------------------------

    def _seed_everything(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self._gen.manual_seed(self.config.seed)

    def _pairs_to_edges(self, pairs) -> torch.Tensor:
        b = self._bundle
        rows = [(b.node_to_idx[g], b.trait_to_idx[t]) for g, t in pairs
                if g in b.node_to_idx and t in b.trait_to_idx]
        if not rows:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(rows, dtype=torch.long).t()

    def _setup(self, bundle: HeteroBundle, index, split: HoldoutSplit | None, graph):
        self._seed_everything()
        self._bundle = bundle
        self._graph = graph
        self._index = index
        self._data = bundle.data.to(self.device)

        masked = mask_index(index, split) if split is not None else index
        if split is not None:
            categories, min_known = split.categories, split.min_known_genes
        else:
            categories, min_known = ("disease",), 10
        self.val_split = make_holdout(masked, categories=categories, min_known_genes=min_known,
                                      hidden_fraction=self.config.val_fraction, seed=self.config.val_seed)
        self._val_index = masked

        all_edges = bundle.data[GENE_TRAIT].edge_index.cpu()
        num_traits = max(len(bundle.trait_list), 1)
        val_pairs = {(g, d) for d, genes in self.val_split.hidden.items() for g in genes}
        self._val_pairs = self._pairs_to_edges(val_pairs)
        val_keys = self._val_pairs[0] * num_traits + self._val_pairs[1]
        keys = all_edges[0] * num_traits + all_edges[1]
        keep = ~torch.isin(keys, val_keys)
        self._all_edges = all_edges.to(self.device)
        self._train_edges = all_edges[:, keep].to(self.device)
        self._known_keys = torch.unique(keys)
        self._gene_weights = torch.bincount(all_edges[0][keep], minlength=len(bundle.node_list)).float()

    # ---- per-epoch edge handling ----------------------------------------------------

    def _epoch_edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the training edges into (supervision, message) for one epoch."""
        edges = self._train_edges
        n = edges.size(1)
        perm = torch.randperm(n, generator=self._gen).to(edges.device)
        n_sup = max(1, int(n * self.config.supervision_ratio)) if n else 0
        return edges[:, perm[:n_sup]], edges[:, perm[n_sup:]]

    def _message_data(self, gene_trait_edges: torch.Tensor) -> HeteroData:
        data = HeteroData()
        data["gene"].x = self._data["gene"].x
        data["trait"].x = self._data["trait"].x
        data[GENE_GENE].edge_index = self._data[GENE_GENE].edge_index
        data[GENE_TRAIT].edge_index = gene_trait_edges
        data[TRAIT_GENE].edge_index = gene_trait_edges.flip(0)
        return data

    def _sample_negatives(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Corrupt the gene side neg_gene times and the trait side neg_trait times per positive,
        dropping any pair that is a known association of the bundle."""
        cfg = self.config
        n_genes = len(self._bundle.node_list)
        n_traits = max(len(self._bundle.trait_list), 1)
        pos = pos.cpu()
        parts_g, parts_t = [], []
        if cfg.neg_gene > 0 and pos.size(1) > 0:
            k = pos.size(1) * cfg.neg_gene
            if cfg.neg_sampling == "degree" and self._gene_weights.sum() > 0:
                genes = torch.multinomial(self._gene_weights, k, replacement=True, generator=self._gen)
            else:
                genes = torch.randint(0, n_genes, (k,), generator=self._gen)
            parts_g.append(genes)
            parts_t.append(pos[1].repeat(cfg.neg_gene))
        if cfg.neg_trait > 0 and pos.size(1) > 0:
            k = pos.size(1) * cfg.neg_trait
            parts_g.append(pos[0].repeat(cfg.neg_trait))
            parts_t.append(torch.randint(0, n_traits, (k,), generator=self._gen))
        if not parts_g:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty
        g = torch.cat(parts_g)
        t = torch.cat(parts_t)
        keep = ~torch.isin(g * n_traits + t, self._known_keys)
        return g[keep].to(self.device), t[keep].to(self.device)

    # ---- training ---------------------------------------------------------------------

    def _train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        sup, msg = self._epoch_edges()
        z_gene, z_trait = self.model.encode(self._message_data(msg))
        neg_g, neg_t = self._sample_negatives(sup)
        gene_idx = torch.cat([sup[0], neg_g])
        trait_idx = torch.cat([sup[1], neg_t])
        labels = torch.cat([torch.ones(sup.size(1), device=self.device), torch.zeros(neg_g.numel(), device=self.device)])
        logits = self.model.decode_logits(z_gene, z_trait, gene_idx, trait_idx)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _encode(self, message_edges: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            return self.model.encode(self._message_data(message_edges))

    def _matrix_from(self, z_gene, z_trait, diseases) -> np.ndarray:
        b = self._bundle
        known = [d for d in diseases if d in b.trait_to_idx]
        matrix = np.zeros((len(b.node_list), len(diseases)))
        if known:
            idx = torch.tensor([b.trait_to_idx[d] for d in known], device=z_gene.device)
            with torch.no_grad():
                scores = self.model.score_matrix(z_gene, z_trait, idx).cpu().numpy()
            cols = [j for j, d in enumerate(diseases) if d in b.trait_to_idx]
            matrix[:, cols] = scores
        return matrix

    def _validate(self) -> dict[str, float]:
        z_gene, z_trait = self._encode(self._train_edges)
        out = {"val_mrr": 0.0, "val_recall50": 0.0, "val_auc": 0.5}
        if self.val_split.diseases:
            matrix = self._matrix_from(z_gene, z_trait, self.val_split.diseases)
            result = run_benchmark(matrix, self._graph, self._bundle.node_list, self._bundle.node_to_idx,
                                   self._val_index, self.val_split)
            out["val_mrr"], out["val_recall50"] = result.mrr, result.recall[50]
        pairs = self._val_pairs
        if pairs.size(1) > 0:
            pos_g, pos_t = pairs[0].to(self.device), pairs[1].to(self.device)
            neg_g = torch.randint(0, len(self._bundle.node_list), (pairs.size(1),), generator=self._gen).to(self.device)
            with torch.no_grad():
                pos = self.model.decode_logits(z_gene, z_trait, pos_g, pos_t)
                neg = self.model.decode_logits(z_gene, z_trait, neg_g, pos_t)
            labels = np.concatenate([np.ones(pos.numel()), np.zeros(neg.numel())])
            out["val_auc"] = float(roc_auc_score(labels, torch.cat([pos, neg]).cpu().numpy()))
        return out

    def train(self, bundle: HeteroBundle, index, split: HoldoutSplit | None, graph, on_epoch=None) -> dict:
        self._setup(bundle, index, split, graph)
        cfg = self.config
        print(f"[DEBUG] hetero training: {len(bundle.node_list)} genes, {len(bundle.trait_list)} traits, "
              f"{self._data[GENE_GENE].edge_index.size(1) // 2} STRING edges, "
              f"{self._train_edges.size(1)} training gene–trait edges, {self._val_pairs.size(1)} validation pairs "
              f"({len(self.val_split.diseases)} diseases), device {self.device}")

        history = {"train_loss": [], "val_mrr": [], "val_recall50": [], "val_auc": [], "best_epoch": 0}
        best, best_state, patience = -1.0, None, 0
        pbar = tqdm(range(cfg.epochs), desc="Epochs", unit="epoch")
        for epoch in pbar:
            loss = self._train_step()
            val = self._validate()
            history["train_loss"].append(loss)
            for key in ("val_mrr", "val_recall50", "val_auc"):
                history[key].append(val[key])
            pbar.set_postfix(loss=f"{loss:.4f}", val_mrr=f"{val['val_mrr']:.4f}", val_r50=f"{val['val_recall50']:.3f}")
            if on_epoch:
                on_epoch(epoch, loss, val["val_mrr"])
            if val["val_mrr"] > best:
                best, patience = val["val_mrr"], 0
                history["best_epoch"] = epoch
                if cfg.early_stopping:
                    best_state = copy.deepcopy(self.model.state_dict())
            elif cfg.early_stopping:
                patience += 1
                if patience >= cfg.patience:
                    break
        pbar.close()
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    # ---- scoring ------------------------------------------------------------------------

    def score_matrix(self, diseases, message_edges: torch.Tensor | None = None) -> np.ndarray:
        """(n_genes, len(diseases)) logits; messages default to every gene–trait edge of the bundle
        (training and validation pairs), which matches honest inference on the saved checkpoint."""
        z_gene, z_trait = self._encode(self._all_edges if message_edges is None else message_edges)
        return self._matrix_from(z_gene, z_trait, list(diseases))

    def evaluate(self, split: HoldoutSplit) -> BenchmarkResult:
        return run_benchmark(self.score_matrix(split.diseases), self._graph, self._bundle.node_list,
                             self._bundle.node_to_idx, self._index, split)
