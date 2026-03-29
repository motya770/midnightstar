# src/models/trainer.py
from dataclasses import dataclass
from typing import Any

import random

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 0.001
    batch_size: int = 512
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    early_stopping: bool = False
    patience: int = 10
    mini_batch: bool = False
    edge_dropout: float = 0.3  # fraction of edges to drop during encode (prevents overfitting)
    device: str = "cpu"  # "cpu", "mps", or "cuda"


class Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig):
        self.config = config
        self._has_kl = hasattr(model, "kl_loss")

        # Set device
        self.device = torch.device(config.device)
        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self._train_data = None
        self._val_data = None
        self._test_data = None

    def _split_data(self, data):
        """Manual edge-level split without RandomLinkSplit (avoids pyg-lib dependency)."""
        edge_index = data.edge_index
        # Get unique undirected edges
        edge_set = set()
        edges = []
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            key = (min(src, dst), max(src, dst))
            if key not in edge_set:
                edge_set.add(key)
                edges.append(key)

        random.shuffle(edges)
        n = len(edges)
        n_train = int(n * self.config.train_ratio)
        test_ratio = max(0.01, 1.0 - self.config.train_ratio - self.config.val_ratio)
        n_val = int(n * self.config.val_ratio)

        train_edges = edges[:n_train]
        val_edges = edges[n_train:n_train + n_val]
        test_edges = edges[n_train + n_val:]

        def _to_undirected(edge_list):
            if not edge_list:
                return torch.zeros((2, 0), dtype=torch.long)
            src = [e[0] for e in edge_list]
            dst = [e[1] for e in edge_list]
            return torch.tensor([src + dst, dst + src], dtype=torch.long)

        # All original edges (undirected) — used for negative sampling to avoid
        # labelling any real edge as a negative, regardless of split.
        all_pos_ei = _to_undirected(edges)

        def _make_split(edge_list, msg_edges):
            """Create a Data object with edge_index (for message passing) and
            edge_label_index + edge_label (for supervision)."""
            ei = _to_undirected(msg_edges)
            # Positive supervision edges
            pos_src = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
            pos_dst = torch.tensor([e[1] for e in edge_list], dtype=torch.long)
            # Negative supervision edges — sample against ALL positive edges
            # (not just msg_edges) to avoid mislabelling val/test edges as negative
            neg = negative_sampling(all_pos_ei, num_nodes=data.num_nodes,
                                    num_neg_samples=len(edge_list))
            neg_src, neg_dst = neg[0], neg[1]
            # Combine
            label_src = torch.cat([pos_src, neg_src])
            label_dst = torch.cat([pos_dst, neg_dst])
            labels = torch.cat([torch.ones(len(edge_list)), torch.zeros(neg_src.size(0))])
            split_data = Data(
                x=data.x, edge_index=ei, num_nodes=data.num_nodes,
                edge_label_index=torch.stack([label_src, label_dst]),
                edge_label=labels,
            )
            # Carry GWAS token tensors through splits
            for attr in ("gwas_token_ids", "gwas_scores", "gwas_cat_ids",
                         "gwas_vocab_size", "gwas_num_categories"):
                if hasattr(data, attr):
                    setattr(split_data, attr, getattr(data, attr))
            return split_data

        # Split train edges into message-passing and supervision subsets.
        # Supervision edges must NOT appear in the MP graph — otherwise the model
        # learns to detect "this edge exists in my graph" which doesn't transfer
        # to held-out edges at val/test time.
        n_msg = int(len(train_edges) * 0.85)
        msg_edges = train_edges[:n_msg]
        sup_edges = train_edges[n_msg:]

        self._train_data = _make_split(sup_edges, msg_edges).to(self.device)
        self._val_data = _make_split(val_edges, msg_edges).to(self.device)
        self._test_data = _make_split(test_edges, msg_edges).to(self.device)
        # Keep all original edges for negative sampling during training
        self._all_pos_ei = all_pos_ei.to(self.device)

    # ---- Full-batch training (original) ----

    def _train_step_full(self) -> float:
        import time as _time
        self.model.train()
        self.optimizer.zero_grad()
        train_data = self._train_data

        t0 = _time.time()
        print("[DEBUG] full: forward pass...", end=" ", flush=True)
        # Extract only positive edges from the split (edge_label_index contains both pos and neg)
        pos_mask = train_data.edge_label == 1
        pos_src = train_data.edge_label_index[0][pos_mask]
        pos_dst = train_data.edge_label_index[1][pos_mask]

        # Fresh negative samples each epoch — sample against ALL positive edges
        neg_edge = negative_sampling(
            edge_index=self._all_pos_ei,
            num_nodes=train_data.num_nodes,
            num_neg_samples=pos_src.size(0),
        )
        neg_src, neg_dst = neg_edge[0], neg_edge[1]

        src = torch.cat([pos_src, neg_src])
        dst = torch.cat([pos_dst, neg_dst])
        labels = torch.cat([
            torch.ones(pos_src.size(0), device=self.device),
            torch.zeros(neg_src.size(0), device=self.device),
        ])

        if hasattr(self.model, 'decode_logits'):
            z = self.model.encode(train_data)
            logits = self.model.decode_logits(z, src, dst)
            print(f"{_time.time()-t0:.1f}s", flush=True)
            t0 = _time.time()
            print("[DEBUG] full: loss + backward...", end=" ", flush=True)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        else:
            preds = self.model(train_data, src, dst)
            print(f"{_time.time()-t0:.1f}s", flush=True)
            t0 = _time.time()
            print("[DEBUG] full: loss + backward...", end=" ", flush=True)
            loss = nn.functional.binary_cross_entropy(preds, labels)

        if self._has_kl:
            loss = loss + self.model.beta * self.model.kl_loss()

        loss.backward()
        self.optimizer.step()
        print(f"{_time.time()-t0:.1f}s", flush=True)
        return loss.item()

    # ---- Mini-batch training (edge chunking) ----
    # Instead of LinkNeighborLoader (requires pyg-lib), we:
    # 1. Compute full node embeddings once per epoch (full-graph forward)
    # 2. Chunk the edge scoring into mini-batches
    # This saves memory on the decode + backward step, not the encode step.
    # For the encode step, we use torch.no_grad() caching + gradient checkpointing.

    def _train_step_mini(self) -> float:
        import time as _time
        self.model.train()
        train_data = self._train_data
        bs = self.config.batch_size

        # GNN/Transformer/VAE: encode with edge dropout, decode in batches
        # Edge dropout: randomly mask edges during encode so the model can't memorize
        t0 = _time.time()
        dropout = self.config.edge_dropout
        if dropout > 0:
            mask = torch.rand(train_data.edge_index.size(1), device=self.device) > dropout
            dropped_ei = train_data.edge_index[:, mask]
            dropped_data = Data(x=train_data.x, edge_index=dropped_ei, num_nodes=train_data.num_nodes)
            for attr in ("gwas_token_ids", "gwas_scores", "gwas_cat_ids",
                         "gwas_vocab_size", "gwas_num_categories"):
                if hasattr(train_data, attr):
                    setattr(dropped_data, attr, getattr(train_data, attr))
            kept = mask.sum().item()
            total_ei = train_data.edge_index.size(1)
            print(f"[DEBUG] mini: encoding with edge dropout={dropout} ({kept}/{total_ei} edges)...", end=" ", flush=True)
        else:
            dropped_data = train_data
            print("[DEBUG] mini: encoding full graph...", end=" ", flush=True)

        z = self.model.encode(dropped_data).detach().requires_grad_(True)
        print(f"{_time.time()-t0:.1f}s (z shape: {z.shape})", flush=True)

        # Extract only positive edges (edge_label_index contains both pos and neg from split)
        pos_mask = train_data.edge_label == 1
        pos_src = train_data.edge_label_index[0][pos_mask]
        pos_dst = train_data.edge_label_index[1][pos_mask]
        neg_edge = negative_sampling(
            edge_index=self._all_pos_ei,
            num_nodes=train_data.num_nodes,
            num_neg_samples=pos_src.size(0),
        )
        all_src = torch.cat([pos_src, neg_edge[0]])
        all_dst = torch.cat([pos_dst, neg_edge[1]])
        all_labels = torch.cat([
            torch.ones(pos_src.size(0), device=self.device),
            torch.zeros(neg_edge[0].size(0), device=self.device),
        ])

        # Shuffle edges
        perm = torch.randperm(all_src.size(0), device=all_src.device)
        all_src, all_dst, all_labels = all_src[perm], all_dst[perm], all_labels[perm]

        total_loss = 0.0
        num_batches = (all_src.size(0) + bs - 1) // bs

        batch_iter = range(0, all_src.size(0), bs)
        batch_pbar = tqdm(batch_iter, desc=f"  Batches", leave=False,
                          total=num_batches, unit="batch")

        # Accumulate gradients on z across batches
        if z.grad is not None:
            z.grad.zero_()

        for start in batch_pbar:
            end = min(start + bs, all_src.size(0))
            batch_src = all_src[start:end]
            batch_dst = all_dst[start:end]
            batch_labels = all_labels[start:end]

            if hasattr(self.model, 'decode_logits'):
                logits = self.model.decode_logits(z, batch_src, batch_dst)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_labels)
            else:
                preds = self.model.decode(z, batch_src, batch_dst)
                loss = nn.functional.binary_cross_entropy(preds, batch_labels)
            loss.backward()

            total_loss += loss.item()
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        batch_pbar.close()

        # Encoder backward using same dropped edges (consistent with forward)
        print("[DEBUG] mini: encoder backward...", end=" ", flush=True)
        t0 = _time.time()
        self.optimizer.zero_grad()
        z_fresh = self.model.encode(dropped_data)
        if self._has_kl:
            kl = self.model.beta * self.model.kl_loss()
            kl.backward(retain_graph=True)
        z_fresh.backward(z.grad)
        self.optimizer.step()
        print(f"{_time.time()-t0:.1f}s", flush=True)

        return total_loss / max(num_batches, 1)

    def _eval_auc_mini(self, split_data) -> float:
        self.model.eval()
        bs = self.config.batch_size

        with torch.no_grad():
            edge_label_index = split_data.edge_label_index
            src, dst = edge_label_index[0], edge_label_index[1]
            labels = split_data.edge_label.float()

            z = self.model.encode(split_data)
            all_scores = []
            for start in range(0, src.size(0), bs):
                end = min(start + bs, src.size(0))
                scores = self.model.decode(z, src[start:end], dst[start:end])
                all_scores.append(scores)
            all_scores = torch.cat(all_scores).cpu().numpy()

            labels = labels.cpu().numpy()

        if len(set(labels)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(labels, all_scores))
        except ValueError:
            return 0.5

    # ---- Full-batch eval (original) ----

    def _eval_auc_full(self, split_data) -> float:
        self.model.eval()
        with torch.no_grad():
            edge_label_index = split_data.edge_label_index
            src, dst = edge_label_index[0], edge_label_index[1]
            labels = split_data.edge_label.float().cpu().numpy()

            if hasattr(self.model, 'decode_logits'):
                z = self.model.encode(split_data)
                scores = torch.sigmoid(self.model.decode_logits(z, src, dst)).cpu().numpy()
            else:
                scores = self.model(split_data, src, dst).cpu().numpy()

        if len(set(labels)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            return 0.5

    # ---- Main train/evaluate ----

    def train(self, data, on_epoch=None) -> dict:
        import time as _time
        t0 = _time.time()
        print(f"[DEBUG] Splitting data ({data.num_nodes} nodes, {data.edge_index.size(1)} edges)...")
        self._split_data(data)
        print(f"[DEBUG] Split done in {_time.time()-t0:.1f}s — train: {self._train_data.edge_index.size(1)} edges, "
              f"val: {self._val_data.edge_label_index.size(1)} labels, "
              f"test: {self._test_data.edge_label_index.size(1)} labels")
        print(f"[DEBUG] Device: {self.device}, mini_batch: {self.config.mini_batch}, bs: {self.config.batch_size}")

        # Precompute RWSE once if the model supports it (graph structure is fixed)
        if hasattr(self.model, 'precompute_rwse'):
            self.model.precompute_rwse(self._train_data)

        use_mini = self.config.mini_batch
        train_step = self._train_step_mini if use_mini else self._train_step_full
        eval_auc = self._eval_auc_mini if use_mini else self._eval_auc_full

        history = {"train_loss": [], "val_auc": []}
        best_val = 0.0
        patience_counter = 0

        epoch_pbar = tqdm(range(self.config.epochs), desc="Epochs", unit="epoch")
        for epoch in epoch_pbar:
            train_loss = train_step()
            val_auc = eval_auc(self._val_data)

            history["train_loss"].append(train_loss)
            history["val_auc"].append(val_auc)

            epoch_pbar.set_postfix(loss=f"{train_loss:.4f}", val_auc=f"{val_auc:.4f}")

            if on_epoch:
                on_epoch(epoch, train_loss, val_auc)

            if self.config.early_stopping:
                if val_auc > best_val:
                    best_val = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        epoch_pbar.close()
                        break

        epoch_pbar.close()

        return history

    def evaluate(self, data) -> dict[str, float]:
        if self._test_data is None:
            self._split_data(data)

        if self.config.mini_batch:
            return self._evaluate_mini()
        return self._evaluate_full()

    def _evaluate_full(self) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            edge_label_index = self._test_data.edge_label_index
            src, dst = edge_label_index[0], edge_label_index[1]
            labels = self._test_data.edge_label.float().cpu().numpy()

            if hasattr(self.model, 'decode_logits'):
                z = self.model.encode(self._test_data)
                scores = torch.sigmoid(self.model.decode_logits(z, src, dst)).cpu().numpy()
            else:
                scores = self.model(self._test_data, src, dst).cpu().numpy()

        if len(set(labels)) < 2:
            return {"auc_roc": 0.5, "avg_precision": float(labels.mean())}

        return {
            "auc_roc": float(roc_auc_score(labels, scores)),
            "avg_precision": float(average_precision_score(labels, scores)),
        }

    def _evaluate_mini(self) -> dict[str, float]:
        bs = self.config.batch_size
        test_data = self._test_data

        self.model.eval()
        with torch.no_grad():
            src = test_data.edge_label_index[0]
            dst = test_data.edge_label_index[1]
            labels = test_data.edge_label.float()

            z = self.model.encode(test_data)
            all_scores = []
            for start in range(0, src.size(0), bs):
                end = min(start + bs, src.size(0))
                scores = self.model.decode(z, src[start:end], dst[start:end])
                all_scores.append(scores)
            all_scores = torch.cat(all_scores).cpu().numpy()

            labels = labels.cpu().numpy()

        if len(set(labels)) < 2:
            return {"auc_roc": 0.5, "avg_precision": float(labels.mean())}

        return {
            "auc_roc": float(roc_auc_score(labels, all_scores)),
            "avg_precision": float(average_precision_score(labels, all_scores)),
        }
