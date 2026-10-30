# src/models/trainer.py
import copy
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import degree, negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Per-node tensors / scalars that travel unchanged with every split.
_NODE_ATTRS = (
    "gwas_token_ids", "gwas_scores", "gwas_cat_ids",
    "gwas_vocab_size", "gwas_num_categories", "degree_col",
)


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0
    warmup_epochs: int = 0
    batch_size: int = 512
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    early_stopping: bool = False
    patience: int = 10
    mini_batch: bool = False
    edge_dropout: float = 0.3  # fraction of message edges dropped per step (mini_batch mode only)
    device: str = "cpu"  # "cpu", "mps", or "cuda"
    seed: int | None = 42  # seeds the split, negatives, dropout and noise; None = fresh randomness

    def __post_init__(self):
        if not 0 < self.train_ratio < 1 or not 0 < self.val_ratio < 1:
            raise ValueError("train_ratio and val_ratio must both be strictly between 0 and 1")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError(
                f"train_ratio + val_ratio must be below 1 to leave test edges "
                f"(got {self.train_ratio} + {self.val_ratio})"
            )


class Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig):
        self.config = config
        self._has_kl = hasattr(model, "kl_loss")

        self.device = torch.device(config.device)
        self.model = model.to(self.device)

        self._base_lr = config.lr
        optimizer_cls = torch.optim.AdamW if config.weight_decay > 0 else torch.optim.Adam
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self._train_data = None
        self._val_data = None
        self._test_data = None
        self._all_pos_ei = None

    # ---- Randomness -------------------------------------------------------

    def _seed_everything(self):
        if self.config.seed is None:
            return
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)  # also seeds CUDA / MPS generators

    def _generator(self) -> torch.Generator:
        gen = torch.Generator()
        if self.config.seed is None:
            gen.seed()
        else:
            gen.manual_seed(self.config.seed)
        return gen

    def _rng_state(self) -> dict:
        state = {"cpu": torch.get_rng_state()}
        if self.device.type == "cuda":
            state["device"] = torch.cuda.get_rng_state(self.device)
        elif self.device.type == "mps":
            state["device"] = torch.mps.get_rng_state()
        return state

    def _set_rng_state(self, state: dict):
        torch.set_rng_state(state["cpu"])
        if "device" in state:
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(state["device"], self.device)
            else:
                torch.mps.set_rng_state(state["device"])

    def _set_epoch_lr(self, epoch: int):
        if self.config.warmup_epochs <= 0:
            lr = self._base_lr
        else:
            lr = self._base_lr * min((epoch + 1) / self.config.warmup_epochs, 1.0)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # ---- Split --------------------------------------------------------------

    def _split_data(self, data):
        """Edge-level split into message-passing / supervision / val / test.

        Supervision edges never appear in the message-passing graph, otherwise the
        model learns to detect "this edge is in my graph", which does not transfer
        to held-out edges. Negatives are sampled against *all* real edges so no
        val/test edge is ever labelled negative.
        """
        self._seed_everything()
        cfg = self.config
        num_nodes = data.num_nodes
        ei = data.edge_index.cpu()
        edge_attr = getattr(data, "edge_attr", None)
        edge_attr = edge_attr.cpu() if edge_attr is not None else None

        # Unique undirected edges, keeping each edge's first occurrence so its
        # edge_attr row stays attached to it.
        lo, hi = torch.minimum(ei[0], ei[1]), torch.maximum(ei[0], ei[1])
        keys = lo * num_nodes + hi
        _, inverse = torch.unique(keys, return_inverse=True)
        n = int(inverse.max().item()) + 1 if keys.numel() else 0
        first = torch.full((n,), keys.numel(), dtype=torch.long)
        first.scatter_reduce_(0, inverse, torch.arange(keys.numel()), reduce="amin")
        edges = torch.stack([lo[first], hi[first]])  # [2, n]
        attrs = edge_attr[first] if edge_attr is not None else None  # [n, F]

        perm = torch.randperm(n, generator=self._generator())
        n_train = int(n * cfg.train_ratio)
        n_val = int(n * cfg.val_ratio)
        n_test = n - n_train - n_val
        n_msg = int(n_train * 0.85)
        n_sup = n_train - n_msg
        if min(n_sup, n_val, n_test) <= 0:
            raise ValueError(
                f"Graph has {n} unique edges, which leaves an empty split "
                f"(train supervision={n_sup}, val={n_val}, test={n_test}). "
                "Use a larger graph or different split ratios."
            )
        msg_idx = perm[:n_msg]
        sup_idx = perm[n_msg:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        def undirected(idx):
            e = edges[:, idx]
            ei_u = torch.cat([e, e.flip(0)], dim=1)
            attr_u = torch.cat([attrs[idx], attrs[idx]], dim=0) if attrs is not None else None
            return ei_u, attr_u

        all_pos_ei, _ = undirected(torch.arange(n))
        msg_ei, msg_attr = undirected(msg_idx)

        # Topology-derived node features must only see the message-passing graph,
        # otherwise held-out edges leak into the inputs through their endpoints.
        x = data.x
        degree_col = getattr(data, "degree_col", None)
        if degree_col is not None:
            deg = degree(msg_ei[0], num_nodes, dtype=torch.float)
            x = data.x.clone()
            x[:, degree_col] = (deg / deg.max().clamp(min=1.0)).to(x.device, x.dtype)

        def make_split(idx):
            pos = edges[:, idx]
            neg = negative_sampling(all_pos_ei, num_nodes=num_nodes, num_neg_samples=pos.size(1)).long()
            split = Data(
                x=x, edge_index=msg_ei, num_nodes=num_nodes,
                edge_label_index=torch.cat([pos, neg], dim=1),
                edge_label=torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]),
            )
            if msg_attr is not None:
                split.edge_attr = msg_attr
            for attr in _NODE_ATTRS:
                if hasattr(data, attr):
                    setattr(split, attr, getattr(data, attr))
            return split.to(self.device)

        self._train_data = make_split(sup_idx)
        self._val_data = make_split(val_idx)
        self._test_data = make_split(test_idx)
        self._all_pos_ei = all_pos_ei.to(self.device)

    # ---- Shared helpers ----------------------------------------------------

    def _supervision_edges(self, train_data):
        """Positive supervision edges plus fresh negatives for this step."""
        pos_mask = train_data.edge_label == 1
        pos_src = train_data.edge_label_index[0][pos_mask]
        pos_dst = train_data.edge_label_index[1][pos_mask]
        neg = negative_sampling(
            edge_index=self._all_pos_ei,
            num_nodes=train_data.num_nodes,
            num_neg_samples=pos_src.size(0),
        )
        src = torch.cat([pos_src, neg[0]])
        dst = torch.cat([pos_dst, neg[1]])
        labels = torch.cat([
            torch.ones(pos_src.size(0), device=self.device),
            torch.zeros(neg.size(1), device=self.device),
        ])
        return src, dst, labels

    def _edge_loss(self, z, src, dst, labels):
        if hasattr(self.model, "decode_logits"):
            return nn.functional.binary_cross_entropy_with_logits(self.model.decode_logits(z, src, dst), labels)
        return nn.functional.binary_cross_entropy(self.model.decode(z, src, dst), labels)

    def _apply_edge_dropout(self, train_data):
        dropout = self.config.edge_dropout
        if dropout <= 0:
            return train_data
        mask = torch.rand(train_data.edge_index.size(1), device=self.device) > dropout
        dropped = Data(x=train_data.x, edge_index=train_data.edge_index[:, mask], num_nodes=train_data.num_nodes)
        edge_attr = getattr(train_data, "edge_attr", None)
        if edge_attr is not None:
            dropped.edge_attr = edge_attr[mask]
        for attr in _NODE_ATTRS:
            if hasattr(train_data, attr):
                setattr(dropped, attr, getattr(train_data, attr))
        return dropped

    # ---- Full-batch training ------------------------------------------------

    def _train_step_full(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        train_data = self._train_data
        src, dst, labels = self._supervision_edges(train_data)

        if hasattr(self.model, "decode_logits"):
            z = self.model.encode(train_data)
            loss = self._edge_loss(z, src, dst, labels)
        else:
            loss = nn.functional.binary_cross_entropy(self.model(train_data, src, dst), labels)
        if self._has_kl:
            loss = loss + self.model.beta * self.model.kl_loss()

        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ---- Mini-batch training (edge chunking) -------------------------------
    # The encoder runs on the full message-passing graph; only the edge scoring is
    # chunked. Pass 1 produces detached embeddings so decoder gradients accumulate
    # in z.grad without holding the encoder graph; pass 2 replays the *same*
    # stochastic forward pass (RNG state restored, so dropout masks, VAE noise and
    # probes are identical) and back-propagates z.grad through the encoder once.

    def _train_step_mini(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        train_data = self._train_data
        bs = self.config.batch_size
        dropped_data = self._apply_edge_dropout(train_data)

        rng_before = self._rng_state()
        z = self.model.encode(dropped_data).detach().requires_grad_(True)

        src, dst, labels = self._supervision_edges(train_data)
        total = src.size(0)
        if total == 0:
            return 0.0
        perm = torch.randperm(total, device=src.device)
        src, dst, labels = src[perm], dst[perm], labels[perm]

        num_batches = (total + bs - 1) // bs
        batch_pbar = tqdm(range(0, total, bs), desc="  Batches", leave=False, total=num_batches, unit="batch")
        recon_loss = 0.0
        for start in batch_pbar:
            end = min(start + bs, total)
            # Weight each batch mean by its share so the sum equals the mean over all edges.
            loss = self._edge_loss(z, src[start:end], dst[start:end], labels[start:end]) * ((end - start) / total)
            loss.backward()
            recon_loss += loss.item()
            batch_pbar.set_postfix(loss=f"{loss.item() * total / (end - start):.4f}")
        batch_pbar.close()

        rng_after = self._rng_state()
        self._set_rng_state(rng_before)
        z_fresh = self.model.encode(dropped_data)
        self._set_rng_state(rng_after)

        total_loss = recon_loss
        if self._has_kl:
            kl = self.model.beta * self.model.kl_loss()
            kl.backward(retain_graph=True)
            total_loss += kl.item()
        z_fresh.backward(z.grad)
        self.optimizer.step()
        return total_loss

    # ---- Evaluation ---------------------------------------------------------

    def _scores_and_labels(self, split_data, batched: bool):
        self.model.eval()
        with torch.no_grad():
            src, dst = split_data.edge_label_index[0], split_data.edge_label_index[1]
            labels = split_data.edge_label.float().cpu().numpy()
            if batched or hasattr(self.model, "decode_logits"):
                z = self.model.encode(split_data)
                if batched:
                    bs = self.config.batch_size
                    scores = torch.cat([
                        self.model.decode(z, src[s:s + bs], dst[s:s + bs]) for s in range(0, src.size(0), bs)
                    ])
                else:
                    scores = torch.sigmoid(self.model.decode_logits(z, src, dst))
            else:
                scores = self.model(split_data, src, dst)
        return scores.cpu().numpy(), labels

    def _eval_auc(self, split_data, batched: bool) -> float:
        scores, labels = self._scores_and_labels(split_data, batched)
        if len(set(labels)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            return 0.5

    def _eval_auc_mini(self, split_data) -> float:
        return self._eval_auc(split_data, batched=True)

    def _eval_auc_full(self, split_data) -> float:
        return self._eval_auc(split_data, batched=False)

    def _evaluate(self, batched: bool) -> dict[str, float]:
        scores, labels = self._scores_and_labels(self._test_data, batched)
        if len(set(labels)) < 2:
            return {"auc_roc": 0.5, "avg_precision": float(labels.mean())}
        return {
            "auc_roc": float(roc_auc_score(labels, scores)),
            "avg_precision": float(average_precision_score(labels, scores)),
        }

    def _evaluate_full(self) -> dict[str, float]:
        return self._evaluate(batched=False)

    def _evaluate_mini(self) -> dict[str, float]:
        return self._evaluate(batched=True)

    # ---- Main train/evaluate --------------------------------------------------

    def train(self, data, on_epoch=None) -> dict:
        print(f"[DEBUG] Splitting data ({data.num_nodes} nodes, {data.edge_index.size(1)} edges)...")
        self._split_data(data)
        print(f"[DEBUG] Split done — train: {self._train_data.edge_index.size(1)} message edges, "
              f"{int((self._train_data.edge_label == 1).sum())} supervision edges, "
              f"val: {self._val_data.edge_label_index.size(1)} labels, "
              f"test: {self._test_data.edge_label_index.size(1)} labels")
        print(f"[DEBUG] Device: {self.device}, mini_batch: {self.config.mini_batch}, "
              f"bs: {self.config.batch_size}, seed: {self.config.seed}")

        # Precompute RWSE once if the model supports it (message graph is fixed)
        if hasattr(self.model, "precompute_rwse"):
            self.model.precompute_rwse(self._train_data)

        use_mini = self.config.mini_batch
        train_step = self._train_step_mini if use_mini else self._train_step_full
        eval_auc = self._eval_auc_mini if use_mini else self._eval_auc_full

        history = {"train_loss": [], "val_auc": [], "best_epoch": 0}
        best_val = -1.0
        best_state = None
        patience_counter = 0

        epoch_pbar = tqdm(range(self.config.epochs), desc="Epochs", unit="epoch")
        for epoch in epoch_pbar:
            self._set_epoch_lr(epoch)
            train_loss = train_step()
            val_auc = eval_auc(self._val_data)

            history["train_loss"].append(train_loss)
            history["val_auc"].append(val_auc)
            epoch_pbar.set_postfix(loss=f"{train_loss:.4f}", val_auc=f"{val_auc:.4f}")

            if on_epoch:
                on_epoch(epoch, train_loss, val_auc)

            if val_auc > best_val:
                best_val = val_auc
                history["best_epoch"] = epoch
                patience_counter = 0
                if self.config.early_stopping:
                    best_state = copy.deepcopy(self.model.state_dict())
            elif self.config.early_stopping:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break
        epoch_pbar.close()

        if best_state is not None:
            # Evaluate and save the model that was best on validation, not the one
            # that had already degraded for `patience` epochs.
            self.model.load_state_dict(best_state)

        return history

    def evaluate(self, data) -> dict[str, float]:
        if self._test_data is None:
            self._split_data(data)
        if self.config.mini_batch:
            return self._evaluate_mini()
        return self._evaluate_full()
