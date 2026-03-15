# src/models/trainer.py
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 0.001
    batch_size: int = 64
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    early_stopping: bool = False
    patience: int = 10


class Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig):
        self.model = model
        self.config = config
        self._is_vae = hasattr(model, "loss") and hasattr(model, "predict_links")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self._train_data = None
        self._val_data = None
        self._test_data = None

    def _split_data(self, data):
        test_ratio = max(0.01, 1.0 - self.config.train_ratio - self.config.val_ratio)
        transform = RandomLinkSplit(
            num_val=self.config.val_ratio,
            num_test=test_ratio,
            add_negative_train_samples=False,
        )
        self._train_data, self._val_data, self._test_data = transform(data)

    def _train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        train_data = self._train_data

        if self._is_vae:
            loss = self.model.loss(train_data.x, train_data.edge_index)
        else:
            pos_edge = train_data.edge_label_index
            pos_src, pos_dst = pos_edge[0], pos_edge[1]

            neg_edge = negative_sampling(
                edge_index=train_data.edge_index,
                num_nodes=train_data.num_nodes,
                num_neg_samples=pos_src.size(0),
            )
            neg_src, neg_dst = neg_edge[0], neg_edge[1]

            src = torch.cat([pos_src, neg_src])
            dst = torch.cat([pos_dst, neg_dst])
            labels = torch.cat([
                torch.ones(pos_src.size(0)),
                torch.zeros(neg_src.size(0)),
            ])

            preds = self.model(train_data, src, dst)
            loss = nn.functional.binary_cross_entropy(preds, labels)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _eval_auc(self, split_data) -> float:
        self.model.eval()
        with torch.no_grad():
            edge_label_index = split_data.edge_label_index
            src, dst = edge_label_index[0], edge_label_index[1]
            labels = split_data.edge_label.float().numpy()

            if self._is_vae:
                scores = self.model.predict_links(split_data.x, split_data.edge_index, src, dst).numpy()
            else:
                scores = self.model(split_data, src, dst).numpy()

        if len(set(labels)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            return 0.5

    def train(self, data, on_epoch=None) -> dict:
        self._split_data(data)

        history = {"train_loss": [], "val_auc": []}
        best_val = 0.0
        patience_counter = 0

        for epoch in range(self.config.epochs):
            train_loss = self._train_step()
            val_auc = self._eval_auc(self._val_data)

            history["train_loss"].append(train_loss)
            history["val_auc"].append(val_auc)

            if on_epoch:
                on_epoch(epoch, train_loss, val_auc)

            if self.config.early_stopping:
                if val_auc > best_val:
                    best_val = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        break

        return history

    def evaluate(self, data) -> dict[str, float]:
        if self._test_data is None:
            self._split_data(data)

        self.model.eval()
        with torch.no_grad():
            edge_label_index = self._test_data.edge_label_index
            src, dst = edge_label_index[0], edge_label_index[1]
            labels = self._test_data.edge_label.float().numpy()

            if self._is_vae:
                scores = self.model.predict_links(
                    self._test_data.x, self._test_data.edge_index, src, dst
                ).numpy()
            else:
                scores = self.model(self._test_data, src, dst).numpy()

        if len(set(labels)) < 2:
            return {"auc_roc": 0.5, "avg_precision": float(labels.mean())}

        return {
            "auc_roc": float(roc_auc_score(labels, scores)),
            "avg_precision": float(average_precision_score(labels, scores)),
        }
