# src/models/trainer.py
from dataclasses import dataclass, field
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    patience: int = 10
    val_ratio: float = 0.1
    test_ratio: float = 0.1


class Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig):
        self.model = model
        self.config = config
        self._is_vae = hasattr(model, "loss") and callable(getattr(model, "loss"))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    def _split_data(self, data):
        transform = RandomLinkSplit(
            num_val=self.config.val_ratio,
            num_test=self.config.test_ratio,
            add_negative_train_samples=False,
        )
        return transform(data)

    def _train_step(self, train_data) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        if self._is_vae:
            loss = self.model.loss(train_data.x, train_data.edge_index)
        else:
            # Positive edges from edge_label_index
            pos_edge = train_data.edge_label_index
            pos_src, pos_dst = pos_edge[0], pos_edge[1]

            # Sample negatives
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

    def _val_loss(self, val_data) -> float:
        self.model.eval()
        with torch.no_grad():
            if self._is_vae:
                return self.model.loss(val_data.x, val_data.edge_index).item()
            else:
                edge_label_index = val_data.edge_label_index
                src, dst = edge_label_index[0], edge_label_index[1]
                labels = val_data.edge_label.float()
                preds = self.model(val_data, src, dst)
                return nn.functional.binary_cross_entropy(preds, labels).item()

    def train(self, data) -> List[Dict[str, Any]]:
        train_data, val_data, _ = self._split_data(data)

        history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            train_loss = self._train_step(train_data)
            val_loss = self._val_loss(val_data)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        return history

    def evaluate(self, data) -> Dict[str, float]:
        _, _, test_data = self._split_data(data)
        self.model.eval()
        with torch.no_grad():
            if self._is_vae:
                edge_label_index = test_data.edge_label_index
                src, dst = edge_label_index[0], edge_label_index[1]
                labels = test_data.edge_label.float().numpy()
                scores = self.model.predict_links(test_data.x, test_data.edge_index, src, dst).numpy()
            else:
                edge_label_index = test_data.edge_label_index
                src, dst = edge_label_index[0], edge_label_index[1]
                labels = test_data.edge_label.float().numpy()
                scores = self.model(test_data, src, dst).numpy()

        # Guard against degenerate label sets (all same class)
        if len(set(labels)) < 2:
            return {"auc_roc": 0.5, "avg_precision": float(labels.mean())}

        return {
            "auc_roc": float(roc_auc_score(labels, scores)),
            "avg_precision": float(average_precision_score(labels, scores)),
        }
