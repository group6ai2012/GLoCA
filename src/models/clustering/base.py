from __future__ import annotations

from typing import Protocol

import torch


class ClusteringHead(Protocol):
    n_clusters: int
    training_mode: str
    embedding_dim: int

    def forward(self, embedding: torch.Tensor) -> dict: ...

    def loss(self, outputs: dict | tuple[dict, dict], batch: dict) -> torch.Tensor: ...

    def predict(self, embedding: torch.Tensor) -> torch.Tensor: ...

    def on_fit_start(self, trainer, pl_module) -> None: ...

    def on_train_epoch_start(self, trainer, pl_module) -> None: ...

