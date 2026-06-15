from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.clustering.kmeans import fit_kmeans
from src.models.utils import SafeBatchNorm1d


class CDCMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_clusters: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.bn1 = SafeBatchNorm1d(int(hidden_dim))
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(int(hidden_dim), int(n_clusters))

    def hidden(self, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.linear1(embedding)
        return self.relu(self.bn1(hidden))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.hidden(embedding))

    @torch.no_grad()
    def copy_prototype_weights(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        if self.linear1.weight.shape != w1.shape:
            raise ValueError(
                f"W1 shape mismatch: expected {tuple(self.linear1.weight.shape)}, got {tuple(w1.shape)}"
            )
        if self.linear2.weight.shape != w2.shape:
            raise ValueError(
                f"W2 shape mismatch: expected {tuple(self.linear2.weight.shape)}, got {tuple(w2.shape)}"
            )
        self.linear1.weight.copy_(
            w1.to(device=self.linear1.weight.device, dtype=self.linear1.weight.dtype)
        )
        self.linear2.weight.copy_(
            w2.to(device=self.linear2.weight.device, dtype=self.linear2.weight.dtype)
        )
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)


class CDCHead(nn.Module):
    def __init__(self, input_dim: int, n_clusters: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_clusters = int(n_clusters)
        self.hidden_dim = int(hidden_dim)
        self.clustering_head = CDCMLP(self.input_dim, self.hidden_dim, self.n_clusters)
        self.calibration_head = CDCMLP(self.input_dim, self.hidden_dim, self.n_clusters)

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        clustering_logits = self.clustering_head(embedding)
        calibration_logits = self.calibration_head(embedding)
        clustering_probabilities = F.softmax(clustering_logits, dim=-1)
        calibration_probabilities = F.softmax(calibration_logits, dim=-1)
        clustering_confidences, clustering_predictions = clustering_probabilities.max(
            dim=-1
        )
        calibrated_confidences, calibrated_predictions = calibration_probabilities.max(
            dim=-1
        )
        return {
            "clustering_logits": clustering_logits,
            "calibration_logits": calibration_logits,
            "clustering_probabilities": clustering_probabilities,
            "calibration_probabilities": calibration_probabilities,
            "predictions": calibrated_predictions,
            "clustering_predictions": clustering_predictions,
            "calibration_predictions": calibrated_predictions,
            "confidences": calibrated_confidences,
            "clustering_confidences": clustering_confidences,
            "calibrated_confidences": calibrated_confidences,
        }

    @torch.no_grad()
    def initialize_from_embeddings(
        self,
        embeddings: torch.Tensor,
        *,
        kmeans_init: str = "kmeans++",
        kmeans_n_init: int = 2,
        kmeans_max_iter: int = 50,
        kmeans_tol: float = 1.0e-4,
        seed: int = 0,
        orthogonalize: bool = False,
    ) -> dict[str, Any]:
        del orthogonalize
        features = torch.nan_to_num(
            embeddings.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0
        )
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            return {
                "cdc_init_mode": "random",
                "prototype_init_used": False,
                "orthogonalization_used": False,
                "fallback_reason": f"expected embeddings [N, {self.input_dim}], got {tuple(features.shape)}",
            }
        if features.shape[0] < self.hidden_dim:
            return {
                "cdc_init_mode": "random",
                "prototype_init_used": False,
                "orthogonalization_used": False,
                "fallback_reason": (
                    f"n_samples={features.shape[0]} is smaller than hidden_dim={self.hidden_dim}; "
                    "first-layer prototype K-Means cannot be initialized."
                ),
            }
        try:
            zscore = _row_zscore(features)
            first = fit_kmeans(
                zscore,
                self.hidden_dim,
                spherical=True,
                init=kmeans_init,
                n_init=kmeans_n_init,
                max_iter=kmeans_max_iter,
                tol=kmeans_tol,
                seed=seed,
                device=torch.device("cpu"),
            )
            w1 = first["centers"].float()
            hidden = F.linear(features, w1, bias=None)
            hidden = F.batch_norm(
                hidden,
                running_mean=torch.zeros(self.hidden_dim),
                running_var=torch.ones(self.hidden_dim),
                weight=torch.ones(self.hidden_dim),
                bias=torch.zeros(self.hidden_dim),
                training=True,
            )
            hidden = F.relu(hidden)
            if hidden.shape[0] < self.n_clusters:
                raise ValueError(
                    f"n_samples={hidden.shape[0]} is smaller than n_clusters={self.n_clusters}; "
                    "second-layer prototype K-Means cannot be initialized."
                )
            second = fit_kmeans(
                _row_zscore(hidden),
                self.n_clusters,
                spherical=True,
                init=kmeans_init,
                n_init=kmeans_n_init,
                max_iter=kmeans_max_iter,
                tol=kmeans_tol,
                seed=seed + 1,
                device=torch.device("cpu"),
            )
            w2 = second["centers"].float()
            self.clustering_head.copy_prototype_weights(w1, w2)
            self.calibration_head.copy_prototype_weights(w1, w2)
            return {
                "cdc_init_mode": "prototype_kmeans",
                "prototype_init_used": True,
                "orthogonalization_used": False,
                "fallback_reason": "",
                "first_kmeans_logs": first["logs"],
                "second_kmeans_logs": second["logs"],
            }
        except Exception as exc:
            return {
                "cdc_init_mode": "random",
                "prototype_init_used": False,
                "orthogonalization_used": False,
                "fallback_reason": str(exc),
            }


def _row_zscore(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=1, keepdim=True)
    scaled = centered / features.std(dim=1, keepdim=True).clamp_min(1.0e-6)
    return F.normalize(torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0), dim=1)
