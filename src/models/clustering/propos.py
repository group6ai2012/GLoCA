from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.utils import SafeBatchNorm1d


class ProPosMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            SafeBatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProPosHead(nn.Module):
    training_mode = "contrastive_two_view"

    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        projection_dim: int = 256,
        projection_hidden_dim: int = 4096,
        predictor_hidden_dim: int = 4096,
        temperature: float = 0.5,
        sigma: float = 0.001,
        ema_momentum: float = 0.996,
    ) -> None:
        super().__init__()
        if isinstance(n_clusters, str):
            raise ValueError("ProPosHead requires resolved integer n_clusters, got 'auto'")
        self.n_clusters = int(n_clusters)
        self.embedding_dim = int(embedding_dim)
        self.projection_dim = int(projection_dim)
        self.temperature = float(temperature)
        self.sigma = float(sigma)
        self.ema_momentum = float(ema_momentum)

        self.projector = ProPosMLP(self.embedding_dim, int(projection_hidden_dim), self.projection_dim)
        self.target_projector = copy.deepcopy(self.projector)
        self.predictor = ProPosMLP(self.projection_dim, int(predictor_hidden_dim), self.projection_dim)
        self._freeze_target_projector()
        self.register_buffer("pseudo_labels", torch.empty(0, dtype=torch.long), persistent=True)

    def _freeze_target_projector(self) -> None:
        for parameter in self.target_projector.parameters():
            parameter.requires_grad = False

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_online(embedding)

    def forward_online(
        self,
        embedding: torch.Tensor,
        attention: torch.Tensor | None = None,
        patch_grid: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        projected = F.normalize(self.projector(embedding), dim=-1)
        predicted = F.normalize(self.predictor(projected), dim=-1)
        return {
            "embedding": embedding,
            "projected": projected,
            "predicted": predicted,
            "attention": attention,
            "patch_grid": patch_grid,
        }

    @torch.no_grad()
    def forward_target(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.target_projector(embedding), dim=-1)

    def positive_sampling_alignment(
        self,
        online_projected: torch.Tensor,
        target_projected: torch.Tensor,
        sigma: float | None = None,
    ) -> torch.Tensor:
        sigma = self.sigma if sigma is None else float(sigma)
        sampled = online_projected + sigma * torch.randn_like(online_projected)
        predicted = self.predictor(sampled)
        return -2.0 * F.cosine_similarity(
            F.normalize(predicted, dim=-1),
            F.normalize(target_projected.detach(), dim=-1),
            dim=-1,
        ).mean()

    def compute_centers(self, features: torch.Tensor, pseudo_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels = pseudo_labels.long()
        centers = features.new_zeros(self.n_clusters, features.shape[1])
        counts = torch.bincount(labels, minlength=self.n_clusters).to(features.device, dtype=features.dtype)
        centers.index_add_(0, labels, features)
        nonempty = counts > 0
        centers[nonempty] = centers[nonempty] / counts[nonempty].unsqueeze(1)
        centers = F.normalize(centers, dim=1)
        return centers, nonempty

    def prototype_scattering_loss(
        self,
        online_projected: torch.Tensor,
        target_projected: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        online_centers, nonempty = self.compute_centers(online_projected, pseudo_labels)
        target_centers, _ = self.compute_centers(target_projected.detach(), pseudo_labels)
        if int(nonempty.sum().item()) < 2:
            return online_projected.new_zeros(()), False

        logits = online_centers.mm(online_centers.T) / self.temperature
        positives = (online_centers * target_centers).sum(dim=1) / self.temperature
        rows = torch.arange(self.n_clusters, device=online_projected.device)
        logits[rows, rows] = positives
        logits[:, ~nonempty] = -10.0

        log_prob = logits.log_softmax(dim=1)
        losses = -log_prob.diagonal()
        return losses[nonempty].mean(), True

    @torch.no_grad()
    def update_target_projector(self, momentum: float | None = None) -> None:
        momentum = self.ema_momentum if momentum is None else float(momentum)
        for online, target in zip(self.projector.parameters(), self.target_projector.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
        for online_buffer, target_buffer in zip(self.projector.buffers(), self.target_projector.buffers()):
            target_buffer.copy_(online_buffer)

    def set_pseudo_labels(self, pseudo_labels: torch.Tensor) -> None:
        self.pseudo_labels = pseudo_labels.detach().long().to(self.pseudo_labels.device)

    def batch_pseudo_labels(self, indices: torch.Tensor) -> torch.Tensor:
        if self.pseudo_labels.numel() == 0:
            raise RuntimeError("ProPos pseudo-labels are empty. Run the E-step before training.")
        lookup_indices = indices.to(self.pseudo_labels.device).long()
        if bool((lookup_indices < 0).any()) or bool((lookup_indices >= self.pseudo_labels.numel()).any()):
            raise IndexError("ProPos batch indices are outside the pseudo-label table.")
        return self.pseudo_labels[lookup_indices].to(indices.device)
