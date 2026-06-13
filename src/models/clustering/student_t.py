from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.clustering.kmeans import fit_kmeans


class StudentTHead(nn.Module):
    training_mode = "single_view"

    def __init__(self, n_clusters: int, embedding_dim: int, alpha: float = 1.0, seed: int = 0) -> None:
        super().__init__()
        if isinstance(n_clusters, str):
            raise ValueError("StudentTHead requires resolved integer n_clusters, got 'auto'")
        self.n_clusters = int(n_clusters)
        self.embedding_dim = int(embedding_dim)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.cluster_centers = nn.Parameter(torch.empty(self.n_clusters, self.embedding_dim))
        nn.init.xavier_uniform_(self.cluster_centers)
        self.register_buffer("target_distribution", torch.empty(0), persistent=False)

    def forward(self, embedding: torch.Tensor) -> dict:
        dist_sq = torch.cdist(embedding, self.cluster_centers, p=2).pow(2)
        numerator = (1.0 + dist_sq / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        q = numerator / numerator.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return {"logits": q, "q": q}

    @staticmethod
    def target_from_q(q: torch.Tensor) -> torch.Tensor:
        weight = q.pow(2) / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        q = outputs["q"]
        if self.target_distribution.numel() == 0:
            p = self.target_from_q(q.detach())
        else:
            p = self.target_distribution[batch["index"].to(self.target_distribution.device)].to(q.device)
        return F.kl_div(q.clamp_min(1e-12).log(), p.detach(), reduction="batchmean")

    def predict(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.forward(embedding)["q"].argmax(dim=1)

    def on_fit_start(self, trainer, pl_module) -> None:
        embeddings, _ = self._collect_embeddings(trainer, pl_module)
        result = fit_kmeans(
            embeddings,
            self.n_clusters,
            spherical=False,
            init="kmeans++",
            n_init=10,
            max_iter=300,
            tol=1.0e-4,
            seed=self.seed,
            device=self.cluster_centers.device,
        )
        self.cluster_centers.data.copy_(result["centers"].to(self.cluster_centers.device))
        self._refresh_target_distribution(trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._refresh_target_distribution(trainer, pl_module)

    def _refresh_target_distribution(self, trainer, pl_module) -> None:
        embeddings, indices = self._collect_embeddings(trainer, pl_module)
        device = self.cluster_centers.device
        q_parts = []
        with torch.no_grad():
            for start in range(0, embeddings.shape[0], 4096):
                part = embeddings[start : start + 4096].to(device)
                q_parts.append(self.forward(part)["q"].cpu())
        q = torch.cat(q_parts, dim=0)
        p = self.target_from_q(q)
        target = torch.empty_like(p)
        target[indices] = p
        self.target_distribution = target.to(device)

    @staticmethod
    def _collect_embeddings(trainer, pl_module) -> tuple[torch.Tensor, torch.Tensor]:
        dataloader = trainer.datamodule.train_eval_dataloader()
        was_training = pl_module.model.training
        pl_module.model.eval()
        embeddings: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        device = pl_module.device
        with torch.no_grad():
            for batch in dataloader:
                image = batch["image"].to(device)
                encoded = pl_module.model.encode_view(image)
                embeddings.append(encoded["embedding"].detach().cpu())
                indices.append(batch["index"].detach().cpu())
        if was_training:
            pl_module.model.train()
        return torch.cat(embeddings, dim=0), torch.cat(indices, dim=0).long()
