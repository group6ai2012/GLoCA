from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DINOCLSAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dims: Sequence[int] = (512, 512, 2048),
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.latent_dim = int(latent_dim)
        self.encoder = _build_mlp(self.input_dim, self.hidden_dims, self.latent_dim)
        self.decoder = _build_mlp(self.latent_dim, tuple(reversed(self.hidden_dims)), self.input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon


class DINOCLSDECModel(DINOCLSAutoencoder):
    def __init__(
        self,
        n_clusters: int,
        input_dim: int = 384,
        hidden_dims: Sequence[int] = (512, 512, 2048),
        latent_dim: int = 64,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
        self.n_clusters = int(n_clusters)
        self.alpha = float(alpha)
        self.cluster_centers = nn.Parameter(torch.empty(self.n_clusters, self.latent_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.cdist(z, self.cluster_centers, p=2).pow(2)
        numerator = (1.0 + dist_sq / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        return numerator / numerator.sum(dim=1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def target_from_q(q: torch.Tensor) -> torch.Tensor:
        weight = q.pow(2) / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(current_dim, int(hidden_dim)), nn.ReLU()])
        current_dim = int(hidden_dim)
    layers.append(nn.Linear(current_dim, int(output_dim)))
    return nn.Sequential(*layers)
