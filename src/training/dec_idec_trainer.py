from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.clustering.kmeans import fit_kmeans
from src.models.baselines.dec_idec import DINOCLSDECModel


class DECIDECTrainer:
    def __init__(
        self,
        model: DINOCLSDECModel,
        x: torch.Tensor,
        config: dict[str, Any],
        mode: str,
        n_clusters: int,
        seed: int,
        device: torch.device,
        labels: torch.Tensor | None = None,
    ) -> None:
        mode = str(mode).lower()
        if mode not in {"dec", "idec"}:
            raise ValueError(f"DECIDECTrainer mode must be 'dec' or 'idec', got {mode!r}")
        if bool(config.get("gloca", {}).get("enabled", False)):
            raise ValueError("Standalone DEC/IDEC baselines must not enable GLoCA.")
        if int(n_clusters) != int(model.n_clusters):
            raise ValueError(f"n_clusters={n_clusters} does not match model.n_clusters={model.n_clusters}")

        self.model = model.to(device)
        self.x = x.detach().cpu().float().contiguous()
        self.config = config
        self.mode = mode
        self.n_clusters = int(n_clusters)
        self.seed = int(seed)
        self.device = device
        self.labels = None if labels is None else labels.detach().cpu().long().contiguous()

        baseline = config.get("baseline", {})
        self.batch_size = int(config.get("trainer", {}).get("batch_size", 128))
        self.pretrain_epochs = int(baseline.get("pretrain_epochs", 20))
        self.refine_epochs = int(baseline.get("refine_epochs", 10))
        self.pretrain_lr = float(baseline.get("pretrain_lr", config.get("trainer", {}).get("lr", 1.0e-3)))
        self.refine_lr = float(baseline.get("refine_lr", config.get("trainer", {}).get("lr", 1.0e-4)))
        self.lambda_recon = float(baseline.get("lambda_recon", 0.1))
        self.target_update_mode, self.target_update_interval = parse_target_update_interval(
            baseline.get("target_update_interval")
        )

        self.pretrain_losses: list[float] = []
        self.refine_losses: list[float] = []
        self.kl_losses: list[float] = []
        self.recon_losses: list[float] = []
        self.epoch_diagnostics: list[dict[str, Any]] = []
        self.target_distribution: torch.Tensor | None = None

    def pretrain(self) -> list[float]:
        optimizer = torch.optim.Adam(self._autoencoder_parameters(), lr=self.pretrain_lr)
        self.pretrain_losses = []
        self.model.train()
        for epoch in range(self.pretrain_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch, _indices in self._loader(shuffle=True, seed=self.seed + epoch):
                batch = batch.to(self.device)
                _z, recon = self.model(batch)
                loss = F.mse_loss(recon, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1
            mean_loss = epoch_loss / max(1, n_batches)
            self.pretrain_losses.append(mean_loss)
            print(f"pretrain_epoch={epoch + 1} loss={mean_loss:.6f}", flush=True)
        return self.pretrain_losses

    def initialize_cluster_centers(self) -> None:
        z = self.encode_all()
        result = fit_kmeans(
            z,
            self.n_clusters,
            spherical=False,
            init=str(self.config["baseline"]["kmeans_init"]),
            n_init=int(self.config["baseline"]["kmeans_n_init"]),
            max_iter=int(self.config["baseline"]["kmeans_max_iter"]),
            tol=float(self.config["baseline"]["kmeans_tol"]),
            seed=self.seed,
            device=self.device,
        )
        self.model.cluster_centers.data.copy_(result["centers"].to(self.device))
        self.target_distribution = None

    def refine(self) -> list[float]:
        optimizer = torch.optim.Adam(self._refinement_parameters(), lr=self.refine_lr)
        self.refine_losses = []
        self.kl_losses = []
        self.recon_losses = []
        self.epoch_diagnostics = []
        self.model.train()
        for epoch in range(self.refine_epochs):
            if self._should_refresh_target(epoch):
                self.refresh_target_distribution()
            if self.target_distribution is None:
                raise RuntimeError("Target distribution was not initialized before DEC/IDEC refinement.")

            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_recon = 0.0
            n_batches = 0
            for batch, indices in self._loader(shuffle=True, seed=self.seed + epoch):
                batch = batch.to(self.device)
                indices = indices.to(self.device)
                z, recon = self.model(batch)
                q = self.model.soft_assign(z)
                p = self.target_distribution[indices]
                kl_loss = F.kl_div(q.clamp_min(1e-12).log(), p.detach(), reduction="batchmean")
                recon_loss = F.mse_loss(recon, batch) if self.mode == "idec" else torch.tensor(0.0, device=self.device)
                loss = kl_loss + (self.lambda_recon * recon_loss if self.mode == "idec" else 0.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                epoch_kl += float(kl_loss.detach().cpu())
                epoch_recon += float(recon_loss.detach().cpu())
                n_batches += 1

            mean_loss = epoch_loss / max(1, n_batches)
            mean_kl = epoch_kl / max(1, n_batches)
            mean_recon = epoch_recon / max(1, n_batches)
            self.refine_losses.append(mean_loss)
            self.kl_losses.append(mean_kl)
            self.recon_losses.append(mean_recon)
            self.epoch_diagnostics.append(
                {
                    "epoch": epoch + 1,
                    "loss": mean_loss,
                    "kl_loss": mean_kl,
                    "recon_loss": mean_recon,
                    "target_update_mode": self.target_update_mode,
                }
            )
            print(f"refine_epoch={epoch + 1} loss={mean_loss:.6f}", flush=True)
        return self.refine_losses

    def predict_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        z_parts: list[torch.Tensor] = []
        assignment_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for batch, _indices in self._loader(shuffle=False, seed=0):
                z = self.model.encode(batch.to(self.device))
                q = self.model.soft_assign(z)
                z_parts.append(z.detach().cpu())
                assignment_parts.append(q.argmax(dim=1).detach().cpu())
        return torch.cat(z_parts, dim=0), torch.cat(assignment_parts, dim=0)

    def encode_all(self) -> torch.Tensor:
        self.model.eval()
        z_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for batch, _indices in self._loader(shuffle=False, seed=0):
                z_parts.append(self.model.encode(batch.to(self.device)).detach().cpu())
        self.model.train()
        return torch.cat(z_parts, dim=0)

    def refresh_target_distribution(self) -> torch.Tensor:
        self.model.eval()
        q_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for batch, _indices in self._loader(shuffle=False, seed=0):
                z = self.model.encode(batch.to(self.device))
                q_parts.append(self.model.soft_assign(z).detach().cpu())
        self.model.train()
        q = torch.cat(q_parts, dim=0)
        self.target_distribution = self.model.target_from_q(q).to(self.device)
        return self.target_distribution

    def _should_refresh_target(self, epoch: int) -> bool:
        if self.target_distribution is None:
            return True
        if self.target_update_mode == "fixed":
            return False
        if self.target_update_interval is None:
            return False
        return epoch > 0 and epoch % self.target_update_interval == 0

    def _autoencoder_parameters(self):
        return list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())

    def _refinement_parameters(self):
        if self.mode == "idec":
            return (
                list(self.model.encoder.parameters())
                + list(self.model.decoder.parameters())
                + [self.model.cluster_centers]
            )
        return list(self.model.encoder.parameters()) + [self.model.cluster_centers]

    def _loader(self, shuffle: bool, seed: int) -> DataLoader:
        dataset = TensorDataset(self.x, torch.arange(self.x.shape[0], dtype=torch.long))
        generator = torch.Generator().manual_seed(int(seed))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, generator=generator)


def parse_target_update_interval(value: Any) -> tuple[str, int | None]:
    if value is None:
        return "fixed", None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null", "fixed"}:
            return "fixed", None
        value = int(normalized)
    interval = int(value)
    if interval == 0:
        return "fixed", None
    if interval < 0:
        raise ValueError(f"target_update_interval must be non-negative or null, got {value!r}")
    return "interval", interval
