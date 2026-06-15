from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.clustering.kmeans import fit_kmeans
from src.models.baselines.dec_idec import DINOCLSDECModel
from src.training.checkpointing import (
    atomic_torch_save,
    capture_rng_state,
    copy_as_latest,
    empty_resource_totals,
    prune_old_epoch_checkpoints,
    restore_rng_state,
    should_save_epoch_checkpoint,
    timed_section,
    update_resource_totals,
)


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
        checkpoint_dir: Path | None = None,
        resume_from_checkpoint: Path | None = None,
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
        trainer_config = config.get("trainer", {})
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.checkpoint_interval = int(trainer_config.get("checkpoint_interval", 0))
        self.eval_interval = trainer_config.get("eval_interval", "checkpoint")
        self.keep_last_n_checkpoints = int(
            trainer_config.get("keep_last_n_checkpoints", 3)
        )
        self.profile_resources = bool(trainer_config.get("profile_resources", True))
        self.resource_totals = empty_resource_totals()
        self.phase_history: list[dict[str, Any]] = []
        self.pretrain_optimizer = torch.optim.Adam(
            self._autoencoder_parameters(), lr=self.pretrain_lr
        )
        self.refine_optimizer = torch.optim.Adam(
            self._refinement_parameters(), lr=self.refine_lr
        )
        self.resume_phase = "pretrain"
        self.pretrain_start_epoch = 0
        self.refine_start_epoch = 0
        self.cluster_centers_initialized = False
        if resume_from_checkpoint is not None:
            self.load_resumable_checkpoint(Path(resume_from_checkpoint))

    def fit(self) -> None:
        if self.resume_phase == "complete":
            return
        if self.resume_phase == "pretrain":
            self.pretrain(start_epoch=self.pretrain_start_epoch)
        if not self.cluster_centers_initialized:
            self.initialize_cluster_centers()
            self.cluster_centers_initialized = True
            self.save_phase_checkpoint("cluster_init", epoch=0)
        self.refine(start_epoch=self.refine_start_epoch)
        self.save_phase_checkpoint("complete", epoch=0)

    def pretrain(self, start_epoch: int = 0) -> list[float]:
        if int(start_epoch) <= 0:
            self.pretrain_losses = []
        self.model.train()
        for epoch in range(int(start_epoch), self.pretrain_epochs):
            wall_start = time.perf_counter()
            epoch_timing: dict[str, float] = {}
            epoch_loss = 0.0
            n_batches = 0
            with timed_section(epoch_timing, "train_epoch_time_s"):
                for batch, _indices in self._loader(shuffle=True, seed=self.seed + epoch):
                    batch = batch.to(self.device)
                    _z, recon = self.model(batch)
                    loss = F.mse_loss(recon, batch)
                    self.pretrain_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.pretrain_optimizer.step()
                    epoch_loss += float(loss.detach().cpu())
                    n_batches += 1
            mean_loss = epoch_loss / max(1, n_batches)
            self.pretrain_losses.append(mean_loss)
            checkpoint_saved = should_save_epoch_checkpoint(
                epoch, self.checkpoint_interval
            )
            epoch_timing.setdefault("checkpoint_save_time_s", 0.0)
            epoch_timing.setdefault("eval_time_s", 0.0)
            phase_log = {
                "phase": "pretrain",
                "epoch": int(epoch),
                "loss": mean_loss,
                "phase_train_epoch_time_s": epoch_timing["train_epoch_time_s"],
                "checkpoint_save_time_s": epoch_timing["checkpoint_save_time_s"],
                "eval_time_s": epoch_timing["eval_time_s"],
                "phase_epoch_total_wall_time_s": 0.0,
                "checkpoint_saved": bool(checkpoint_saved),
            }
            self.phase_history.append(phase_log)
            if checkpoint_saved:
                with timed_section(epoch_timing, "checkpoint_save_time_s"):
                    self.save_phase_checkpoint("pretrain", epoch)
            epoch_timing["epoch_total_wall_time_s"] = time.perf_counter() - wall_start
            phase_log["checkpoint_save_time_s"] = epoch_timing[
                "checkpoint_save_time_s"
            ]
            phase_log["phase_epoch_total_wall_time_s"] = epoch_timing[
                "epoch_total_wall_time_s"
            ]
            update_resource_totals(self.resource_totals, epoch_timing)
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
        self.cluster_centers_initialized = True

    def refine(self, start_epoch: int = 0) -> list[float]:
        if int(start_epoch) <= 0:
            self.refine_losses = []
            self.kl_losses = []
            self.recon_losses = []
            self.epoch_diagnostics = []
        self.model.train()
        for epoch in range(int(start_epoch), self.refine_epochs):
            wall_start = time.perf_counter()
            epoch_timing: dict[str, float] = {}
            if self._should_refresh_target(epoch):
                self.refresh_target_distribution()
            if self.target_distribution is None:
                raise RuntimeError("Target distribution was not initialized before DEC/IDEC refinement.")

            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_recon = 0.0
            n_batches = 0
            with timed_section(epoch_timing, "train_epoch_time_s"):
                for batch, indices in self._loader(shuffle=True, seed=self.seed + epoch):
                    batch = batch.to(self.device)
                    indices = indices.to(self.device)
                    z, recon = self.model(batch)
                    q = self.model.soft_assign(z)
                    p = self.target_distribution[indices]
                    kl_loss = F.kl_div(q.clamp_min(1e-12).log(), p.detach(), reduction="batchmean")
                    recon_loss = F.mse_loss(recon, batch) if self.mode == "idec" else torch.tensor(0.0, device=self.device)
                    loss = kl_loss + (self.lambda_recon * recon_loss if self.mode == "idec" else 0.0)
                    self.refine_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.refine_optimizer.step()
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
            checkpoint_saved = should_save_epoch_checkpoint(
                epoch, self.checkpoint_interval
            )
            epoch_timing.setdefault("checkpoint_save_time_s", 0.0)
            epoch_timing.setdefault("eval_time_s", 0.0)
            diagnostic = {
                "epoch": epoch + 1,
                "loss": mean_loss,
                "kl_loss": mean_kl,
                "recon_loss": mean_recon,
                "target_update_mode": self.target_update_mode,
                "phase_train_epoch_time_s": epoch_timing["train_epoch_time_s"],
                "checkpoint_save_time_s": epoch_timing["checkpoint_save_time_s"],
                "eval_time_s": epoch_timing["eval_time_s"],
                "phase_epoch_total_wall_time_s": 0.0,
                "checkpoint_saved": bool(checkpoint_saved),
            }
            phase_log = {
                "phase": "refine",
                "epoch": int(epoch),
                "loss": mean_loss,
                "kl_loss": mean_kl,
                "recon_loss": mean_recon,
                "phase_train_epoch_time_s": epoch_timing["train_epoch_time_s"],
                "checkpoint_save_time_s": epoch_timing["checkpoint_save_time_s"],
                "eval_time_s": epoch_timing["eval_time_s"],
                "phase_epoch_total_wall_time_s": 0.0,
                "checkpoint_saved": bool(checkpoint_saved),
            }
            self.epoch_diagnostics.append(diagnostic)
            self.phase_history.append(phase_log)
            if checkpoint_saved:
                with timed_section(epoch_timing, "checkpoint_save_time_s"):
                    self.save_phase_checkpoint("refine", epoch)
            epoch_timing["epoch_total_wall_time_s"] = time.perf_counter() - wall_start
            diagnostic["checkpoint_save_time_s"] = epoch_timing[
                "checkpoint_save_time_s"
            ]
            phase_log["checkpoint_save_time_s"] = epoch_timing[
                "checkpoint_save_time_s"
            ]
            diagnostic["phase_epoch_total_wall_time_s"] = epoch_timing[
                "epoch_total_wall_time_s"
            ]
            phase_log["phase_epoch_total_wall_time_s"] = epoch_timing[
                "epoch_total_wall_time_s"
            ]
            update_resource_totals(self.resource_totals, epoch_timing)
            print(f"refine_epoch={epoch + 1} loss={mean_loss:.6f}", flush=True)
        return self.refine_losses

    def resumable_checkpoint_payload(self, phase: str, epoch: int) -> dict[str, Any]:
        return {
            "checkpoint_version": 1,
            "method": self.mode,
            "phase": str(phase),
            "phase_epoch": int(epoch),
            "next_phase_epoch": int(epoch) + 1,
            "model_state_dict": self.model.state_dict(),
            "pretrain_optimizer_state_dict": self.pretrain_optimizer.state_dict()
            if self.pretrain_optimizer is not None
            else None,
            "refine_optimizer_state_dict": self.refine_optimizer.state_dict()
            if self.refine_optimizer is not None
            else None,
            "target_distribution": None
            if self.target_distribution is None
            else self.target_distribution.detach().cpu(),
            "cluster_centers_initialized": bool(self.cluster_centers_initialized),
            "config": self.config,
            "pretrain_losses": self.pretrain_losses,
            "refine_losses": self.refine_losses,
            "kl_losses": self.kl_losses,
            "recon_losses": self.recon_losses,
            "epoch_diagnostics": self.epoch_diagnostics,
            "phase_history": self.phase_history,
            "resource_totals": self.resource_totals,
            "checkpoint_interval": self.checkpoint_interval,
            "eval_interval": self.eval_interval,
            "rng_state": capture_rng_state(),
        }

    def load_resumable_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("method") != self.mode:
            raise ValueError(
                f"Expected {self.mode.upper()} checkpoint, got {ckpt.get('method')!r}"
            )
        self.model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("pretrain_optimizer_state_dict") is not None:
            self.pretrain_optimizer.load_state_dict(
                ckpt["pretrain_optimizer_state_dict"]
            )
        if ckpt.get("refine_optimizer_state_dict") is not None:
            self.refine_optimizer.load_state_dict(ckpt["refine_optimizer_state_dict"])
        target_distribution = ckpt.get("target_distribution")
        self.target_distribution = (
            None if target_distribution is None else target_distribution.to(self.device)
        )
        self.cluster_centers_initialized = bool(
            ckpt.get("cluster_centers_initialized", False)
        )
        self.pretrain_losses = list(ckpt.get("pretrain_losses", []))
        self.refine_losses = list(ckpt.get("refine_losses", []))
        self.kl_losses = list(ckpt.get("kl_losses", []))
        self.recon_losses = list(ckpt.get("recon_losses", []))
        self.epoch_diagnostics = list(ckpt.get("epoch_diagnostics", []))
        self.phase_history = list(ckpt.get("phase_history", []))
        self.resource_totals = {
            **empty_resource_totals(),
            **dict(ckpt.get("resource_totals", {})),
        }
        restore_rng_state(ckpt.get("rng_state"))
        phase = str(ckpt.get("phase", "pretrain"))
        next_epoch = int(ckpt.get("next_phase_epoch", int(ckpt.get("phase_epoch", 0)) + 1))
        self.resume_phase = phase
        if phase == "pretrain":
            self.pretrain_start_epoch = next_epoch
            self.refine_start_epoch = 0
        elif phase == "cluster_init":
            self.pretrain_start_epoch = self.pretrain_epochs
            self.refine_start_epoch = 0
        elif phase == "refine":
            self.pretrain_start_epoch = self.pretrain_epochs
            self.refine_start_epoch = next_epoch
        elif phase == "complete":
            self.pretrain_start_epoch = self.pretrain_epochs
            self.refine_start_epoch = self.refine_epochs
            self.resume_phase = "complete"
        else:
            raise ValueError(f"Unknown DEC/IDEC checkpoint phase {phase!r}")

    def save_phase_checkpoint(self, phase: str, epoch: int) -> None:
        if self.checkpoint_dir is None:
            return
        payload = self.resumable_checkpoint_payload(phase, epoch)
        epoch_path = self.checkpoint_dir / f"epoch_{phase}_{epoch + 1:04d}.ckpt"
        latest_path = self.checkpoint_dir / "latest.ckpt"
        atomic_torch_save(payload, epoch_path)
        copy_as_latest(epoch_path, latest_path)
        prune_old_epoch_checkpoints(
            self.checkpoint_dir, keep_last_n=self.keep_last_n_checkpoints
        )

    def training_logs(self) -> dict[str, Any]:
        return {
            "pretrain_losses": self.pretrain_losses,
            "refine_losses": self.refine_losses,
            "kl_losses": self.kl_losses,
            "recon_losses": self.recon_losses,
            "epoch_diagnostics": self.epoch_diagnostics,
            "phase_history": self.phase_history,
            "resource_totals": self.resource_totals,
            "eval_interval": self.eval_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "keep_last_n_checkpoints": self.keep_last_n_checkpoints,
            "profile_resources": self.profile_resources,
            "cluster_centers_initialized": self.cluster_centers_initialized,
            "resume_phase": self.resume_phase,
        }

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
