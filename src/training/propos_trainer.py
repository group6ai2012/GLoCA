from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch import nn
from tqdm.auto import tqdm

from src.data import ClusteringDataModule
from src.evaluation.clustering_metrics import clustering_accuracy
from src.models.clustering.kmeans import fit_kmeans
from src.training.checkpointing import (
    atomic_torch_save,
    capture_rng_state,
    copy_as_latest,
    empty_resource_totals,
    restore_rng_state,
    should_run_eval,
    should_save_epoch_checkpoint,
    timed_section,
    update_resource_totals,
)
from src.training.gloca_utils import (
    assert_backbone_frozen,
    compute_gloca_diagnostics,
    current_gloca_alpha,
    desired_gloca_trainable,
    gloca_alpha_parameters,
    gloca_optimizer_groups,
    set_gloca_trainable,
    split_gloca_parameters,
)


class ProPosTargetEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, adapter: nn.Module | None, normalize_cls: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = clone_target_adapter(adapter)
        self.normalize_cls = bool(normalize_cls)

    def encode_view(self, image: torch.Tensor) -> dict[str, Any]:
        backbone_out = self.backbone(image)
        if self.adapter is None:
            embedding = backbone_out["cls"]
            if self.normalize_cls:
                embedding = F.normalize(embedding, dim=-1)
            return {"embedding": embedding, "attention": None, "patch_grid": backbone_out["patch_grid"]}
        return self.adapter(
            cls=backbone_out["cls"],
            patch_tokens=backbone_out["patch_tokens"],
            patch_grid=backbone_out["patch_grid"],
        )


class ProPosTrainer:
    def __init__(
        self,
        model: nn.Module,
        datamodule: ClusteringDataModule,
        config: dict[str, Any],
        device: torch.device,
        checkpoint_dir: Path | None = None,
        resume_from_checkpoint: Path | None = None,
        checkpoint_metric_logger: Any | None = None,
    ) -> None:
        self.model = model.to(device)
        self.datamodule = datamodule
        self.config = config
        self.device = device
        self.propos_config = config.get("propos", {})
        self.target_encoder = ProPosTargetEncoder(
            backbone=model.backbone,
            adapter=model.adapter,
            normalize_cls=model.normalize_cls,
        ).to(device)

        self.loss_psa_final = float("nan")
        self.loss_psl_final = float("nan")
        self.loss_total_final = float("nan")
        self.ema_momentum_final = float(self.propos_config["ema_momentum"])
        self.n_empty_cluster_batches = 0
        self.n_invalid_psl_batches = 0
        self.estep_history: list[dict[str, Any]] = []
        self.step_history: list[dict[str, Any]] = []
        self.epoch_history: list[dict[str, Any]] = []
        self.profile_batches = bool(self.propos_config["profile_batches"])
        self.progress_enabled = bool(self.propos_config.get("progress", True))
        self.gloca_lr_multiplier = float(self.propos_config["gloca_lr_multiplier"])
        self.gloca_alpha_lr_multiplier = float(self.propos_config["gloca_alpha_lr_multiplier"])
        self.freeze_gloca = bool(self.propos_config["freeze_gloca"])
        self.freeze_gloca_epochs = int(self.propos_config["freeze_gloca_epochs"])
        self.log_gloca_diagnostics = bool(self.propos_config["log_gloca_diagnostics"])
        self.gloca_alpha_initial = self._current_gloca_alpha()
        self.gloca_diagnostics_history: list[dict[str, Any]] = []
        self.gloca_alpha_found = bool(self._gloca_alpha_parameters())
        self.gloca_trainable = False
        self.profile_totals = {
            "profiled_batches": 0,
            "data_time_s": 0.0,
            "forward_time_s": 0.0,
            "loss_time_s": 0.0,
            "backward_time_s": 0.0,
        }
        trainer_config = config.get("trainer", {})
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.checkpoint_interval = int(trainer_config.get("checkpoint_interval", 0))
        self.eval_interval = trainer_config.get("eval_interval", "checkpoint")
        self.profile_resources = bool(trainer_config.get("profile_resources", True))
        self.checkpoint_metric_logger = checkpoint_metric_logger
        self.start_epoch = 0
        self.resource_totals = empty_resource_totals()
        self.global_step = 0
        self.total_steps = 1
        self.set_gloca_trainable(self._desired_gloca_trainable(epoch=0))
        self.optimizer = self._build_optimizer()
        if resume_from_checkpoint is not None:
            self.load_resumable_checkpoint(Path(resume_from_checkpoint))

    def fit(self) -> None:
        self.assert_backbone_frozen()
        if self.start_epoch == 0:
            self.model.head.update_target_projector(momentum=0.0)
            self.update_target_adapter(momentum=0.0)
        else:
            tqdm.write(f"Resuming ProPos from epoch {self.start_epoch}")

        train_loader = self.datamodule.train_dataloader()
        max_epochs = int(self.config["trainer"]["max_epochs"])
        self.total_steps = max(1, max_epochs * len(train_loader))
        kmeans_interval = int(self.propos_config["kmeans_interval"])
        if kmeans_interval <= 0:
            raise ValueError(f"propos.kmeans_interval must be positive, got {kmeans_interval}")

        if self.start_epoch >= max_epochs:
            tqdm.write("ProPos checkpoint already reached max_epochs; skipping training.")
            return

        for epoch in range(self.start_epoch, max_epochs):
            wall_start = time.perf_counter()
            epoch_timing: dict[str, float] = {}
            self.maybe_update_gloca_freeze_state(epoch)
            if epoch % kmeans_interval == 0:
                self.run_estep(epoch)
            with timed_section(epoch_timing, "train_epoch_time_s"):
                epoch_logs = self.train_epoch(
                    epoch, train_loader, max_epochs=max_epochs
                )
            should_checkpoint = should_save_epoch_checkpoint(
                epoch, self.checkpoint_interval
            )
            run_eval = should_run_eval(
                epoch=epoch,
                max_epochs=max_epochs,
                eval_interval=self.eval_interval,
                checkpoint_interval=self.checkpoint_interval,
            )
            epoch_timing.setdefault("eval_time_s", 0.0)
            epoch_timing.setdefault("checkpoint_save_time_s", 0.0)
            gloca_diagnostics = self.compute_gloca_diagnostics(epoch)
            if gloca_diagnostics is not None:
                self.gloca_diagnostics_history.append(gloca_diagnostics)
                epoch_logs["gloca_diagnostics"] = gloca_diagnostics
            epoch_logs.update(epoch_timing)
            epoch_logs["evaluated"] = bool(run_eval)
            epoch_logs["checkpoint_saved"] = bool(should_checkpoint)
            self.epoch_history.append(epoch_logs)
            if should_checkpoint:
                with timed_section(epoch_timing, "checkpoint_save_time_s"):
                    self.save_epoch_checkpoint(epoch)
            epoch_timing["epoch_total_wall_time_s"] = time.perf_counter() - wall_start
            epoch_logs.update(epoch_timing)
            update_resource_totals(self.resource_totals, epoch_timing)
            if should_checkpoint and self.checkpoint_metric_logger is not None:
                self.checkpoint_metric_logger(epoch_logs)
            tqdm.write(
                "Epoch "
                f"{epoch + 1}/{max_epochs} "
                f"loss={epoch_logs['loss_total_mean']:.4f} "
                f"nmi={epoch_logs['nmi']:.4f} "
                f"ari={epoch_logs['ari']:.4f} "
                f"acc={epoch_logs['acc']:.4f}"
            )

    def resumable_checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return {
            "checkpoint_version": 1,
            "method": "propos",
            "epoch": int(epoch),
            "next_epoch": int(epoch) + 1,
            "global_step": int(self.global_step),
            "total_steps": int(self.total_steps),
            "model_state_dict": self.model.state_dict(),
            "target_encoder_state_dict": self.target_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "estep_history": self.estep_history,
            "step_history_tail": self.step_history[-20:],
            "epoch_history": self.epoch_history,
            "profile_totals": self.profile_totals,
            "resource_totals": self.resource_totals,
            "gloca_diagnostics_history": self.gloca_diagnostics_history,
            "loss_psa_final": self.loss_psa_final,
            "loss_psl_final": self.loss_psl_final,
            "loss_total_final": self.loss_total_final,
            "ema_momentum_final": self.ema_momentum_final,
            "n_empty_cluster_batches": self.n_empty_cluster_batches,
            "n_invalid_psl_batches": self.n_invalid_psl_batches,
            "gloca_trainable": self.gloca_trainable,
            "checkpoint_interval": self.checkpoint_interval,
            "eval_interval": self.eval_interval,
            "rng_state": capture_rng_state(),
        }

    def load_resumable_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("method") != "propos":
            raise ValueError(f"Expected ProPos checkpoint, got {ckpt.get('method')!r}")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.target_encoder.load_state_dict(ckpt["target_encoder_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.total_steps = int(ckpt.get("total_steps", 1))
        self.estep_history = list(ckpt.get("estep_history", []))
        self.step_history = list(ckpt.get("step_history_tail", []))
        self.epoch_history = list(ckpt.get("epoch_history", []))
        self.profile_totals = {**self.profile_totals, **dict(ckpt.get("profile_totals", {}))}
        self.resource_totals = {
            **empty_resource_totals(),
            **dict(ckpt.get("resource_totals", {})),
        }
        self.gloca_diagnostics_history = list(
            ckpt.get("gloca_diagnostics_history", [])
        )
        self.loss_psa_final = float(ckpt.get("loss_psa_final", float("nan")))
        self.loss_psl_final = float(ckpt.get("loss_psl_final", float("nan")))
        self.loss_total_final = float(ckpt.get("loss_total_final", float("nan")))
        self.ema_momentum_final = float(
            ckpt.get("ema_momentum_final", self.ema_momentum_final)
        )
        self.n_empty_cluster_batches = int(ckpt.get("n_empty_cluster_batches", 0))
        self.n_invalid_psl_batches = int(ckpt.get("n_invalid_psl_batches", 0))
        self.set_gloca_trainable(bool(ckpt.get("gloca_trainable", self.gloca_trainable)))
        restore_rng_state(ckpt.get("rng_state"))
        self.start_epoch = int(ckpt.get("next_epoch", int(ckpt["epoch"]) + 1))

    def save_epoch_checkpoint(self, epoch: int) -> None:
        if self.checkpoint_dir is None:
            return
        payload = self.resumable_checkpoint_payload(epoch)
        epoch_path = self.checkpoint_dir / f"epoch_{epoch + 1:04d}.ckpt"
        latest_path = self.checkpoint_dir / "latest.ckpt"
        atomic_torch_save(payload, epoch_path)
        copy_as_latest(epoch_path, latest_path)

    def run_estep(self, epoch: int) -> None:
        self.model.eval()
        self.target_encoder.eval()
        self.model.head.target_projector.eval()

        feature_parts: list[torch.Tensor] = []
        index_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in self.datamodule.train_eval_dataloader():
                image = batch["image"].to(self.device)
                encoded = self.target_encoder.encode_view(image)
                projected = self.model.head.forward_target(encoded["embedding"])
                feature_parts.append(projected.detach().cpu())
                index_parts.append(batch["index"].detach().cpu())
                label_parts.append(batch["label"].detach().cpu())

        features = F.normalize(torch.cat(feature_parts, dim=0).float(), dim=-1)
        indices = torch.cat(index_parts, dim=0).long()
        labels = torch.cat(label_parts, dim=0).long()
        order = torch.argsort(indices)
        ordered_features = features[order].contiguous()
        ordered_indices = indices[order].contiguous()
        ordered_labels = labels[order].contiguous()

        kmeans_init = str(self.propos_config["kmeans_init"]).lower()
        kmeans_seed = int(self.config["experiment"]["seed"]) + int(epoch)
        if kmeans_init not in {"random", "kmeans++"}:
            raise ValueError("propos.kmeans_init must be either 'random' or 'kmeans++'")
        kmeans_result = fit_kmeans(
            ordered_features,
            int(self.model.head.n_clusters),
            spherical=True,
            init=kmeans_init,
            n_init=int(self.propos_config["kmeans_n_init"]),
            max_iter=int(self.propos_config["kmeans_max_iter"]),
            tol=float(self.propos_config["kmeans_tol"]),
            seed=kmeans_seed,
            device=self.device,
        )
        assignments = kmeans_result["assignments"]
        kmeans_logs = kmeans_result["logs"]
        table_size = int(ordered_indices.max().item()) + 1
        pseudo_labels = torch.full((table_size,), -1, dtype=torch.long)
        pseudo_labels[ordered_indices] = assignments
        if bool((pseudo_labels[ordered_indices] < 0).any()):
            raise RuntimeError("Failed to populate ProPos pseudo-labels by dataset index.")
        self.model.head.set_pseudo_labels(pseudo_labels.to(self.device))

        counts = torch.bincount(assignments, minlength=int(self.model.head.n_clusters))
        eval_metrics = self._assignment_metrics(ordered_labels, assignments)
        history = {
            "epoch": int(epoch),
            "n_samples": int(assignments.numel()),
            "n_nonempty_clusters": int((counts > 0).sum().item()),
            "cluster_size_min": int(counts[counts > 0].min().item()) if bool((counts > 0).any()) else 0,
            "cluster_size_max": int(counts.max().item()) if counts.numel() else 0,
            **eval_metrics,
            "kmeans_backend": "torch",
            "kmeans_init": kmeans_init,
            "kmeans_logs": kmeans_logs,
        }
        self.estep_history.append(history)

    def train_epoch(self, epoch: int, train_loader=None, max_epochs: int | None = None) -> dict[str, Any]:
        train_loader = train_loader or self.datamodule.train_dataloader()
        self.model.train()
        self.model.backbone.eval()
        self.target_encoder.eval()
        self.model.head.target_projector.eval()
        train_iter = iter(train_loader)
        progress = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}" if max_epochs is None else f"Epoch {epoch + 1}/{max_epochs}",
            unit="batch",
            leave=False,
            disable=not self.progress_enabled,
        )
        epoch_loss_total = 0.0
        epoch_loss_psa = 0.0
        epoch_loss_psl = 0.0
        n_batches = 0
        while True:
            data_start = time.perf_counter()
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            data_time_s = time.perf_counter() - data_start
            self.optimizer.zero_grad(set_to_none=True)
            loss, step_logs = self.train_step(batch, epoch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite ProPos loss at epoch={epoch} step={self.global_step}: {loss}")
            backward_start = time.perf_counter()
            loss.backward()
            self.optimizer.step()
            momentum = self.ema_momentum_for_step(self.global_step)
            self.update_target(momentum=momentum)
            if self.profile_batches:
                self._sync_device()
            backward_time_s = time.perf_counter() - backward_start
            self.ema_momentum_final = float(momentum)
            if self.profile_batches:
                step_logs["data_time_s"] = float(data_time_s)
                step_logs["backward_time_s"] = float(backward_time_s)
                self.profile_totals["profiled_batches"] += 1
                self.profile_totals["data_time_s"] += float(data_time_s)
                self.profile_totals["forward_time_s"] += float(step_logs.get("forward_time_s", 0.0))
                self.profile_totals["loss_time_s"] += float(step_logs.get("loss_time_s", 0.0))
                self.profile_totals["backward_time_s"] += float(backward_time_s)
            self.step_history.append(step_logs)
            epoch_loss_total += float(step_logs["loss_total"])
            epoch_loss_psa += float(step_logs["loss_psa"])
            epoch_loss_psl += float(step_logs["loss_psl"])
            n_batches += 1
            progress.set_postfix(
                loss=f"{step_logs['loss_total']:.4f}",
                nmi=f"{self._latest_estep_metric('nmi'):.4f}",
                ari=f"{self._latest_estep_metric('ari'):.4f}",
                acc=f"{self._latest_estep_metric('acc'):.4f}",
            )
            progress.update(1)
            self.global_step += 1
        progress.close()
        divisor = max(1, n_batches)
        latest_metrics = self._latest_estep_metrics()
        return {
            "epoch": int(epoch),
            "n_batches": int(n_batches),
            "loss_total_mean": float(epoch_loss_total / divisor),
            "loss_psa_mean": float(epoch_loss_psa / divisor),
            "loss_psl_mean": float(epoch_loss_psl / divisor),
            **latest_metrics,
        }

    def train_step(self, batch: dict[str, Any], epoch: int) -> tuple[torch.Tensor, dict[str, Any]]:
        view1, view2 = batch["views"]
        view1 = view1.to(self.device)
        view2 = view2.to(self.device)
        indices = batch["index"].to(self.device)
        pseudo_labels = self.model.head.batch_pseudo_labels(indices)
        if bool((pseudo_labels < 0).any()):
            raise RuntimeError("ProPos pseudo-label table contains missing entries for this batch.")

        forward_start = time.perf_counter()
        encoded1 = self.model.encode_view(view1)
        encoded2 = self.model.encode_view(view2)
        online1 = self.model.head.forward_online(
            encoded1["embedding"],
            attention=encoded1["attention"],
            patch_grid=encoded1["patch_grid"],
        )
        online2 = self.model.head.forward_online(
            encoded2["embedding"],
            attention=encoded2["attention"],
            patch_grid=encoded2["patch_grid"],
        )
        with torch.no_grad():
            target1_encoded = self.target_encoder.encode_view(view1)
            target2_encoded = self.target_encoder.encode_view(view2)
            target1 = self.model.head.forward_target(target1_encoded["embedding"])
            target2 = self.model.head.forward_target(target2_encoded["embedding"])
        if self.profile_batches:
            self._sync_device()
        forward_time_s = time.perf_counter() - forward_start

        loss_start = time.perf_counter()
        warmup_active = self.is_warmup(epoch)
        sigma_active = 0.0 if warmup_active else float(self.propos_config["sigma"])
        loss_psa_12 = self.model.head.positive_sampling_alignment(online1["projected"], target2, sigma=sigma_active)
        loss_psl_12, psl_valid_12 = self.model.head.prototype_scattering_loss(
            online1["projected"],
            target2,
            pseudo_labels,
        )
        loss_psa = loss_psa_12
        loss_psl = loss_psl_12
        psl_valid = psl_valid_12

        if bool(self.propos_config["symmetric_loss"]):
            loss_psa_21 = self.model.head.positive_sampling_alignment(
                online2["projected"],
                target1,
                sigma=sigma_active,
            )
            loss_psl_21, psl_valid_21 = self.model.head.prototype_scattering_loss(
                online2["projected"],
                target1,
                pseudo_labels,
            )
            loss_psa = 0.5 * (loss_psa_12 + loss_psa_21)
            loss_psl = 0.5 * (loss_psl_12 + loss_psl_21)
            psl_valid = psl_valid_12 and psl_valid_21

        if not psl_valid:
            self.n_invalid_psl_batches += 1
            self.n_empty_cluster_batches += 1
        elif float(loss_psl.detach().cpu()) == 0.0:
            self.n_empty_cluster_batches += 1

        lambda_psl = float(self.propos_config["lambda_psl"])
        loss = loss_psa if warmup_active else loss_psa + lambda_psl * loss_psl
        if self.profile_batches:
            self._sync_device()
        loss_time_s = time.perf_counter() - loss_start
        self.loss_psa_final = float(loss_psa.detach().cpu())
        self.loss_psl_final = float(loss_psl.detach().cpu())
        self.loss_total_final = float(loss.detach().cpu())
        logs = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "loss_psa": self.loss_psa_final,
            "loss_psl": self.loss_psl_final,
            "loss_total": self.loss_total_final,
            "sigma_active": float(sigma_active),
            "ema_momentum": float(self.ema_momentum_for_step(self.global_step)),
            "psl_valid": bool(psl_valid),
        }
        if self.profile_batches:
            logs["forward_time_s"] = float(forward_time_s)
            logs["loss_time_s"] = float(loss_time_s)
        return loss, logs

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        self.model.head.update_target_projector(momentum=momentum)
        self.update_target_adapter(momentum=momentum)

    @torch.no_grad()
    def update_target_adapter(self, momentum: float) -> None:
        if self.model.adapter is None or self.target_encoder.adapter is None:
            return
        for online, target in zip(self.model.adapter.parameters(), self.target_encoder.adapter.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
        for online_buffer, target_buffer in zip(self.model.adapter.buffers(), self.target_encoder.adapter.buffers()):
            target_buffer.copy_(online_buffer)

    @torch.no_grad()
    def extract_deterministic_features(self) -> dict[str, Any]:
        self.model.eval()
        self.target_encoder.eval()
        self.model.head.target_projector.eval()

        embedding_parts: list[torch.Tensor] = []
        attention_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        index_parts: list[torch.Tensor] = []
        image_ids: list[str] = []
        patch_grid = None
        for batch in self.datamodule.predict_dataloader():
            image = batch["image"].to(self.device)
            encoded = self.target_encoder.encode_view(image)
            projected = self.model.head.forward_target(encoded["embedding"])
            embedding_parts.append(projected.detach().cpu())
            if encoded["attention"] is not None:
                attention_parts.append(encoded["attention"].detach().cpu())
            label_parts.append(batch["label"].detach().cpu())
            index_parts.append(batch["index"].detach().cpu())
            image_ids.extend(batch["image_id"])
            patch_grid = encoded["patch_grid"]

        indices = torch.cat(index_parts, dim=0).long()
        order = torch.argsort(indices)
        attention = torch.cat(attention_parts, dim=0)[order].contiguous() if attention_parts else None
        ordered_ids = [image_ids[i] for i in order.tolist()]
        embeddings = F.normalize(torch.cat(embedding_parts, dim=0)[order].float(), dim=-1).contiguous()
        return {
            "embeddings": embeddings,
            "attention": attention,
            "labels": torch.cat(label_parts, dim=0)[order].long().contiguous(),
            "indices": indices[order].contiguous(),
            "image_ids": ordered_ids,
            "patch_grid": patch_grid,
        }

    def ema_momentum_for_step(self, step: int) -> float:
        base = float(self.propos_config["ema_momentum"])
        if not bool(self.propos_config["ema_momentum_increase"]):
            return base
        max_momentum = float(self.propos_config["ema_momentum_max"])
        progress_step = min(max(0, int(step)), self.total_steps)
        return max_momentum - (max_momentum - base) * (
            math.cos(math.pi * progress_step / max(1, self.total_steps)) + 1.0
        ) / 2.0

    def is_warmup(self, epoch: int) -> bool:
        # Matches the official ProPos condition: warmup is active while not current_epoch > warmup_epochs.
        return int(epoch) <= int(self.propos_config["warmup_epochs"])

    def assert_backbone_frozen(self) -> None:
        assert_backbone_frozen(self.model.backbone)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "target_adapter_state_dict": None
            if self.target_encoder.adapter is None
            else self.target_encoder.adapter.state_dict(),
            "config": self.config,
            "propos_logs": self.training_logs(),
        }

    def training_logs(self) -> dict[str, Any]:
        return {
            "loss_psa_final": self.loss_psa_final,
            "loss_psl_final": self.loss_psl_final,
            "loss_total_final": self.loss_total_final,
            "ema_momentum_final": self.ema_momentum_final,
            "n_empty_cluster_batches": self.n_empty_cluster_batches,
            "n_invalid_psl_batches": self.n_invalid_psl_batches,
            "kmeans_backend": "torch",
            "kmeans_init": str(self.propos_config["kmeans_init"]).lower(),
            "estep_kmeans_backend": "torch",
            "estep_kmeans_logs": [history.get("kmeans_logs", {}) for history in self.estep_history],
            "estep_history": self.estep_history,
            "epoch_history": self.epoch_history,
            "last_steps": self.step_history[-20:],
            "global_step": self.global_step,
            "total_steps": self.total_steps,
            "profile_batches": self.profile_batches,
            "resource_totals": self.resource_totals,
            "eval_interval": self.eval_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "profile_resources": self.profile_resources,
            "gloca_diagnostics_history": self.gloca_diagnostics_history,
            "gloca_alpha_initial": self.gloca_alpha_initial,
            "gloca_alpha_final": self._current_gloca_alpha(),
            "gloca_alpha_found": self.gloca_alpha_found,
            "gloca_lr_multiplier": self.gloca_lr_multiplier,
            "gloca_alpha_lr_multiplier": self.gloca_alpha_lr_multiplier,
            "freeze_gloca": self.freeze_gloca,
            "freeze_gloca_epochs": self.freeze_gloca_epochs,
            "gloca_trainable": self.gloca_trainable,
            **self.profile_totals,
        }

    def set_gloca_trainable(self, trainable: bool) -> None:
        self.gloca_trainable = set_gloca_trainable(self.model.adapter, trainable)

    def maybe_update_gloca_freeze_state(self, epoch: int) -> None:
        self.set_gloca_trainable(self._desired_gloca_trainable(epoch))

    def _desired_gloca_trainable(self, epoch: int) -> bool:
        return desired_gloca_trainable(
            self.model.adapter,
            freeze_gloca=self.freeze_gloca,
            freeze_gloca_epochs=self.freeze_gloca_epochs,
            epoch=epoch,
        )

    def _split_gloca_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        return split_gloca_parameters(self.model.adapter)

    def _gloca_alpha_parameters(self) -> list[nn.Parameter]:
        return gloca_alpha_parameters(self.model.adapter)

    def _current_gloca_alpha(self) -> float | None:
        return current_gloca_alpha(self.model.adapter)

    def compute_gloca_diagnostics(self, epoch: int) -> dict[str, Any] | None:
        if self.model.adapter is None or not self.log_gloca_diagnostics:
            return None
        return compute_gloca_diagnostics(
            model=self.model,
            datamodule=self.datamodule,
            device=self.device,
            epoch=epoch,
            restore_training_fn=self._restore_after_gloca_diagnostics,
        )

    def _restore_after_gloca_diagnostics(self) -> None:
        self.model.train()
        self.model.backbone.eval()
        self.target_encoder.eval()
        self.model.head.target_projector.eval()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        base_lr = float(self.config["trainer"]["lr"])
        predictor_lr = base_lr * float(self.propos_config["predictor_lr_multiplier"])
        weight_decay = float(self.propos_config["weight_decay"])
        projector_params = list(self.model.head.projector.parameters())
        predictor_params = list(self.model.head.predictor.parameters())
        param_groups = [
            {"params": projector_params, "lr": base_lr, "weight_decay": weight_decay, "name": "projector"},
            {"params": predictor_params, "lr": predictor_lr, "weight_decay": weight_decay, "name": "predictor"},
        ]
        if self.model.adapter is not None and not self.freeze_gloca:
            param_groups.extend(
                gloca_optimizer_groups(
                    self.model.adapter,
                    base_lr=base_lr,
                    weight_decay=weight_decay,
                    gloca_lr_multiplier=self.gloca_lr_multiplier,
                    gloca_alpha_lr_multiplier=self.gloca_alpha_lr_multiplier,
                )
            )
        optimizer_name = str(self.propos_config["optimizer"]).lower()
        if optimizer_name != "adamw":
            raise ValueError(f"Unsupported ProPos optimizer '{optimizer_name}'. Only 'adamw' is implemented.")
        return torch.optim.AdamW(param_groups)

    def _sync_device(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _assignment_metrics(labels: torch.Tensor, assignments: torch.Tensor) -> dict[str, float]:
        labels_np = labels.detach().cpu().numpy()
        assignments_np = assignments.detach().cpu().numpy()
        return {
            "ari": float(adjusted_rand_score(labels_np, assignments_np)),
            "nmi": float(normalized_mutual_info_score(labels_np, assignments_np)),
            "acc": float(clustering_accuracy(labels_np, assignments_np)),
        }

    def _latest_estep_metrics(self) -> dict[str, float]:
        if not self.estep_history:
            return {"ari": float("nan"), "nmi": float("nan"), "acc": float("nan")}
        latest = self.estep_history[-1]
        return {
            "ari": float(latest.get("ari", float("nan"))),
            "nmi": float(latest.get("nmi", float("nan"))),
            "acc": float(latest.get("acc", float("nan"))),
        }

    def _latest_estep_metric(self, name: str) -> float:
        return self._latest_estep_metrics()[name]


def clone_target_adapter(adapter: nn.Module | None) -> nn.Module | None:
    if adapter is None:
        return None
    target = copy.deepcopy(adapter)
    for parameter in target.parameters():
        parameter.requires_grad = False
    return target
