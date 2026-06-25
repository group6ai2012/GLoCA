from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch import nn
from tqdm.auto import tqdm

from src.data import ClusteringDataModule
from src.evaluation.clustering_metrics import clustering_accuracy
from src.training.cdc_utils import (
    calibration_target_loss,
    confidence_stats,
    pseudo_label_entropy,
    select_reliable_samples,
    split_sub_batch_indices,
)
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
    desired_gloca_trainable,
    gloca_optimizer_groups,
    set_gloca_trainable,
    split_gloca_parameters,
)


@dataclass
class CDCMetaBatch:
    p_cal_parts: list[torch.Tensor] = field(default_factory=list)
    pseudo_label_parts: list[torch.Tensor] = field(default_factory=list)
    calibrated_confidence_parts: list[torch.Tensor] = field(default_factory=list)
    clustering_confidence_parts: list[torch.Tensor] = field(default_factory=list)
    z_cal_parts: list[torch.Tensor] = field(default_factory=list)
    index_parts: list[torch.Tensor] = field(default_factory=list)
    label_parts: list[torch.Tensor] = field(default_factory=list)
    strong_parts: list[torch.Tensor] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return int(sum(part.shape[0] for part in self.index_parts))

    def append(
        self,
        *,
        p_cal: torch.Tensor,
        pseudo_labels: torch.Tensor,
        calibrated_confidences: torch.Tensor,
        clustering_confidences: torch.Tensor,
        z_cal: torch.Tensor,
        indices: torch.Tensor,
        labels: torch.Tensor | None = None,
        strong: torch.Tensor | None = None,
    ) -> None:
        self.p_cal_parts.append(p_cal.detach().float().cpu())
        self.pseudo_label_parts.append(pseudo_labels.detach().long().cpu())
        self.calibrated_confidence_parts.append(
            calibrated_confidences.detach().float().cpu()
        )
        self.clustering_confidence_parts.append(
            clustering_confidences.detach().float().cpu()
        )
        self.z_cal_parts.append(z_cal.detach().float().cpu())
        self.index_parts.append(indices.detach().long().cpu())
        if labels is not None:
            self.label_parts.append(labels.detach().long().cpu())
        if strong is not None:
            self.strong_parts.append(strong.detach().cpu())

    def finalize(self) -> dict[str, torch.Tensor]:
        if not self.index_parts:
            raise ValueError("Cannot finalize an empty CDC meta-batch.")
        payload = {
            "p_cal": torch.cat(self.p_cal_parts, dim=0),
            "pseudo_labels": torch.cat(self.pseudo_label_parts, dim=0).long(),
            "calibrated_confidences": torch.cat(
                self.calibrated_confidence_parts, dim=0
            ).float(),
            "clustering_confidences": torch.cat(
                self.clustering_confidence_parts, dim=0
            ).float(),
            "z_cal": torch.cat(self.z_cal_parts, dim=0).float(),
            "indices": torch.cat(self.index_parts, dim=0).long(),
        }
        if self.label_parts:
            payload["labels"] = torch.cat(self.label_parts, dim=0).long()
        if self.strong_parts:
            payload["strong"] = torch.cat(self.strong_parts, dim=0)
        actual_b = int(payload["indices"].shape[0])
        assert payload["p_cal"].shape[0] == actual_b
        assert payload["pseudo_labels"].shape[0] == actual_b
        assert payload["z_cal"].shape[0] == actual_b
        assert payload["indices"].shape[0] == actual_b
        assert "strong" in payload and payload["strong"].shape[0] == actual_b
        return payload

    def clear(self) -> None:
        self.p_cal_parts.clear()
        self.pseudo_label_parts.clear()
        self.calibrated_confidence_parts.clear()
        self.clustering_confidence_parts.clear()
        self.z_cal_parts.clear()
        self.index_parts.clear()
        self.label_parts.clear()
        self.strong_parts.clear()


class CDCTrainer:
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
        self.cdc_config = config.get("cdc", {})
        self.progress_enabled = bool(self.cdc_config.get("progress", True))
        self.gloca_lr_multiplier = float(self.cdc_config["gloca_lr_multiplier"])
        self.gloca_alpha_lr_multiplier = float(
            self.cdc_config["gloca_alpha_lr_multiplier"]
        )
        self.freeze_gloca = bool(self.cdc_config["freeze_gloca"])
        self.freeze_gloca_epochs = int(self.cdc_config["freeze_gloca_epochs"])
        self.log_gloca_diagnostics = bool(self.cdc_config["log_gloca_diagnostics"])
        self.physical_batch_size = int(self.config["trainer"]["batch_size"])
        self.meta_batch_size = int(self.cdc_config["meta_batch_size"])
        self.sub_batch_size = int(self.cdc_config["sub_batch_size"])
        self.meta_batch_drop_last = bool(self.cdc_config["meta_batch_drop_last"])
        self.configured_full_batch_size = self.meta_batch_size
        trainer_config = config.get("trainer", {})
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.checkpoint_interval = int(trainer_config.get("checkpoint_interval", 0))
        self.eval_interval = trainer_config.get("eval_interval", "checkpoint")
        self.profile_resources = bool(trainer_config.get("profile_resources", True))
        self.checkpoint_metric_logger = checkpoint_metric_logger
        self.start_epoch = 0
        self.resource_totals = empty_resource_totals()
        self.optimizer_cluster, self.optimizer_calibration = self._build_optimizers()
        self._assert_optimizer_ownership()
        self.epoch_history: list[dict[str, Any]] = []
        self.step_history: list[dict[str, Any]] = []
        self.gloca_diagnostics_history: list[dict[str, Any]] = []
        self.global_step = 0
        self.skipped_clustering_updates = 0
        self.calibration_target_calls = 0
        self.init_logs: dict[str, Any] = {
            "cdc_init_mode": "random",
            "prototype_init_used": False,
            "orthogonalization_used": False,
            "fallback_reason": "CDC initialization has not run yet.",
        }
        self.loss_clustering_final = float("nan")
        self.loss_calibration_final = float("nan")
        self.loss_entropy_final = float("nan")
        self.loss_total_final = float("nan")
        self.gloca_trainable = False
        self.set_gloca_trainable(self._desired_gloca_trainable(epoch=0))
        if resume_from_checkpoint is not None:
            self.load_resumable_checkpoint(Path(resume_from_checkpoint))

    def fit(self) -> None:
        self.assert_backbone_frozen()
        if self.start_epoch == 0 and bool(self.cdc_config.get("prototype_init", True)):
            self.initialize_head()
        elif self.start_epoch == 0:
            self.init_logs = {
                "cdc_init_mode": "random",
                "prototype_init_used": False,
                "orthogonalization_used": False,
                "fallback_reason": "cdc.prototype_init is false.",
            }
        else:
            tqdm.write(f"Resuming CDC from epoch {self.start_epoch}")

        train_loader = self.datamodule.train_dataloader()
        max_epochs = int(self.config["trainer"]["max_epochs"])
        if self.start_epoch >= max_epochs:
            tqdm.write("CDC checkpoint already reached max_epochs; skipping training.")
            return

        for epoch in range(self.start_epoch, max_epochs):
            wall_start = time.perf_counter()
            epoch_timing: dict[str, float] = {}
            self.maybe_update_gloca_freeze_state(epoch)
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
            epoch_logs.update(
                {"ari": float("nan"), "nmi": float("nan"), "acc": float("nan")}
            )
            diagnostics = self.compute_gloca_diagnostics(epoch)
            if diagnostics is not None:
                self.gloca_diagnostics_history.append(diagnostics)
                epoch_logs["gloca_diagnostics"] = diagnostics
            epoch_timing.setdefault("checkpoint_save_time_s", 0.0)
            epoch_timing.setdefault("eval_time_s", 0.0)
            epoch_logs.update(epoch_timing)
            epoch_logs["evaluated"] = bool(run_eval)
            epoch_logs["checkpoint_saved"] = bool(should_checkpoint)
            self.epoch_history.append(epoch_logs)
            if should_checkpoint:
                with timed_section(epoch_timing, "checkpoint_save_time_s"):
                    self.save_epoch_checkpoint(epoch)
            if run_eval:
                with timed_section(epoch_timing, "eval_time_s"):
                    epoch_logs.update(self.evaluate_epoch_metrics())
            epoch_timing.setdefault("checkpoint_save_time_s", 0.0)
            epoch_timing.setdefault("eval_time_s", 0.0)
            epoch_timing["epoch_total_wall_time_s"] = time.perf_counter() - wall_start
            epoch_logs.update(epoch_timing)
            update_resource_totals(self.resource_totals, epoch_timing)
            if should_checkpoint and run_eval and self.checkpoint_metric_logger is not None:
                self.checkpoint_metric_logger(epoch_logs)
            if run_eval:
                tqdm.write(
                    "Epoch "
                    f"{epoch + 1}/{max_epochs} "
                    f"loss={epoch_logs['loss_total_mean']:.4f} "
                    f"nmi={epoch_logs['nmi']:.4f} "
                    f"ari={epoch_logs['ari']:.4f} "
                    f"acc={epoch_logs['acc']:.4f} "
                    f"selected={epoch_logs['selected_pseudo_label_count']} "
                    f"ratio={epoch_logs['reliable_sample_ratio']:.4f}"
                )
            else:
                tqdm.write(
                    "Epoch "
                    f"{epoch + 1}/{max_epochs} "
                    f"loss={epoch_logs['loss_total_mean']:.4f} "
                    f"selected={epoch_logs['selected_pseudo_label_count']} "
                    f"ratio={epoch_logs['reliable_sample_ratio']:.4f} "
                    "eval=skipped"
                )

    def resumable_checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return {
            "checkpoint_version": 1,
            "method": "cdc",
            "epoch": int(epoch),
            "next_epoch": int(epoch) + 1,
            "global_step": int(self.global_step),
            "model_state_dict": self.model.state_dict(),
            "optimizer_cluster_state_dict": self.optimizer_cluster.state_dict(),
            "optimizer_calibration_state_dict": self.optimizer_calibration.state_dict(),
            "config": self.config,
            "epoch_history": self.epoch_history,
            "step_history_tail": [
                {k: v for k, v in step.items() if not str(k).startswith("_")}
                for step in self.step_history[-20:]
            ],
            "gloca_diagnostics_history": self.gloca_diagnostics_history,
            "skipped_clustering_updates": int(self.skipped_clustering_updates),
            "calibration_target_calls": int(self.calibration_target_calls),
            "init_logs": self.init_logs,
            "loss_clustering_final": self.loss_clustering_final,
            "loss_calibration_final": self.loss_calibration_final,
            "loss_entropy_final": self.loss_entropy_final,
            "loss_total_final": self.loss_total_final,
            "gloca_trainable": self.gloca_trainable,
            "resource_totals": self.resource_totals,
            "checkpoint_interval": self.checkpoint_interval,
            "eval_interval": self.eval_interval,
            "rng_state": capture_rng_state(),
        }

    def load_resumable_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("method") != "cdc":
            raise ValueError(f"Expected CDC checkpoint, got {ckpt.get('method')!r}")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer_cluster.load_state_dict(ckpt["optimizer_cluster_state_dict"])
        self.optimizer_calibration.load_state_dict(
            ckpt["optimizer_calibration_state_dict"]
        )
        self.global_step = int(ckpt.get("global_step", 0))
        self.epoch_history = list(ckpt.get("epoch_history", []))
        self.gloca_diagnostics_history = list(
            ckpt.get("gloca_diagnostics_history", [])
        )
        self.skipped_clustering_updates = int(
            ckpt.get("skipped_clustering_updates", 0)
        )
        self.calibration_target_calls = int(ckpt.get("calibration_target_calls", 0))
        self.init_logs = dict(ckpt.get("init_logs", self.init_logs))
        self.loss_clustering_final = float(
            ckpt.get("loss_clustering_final", float("nan"))
        )
        self.loss_calibration_final = float(
            ckpt.get("loss_calibration_final", float("nan"))
        )
        self.loss_entropy_final = float(ckpt.get("loss_entropy_final", float("nan")))
        self.loss_total_final = float(ckpt.get("loss_total_final", float("nan")))
        self.resource_totals = {
            **empty_resource_totals(),
            **dict(ckpt.get("resource_totals", {})),
        }
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

    @torch.no_grad()
    def initialize_head(self) -> None:
        self.model.eval()
        embeddings = []
        for batch in self.datamodule.predict_dataloader():
            image = batch["image"].to(self.device)
            encoded = self.model.encode_view(image)
            embeddings.append(encoded["embedding"].detach().cpu())
        if not embeddings:
            self.init_logs = {
                "cdc_init_mode": "random",
                "prototype_init_used": False,
                "orthogonalization_used": False,
                "fallback_reason": "No deterministic embeddings were available for initialization.",
            }
            return
        all_embeddings = torch.cat(embeddings, dim=0)
        self.init_logs = self.model.head.initialize_from_embeddings(
            all_embeddings,
            kmeans_init=str(self.cdc_config["kmeans_init"]),
            kmeans_n_init=int(self.cdc_config["kmeans_n_init"]),
            kmeans_max_iter=int(self.cdc_config["kmeans_max_iter"]),
            kmeans_tol=float(self.cdc_config["kmeans_tol"]),
            seed=int(self.config["experiment"]["seed"]),
            orthogonalize=bool(self.cdc_config.get("orthogonalize_init", False)),
            orthogonalize_epochs=int(
                self.cdc_config.get("orthogonalize_epochs", 2000)
            ),
            orthogonalize_scale=float(self.cdc_config.get("orthogonalize_scale", 5.0)),
        )

    def train_epoch(
        self, epoch: int, train_loader=None, max_epochs: int | None = None
    ) -> dict[str, Any]:
        train_loader = train_loader or self.datamodule.train_dataloader()
        self.model.train()
        self.model.backbone.eval()
        loss_total_sum = 0.0
        loss_ce_sum = 0.0
        loss_cal_sum = 0.0
        loss_cali_sum = 0.0
        loss_entropy_sum = 0.0
        selected_count_sum = 0
        n_seen = 0
        n_meta_batches = 0
        n_chunks = 0
        skipped_chunks = 0
        skipped_meta_batches = 0
        empty_mini_clusters = 0
        meta_sizes: list[int] = []
        selected_per_meta_batch: list[int] = []
        per_class_selected_nums: list[int] = []
        confidence_parts = []
        calibrated_confidence_parts = []
        pseudo_label_parts = []
        classwise_counts = torch.zeros(
            int(self.model.head.n_clusters), dtype=torch.long
        )

        progress = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}"
            if max_epochs is None
            else f"Epoch {epoch + 1}/{max_epochs}",
            unit="batch",
            leave=False,
            disable=not self.progress_enabled,
        )
        current_meta = CDCMetaBatch()
        for batch in train_loader:
            ready, current_meta = self._add_physical_batch_to_meta_batches(
                current_meta, batch
            )
            for meta_batch in ready:
                logs = self._process_meta_batch_or_skip(meta_batch, epoch)
                if logs is None:
                    skipped_meta_batches += 1
                    continue
                if not torch.isfinite(torch.tensor(logs["loss_total"])):
                    raise RuntimeError(
                        f"Non-finite CDC loss at epoch={epoch} step={self.global_step}: {logs['loss_total']}"
                    )
                self.step_history.append(logs)
                loss_total_sum += float(logs["loss_total"])
                loss_ce_sum += float(logs["loss_ce_mean"])
                loss_cal_sum += float(logs["loss_cal_mean"])
                loss_cali_sum += float(logs["loss_cali_mean"])
                loss_entropy_sum += float(logs["loss_entropy"])
                selected_count_sum += int(logs["selected_pseudo_label_count"])
                n_seen += int(logs["actual_meta_batch_size"])
                n_meta_batches += 1
                n_chunks += int(logs["cdc_num_optimization_chunks"])
                skipped_chunks += int(logs["skipped_clustering_chunks"])
                empty_mini_clusters += int(logs["empty_calibration_mini_clusters"])
                meta_sizes.append(int(logs["actual_meta_batch_size"]))
                selected_per_meta_batch.append(int(logs["selected_pseudo_label_count"]))
                per_class_selected_nums.append(int(logs["per_class_selected_num"]))
                confidence_parts.append(logs["_clustering_confidences"])
                calibrated_confidence_parts.append(logs["_calibrated_confidences"])
                pseudo_label_parts.append(logs["_pseudo_labels"])
                classwise_counts += torch.tensor(
                    logs["classwise_selected_counts"], dtype=torch.long
                )
                progress.set_postfix(
                    loss=f"{logs['loss_total']:.4f}",
                    selected=int(logs["selected_pseudo_label_count"]),
                    meta_B=int(logs["actual_meta_batch_size"]),
                    chunks=int(logs["cdc_num_optimization_chunks"]),
                )
                self.global_step += 1
            progress.update(1)
        if current_meta.num_samples > 0:
            if self.meta_batch_drop_last:
                skipped_meta_batches += 1
            else:
                logs = self._process_meta_batch_or_skip(current_meta, epoch)
                if logs is None:
                    skipped_meta_batches += 1
                else:
                    self.step_history.append(logs)
                    loss_total_sum += float(logs["loss_total"])
                    loss_ce_sum += float(logs["loss_ce_mean"])
                    loss_cal_sum += float(logs["loss_cal_mean"])
                    loss_cali_sum += float(logs["loss_cali_mean"])
                    loss_entropy_sum += float(logs["loss_entropy"])
                    selected_count_sum += int(logs["selected_pseudo_label_count"])
                    n_seen += int(logs["actual_meta_batch_size"])
                    n_meta_batches += 1
                    n_chunks += int(logs["cdc_num_optimization_chunks"])
                    skipped_chunks += int(logs["skipped_clustering_chunks"])
                    empty_mini_clusters += int(logs["empty_calibration_mini_clusters"])
                    meta_sizes.append(int(logs["actual_meta_batch_size"]))
                    selected_per_meta_batch.append(
                        int(logs["selected_pseudo_label_count"])
                    )
                    per_class_selected_nums.append(int(logs["per_class_selected_num"]))
                    confidence_parts.append(logs["_clustering_confidences"])
                    calibrated_confidence_parts.append(logs["_calibrated_confidences"])
                    pseudo_label_parts.append(logs["_pseudo_labels"])
                    classwise_counts += torch.tensor(
                        logs["classwise_selected_counts"], dtype=torch.long
                    )
                    self.global_step += 1
        progress.close()

        divisor = max(1, n_meta_batches)
        clustering_confidences = (
            torch.cat(confidence_parts) if confidence_parts else torch.empty(0)
        )
        calibrated_confidences = (
            torch.cat(calibrated_confidence_parts)
            if calibrated_confidence_parts
            else torch.empty(0)
        )
        pseudo_labels = (
            torch.cat(pseudo_label_parts)
            if pseudo_label_parts
            else torch.empty(0, dtype=torch.long)
        )
        return {
            "epoch": int(epoch),
            "n_batches": int(n_meta_batches),
            "cdc_physical_batch_size": int(self.physical_batch_size),
            "cdc_meta_batch_size_config": int(self.meta_batch_size),
            "cdc_full_batch_size": int(self.meta_batch_size),
            "cdc_meta_batch_size_actual_mean": float(
                sum(meta_sizes) / max(1, len(meta_sizes))
            ),
            "cdc_meta_batch_size_actual_min": int(min(meta_sizes)) if meta_sizes else 0,
            "cdc_meta_batch_size_actual_max": int(max(meta_sizes)) if meta_sizes else 0,
            "cdc_sub_batch_size": int(self.sub_batch_size),
            "cdc_num_meta_batches": int(n_meta_batches),
            "cdc_num_optimization_chunks": int(n_chunks),
            "cdc_num_sub_batches": int(n_chunks),
            "calibration_k": int(self.cdc_config["calibration_k"]),
            "loss_total_mean": float(loss_total_sum / divisor),
            "loss_clustering_mean": float(loss_ce_sum / divisor),
            "loss_ce_mean": float(loss_ce_sum / divisor),
            "loss_calibration_mean": float(loss_cal_sum / divisor),
            "loss_cal_mean": float(loss_cal_sum / divisor),
            "loss_entropy_mean": float(loss_entropy_sum / divisor),
            "loss_cali_mean": float(loss_cali_sum / divisor),
            "selected_pseudo_label_count": int(selected_count_sum),
            "selected_total": int(selected_count_sum),
            "seen_total": int(n_seen),
            "reliable_sample_ratio": float(selected_count_sum / max(1, n_seen)),
            "selected_ratio": float(selected_count_sum / max(1, n_seen)),
            "selected_count_per_meta_batch_mean": float(
                sum(selected_per_meta_batch) / max(1, len(selected_per_meta_batch))
            ),
            "selected_count_per_meta_batch_min": int(min(selected_per_meta_batch))
            if selected_per_meta_batch
            else 0,
            "selected_count_per_meta_batch_max": int(max(selected_per_meta_batch))
            if selected_per_meta_batch
            else 0,
            "selected_count_per_batch_mean": float(
                sum(selected_per_meta_batch) / max(1, len(selected_per_meta_batch))
            ),
            "selected_count_per_batch_min": int(min(selected_per_meta_batch))
            if selected_per_meta_batch
            else 0,
            "selected_count_per_batch_max": int(max(selected_per_meta_batch))
            if selected_per_meta_batch
            else 0,
            "per_class_selected_num": int(per_class_selected_nums[-1])
            if per_class_selected_nums
            else 0,
            "per_class_selected_num_mean": float(
                sum(per_class_selected_nums) / max(1, len(per_class_selected_nums))
            ),
            "per_class_selected_num_min": int(min(per_class_selected_nums))
            if per_class_selected_nums
            else 0,
            "per_class_selected_num_max": int(max(per_class_selected_nums))
            if per_class_selected_nums
            else 0,
            "skipped_meta_batches": int(skipped_meta_batches),
            "skipped_clustering_chunks": int(skipped_chunks),
            "skipped_clustering_sub_batches": int(skipped_chunks),
            "empty_calibration_mini_clusters": int(empty_mini_clusters),
            "pseudo_label_entropy": pseudo_label_entropy(
                pseudo_labels, int(self.model.head.n_clusters)
            ),
            "classwise_selected_counts": classwise_counts.tolist(),
            "selected_count_per_class": classwise_counts.tolist(),
            **confidence_stats(clustering_confidences, "clustering_confidence"),
            **confidence_stats(calibrated_confidences, "calibrated_confidence"),
        }

    def train_step(
        self, batch: dict[str, Any], epoch: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        meta_batch = CDCMetaBatch()
        ready, meta_batch = self._add_physical_batch_to_meta_batches(
            meta_batch, batch, force_single_meta=True
        )
        if ready:
            meta_batch = ready[0]
        logs = self._process_meta_batch_or_skip(meta_batch, epoch)
        if logs is None:
            logs = self._empty_step_logs(batch_size=int(batch["index"].shape[0]))
        return torch.tensor(float(logs["loss_total"]), device=self.device), logs

    def _process_meta_batch_or_skip(
        self, meta_batch: CDCMetaBatch, epoch: int
    ) -> dict[str, Any] | None:
        actual_meta_b = meta_batch.num_samples
        if actual_meta_b < int(self.model.head.n_clusters):
            meta_batch.clear()
            return None
        if int(self.cdc_config["calibration_k"]) > actual_meta_b:
            raise ValueError(
                f"cdc.calibration_k={int(self.cdc_config['calibration_k'])} exceeds actual meta-batch size {actual_meta_b}."
            )
        meta = meta_batch.finalize()
        selection = select_reliable_samples(
            meta["p_cal"], self._per_class_selected_num(actual_meta_b)
        )
        selected_mask = selection["selected_mask"].cpu()
        pseudo_labels = meta["pseudo_labels"].cpu()
        selected_count = int(selection["selected_count"])
        super_target, empty_mini_clusters = self._calibration_targets_from_embeddings(
            meta["z_cal"], epoch
        )
        chunks = split_sub_batch_indices(
            actual_meta_b, self.sub_batch_size, device=torch.device("cpu")
        )
        loss_ce_values: list[float] = []
        loss_cal_values: list[float] = []
        loss_entropy_values: list[float] = []
        loss_cali_values: list[float] = []
        skipped_chunks = 0

        self.model.head.clustering_head.train()
        self.model.head.calibration_head.train()
        self.model.backbone.eval()
        if self.model.adapter is not None:
            self.model.adapter.train(self.gloca_trainable)

        for sub_idx in chunks:
            sub_selected = selected_mask[sub_idx]
            if bool(sub_selected.any()):
                self.optimizer_cluster.zero_grad(set_to_none=True)
                strong_images = meta["strong"][sub_idx].to(
                    self.device, non_blocking=True
                )
                strong_encoded = self.model.encode_view(strong_images)
                clustering_logits = self.model.head.clustering_head(
                    strong_encoded["embedding"]
                )
                sub_selected_device = sub_selected.to(self.device)
                chunk_pseudo_labels = pseudo_labels[sub_idx].to(self.device)
                loss_ce = F.cross_entropy(
                    clustering_logits[sub_selected_device],
                    chunk_pseudo_labels[sub_selected_device],
                )
                loss_ce.backward()
                self.optimizer_cluster.step()
                loss_ce_values.append(float(loss_ce.detach().cpu()))
            else:
                skipped_chunks += 1
                self.skipped_clustering_updates += 1

            self.optimizer_calibration.zero_grad(set_to_none=True)
            z_cal = meta["z_cal"][sub_idx].to(self.device, non_blocking=True)
            target = super_target[sub_idx].to(self.device, non_blocking=True)
            calibration_logits = self.model.head.calibration_head(z_cal)
            loss_cali, loss_cal, loss_entropy = calibration_target_loss(
                calibration_logits,
                target,
                entropy_weight=float(self.cdc_config["w_en"]),
            )
            loss_cali.backward()
            self.optimizer_calibration.step()
            loss_cal_values.append(float(loss_cal.detach().cpu()))
            loss_entropy_values.append(float(loss_entropy.detach().cpu()))
            loss_cali_values.append(float(loss_cali.detach().cpu()))

        loss_ce_mean = float(sum(loss_ce_values) / max(1, len(loss_ce_values)))
        loss_cal_mean = float(sum(loss_cal_values) / max(1, len(loss_cal_values)))
        loss_entropy_mean = float(
            sum(loss_entropy_values) / max(1, len(loss_entropy_values))
        )
        loss_cali_mean = float(sum(loss_cali_values) / max(1, len(loss_cali_values)))
        loss_total = loss_ce_mean + loss_cali_mean

        self.loss_clustering_final = loss_ce_mean
        self.loss_calibration_final = loss_cal_mean
        self.loss_entropy_final = loss_entropy_mean
        self.loss_total_final = loss_total
        logs = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "batch_size": int(actual_meta_b),
            "actual_meta_batch_size": int(actual_meta_b),
            "cdc_physical_batch_size": int(self.physical_batch_size),
            "cdc_meta_batch_size_config": int(self.meta_batch_size),
            "cdc_full_batch_size": int(self.meta_batch_size),
            "cdc_actual_full_batch_size": int(actual_meta_b),
            "cdc_meta_batch_size_actual": int(actual_meta_b),
            "cdc_sub_batch_size": int(self.sub_batch_size),
            "cdc_num_optimization_chunks": int(len(chunks)),
            "cdc_num_sub_batches": int(len(chunks)),
            "calibration_k": int(self.cdc_config["calibration_k"]),
            "loss_clustering": loss_ce_mean,
            "loss_ce_mean": loss_ce_mean,
            "loss_calibration": loss_cal_mean,
            "loss_cal_mean": loss_cal_mean,
            "loss_entropy": loss_entropy_mean,
            "loss_cali_mean": loss_cali_mean,
            "loss_total": loss_total,
            "selected_pseudo_label_count": selected_count,
            "selected_total": selected_count,
            "seen_total": int(actual_meta_b),
            "reliable_sample_ratio": float(selection["reliable_sample_ratio"]),
            "selected_ratio": float(selection["reliable_sample_ratio"]),
            "per_class_selected_num": int(selection["per_class_selected_num"]),
            "classwise_selected_counts": selection["classwise_selected_counts"],
            "selected_count_per_class": selection["classwise_selected_counts"],
            "pseudo_label_entropy": pseudo_label_entropy(
                pseudo_labels[selected_mask].detach(), int(self.model.head.n_clusters)
            ),
            "skipped_clustering_update": selected_count == 0,
            "skipped_clustering_chunks": int(skipped_chunks),
            "skipped_clustering_sub_batches": int(skipped_chunks),
            "empty_calibration_mini_clusters": int(empty_mini_clusters),
            "_clustering_confidences": meta["clustering_confidences"],
            "_calibrated_confidences": meta["calibrated_confidences"],
            "_pseudo_labels": pseudo_labels.detach().cpu(),
        }
        meta_batch.clear()
        return logs

    @torch.no_grad()
    def _add_physical_batch_to_meta_batches(
        self,
        current_meta: CDCMetaBatch,
        batch: dict[str, Any],
        *,
        force_single_meta: bool = False,
    ) -> tuple[list[CDCMetaBatch], CDCMetaBatch]:
        stats = self._collect_physical_batch_stats(batch)
        ready: list[CDCMetaBatch] = []
        offset = 0
        n_samples = int(stats["indices"].shape[0])
        while offset < n_samples:
            if force_single_meta:
                take = n_samples - offset
            else:
                available = self.meta_batch_size - current_meta.num_samples
                take = min(available, n_samples - offset)
            self._append_stats_slice(current_meta, stats, offset, offset + take)
            offset += take
            if (
                not force_single_meta
                and current_meta.num_samples >= self.meta_batch_size
            ):
                ready.append(current_meta)
                current_meta = CDCMetaBatch()
        return ready, current_meta

    @torch.no_grad()
    def _collect_physical_batch_stats(
        self, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        if "weak" in batch:
            weak = batch["weak"].to(self.device)
            calibration_view = batch["calibration"].to(self.device)
            strong = batch["strong"]
        else:
            weak = batch["views"][0].to(self.device)
            calibration_view = batch.get("calibration", batch["views"][0]).to(
                self.device
            )
            strong = batch["views"][1]
        cluster_was_training = self.model.head.clustering_head.training
        calibration_was_training = self.model.head.calibration_head.training
        # Meta-batch statistics are no-grad passes. Keep CDC heads in eval mode so
        # BatchNorm running stats are updated only during optimizer chunks.
        self.model.head.clustering_head.eval()
        self.model.head.calibration_head.eval()
        try:
            weak_encoded = self.model.encode_view(weak)
            weak_out = self.model.head(weak_encoded["embedding"])
            cal_encoded = self.model.encode_view(calibration_view)
            payload = {
                "p_cal": weak_out["calibration_probabilities"].detach().cpu(),
                "pseudo_labels": weak_out["calibration_predictions"].detach().cpu(),
                "calibrated_confidences": weak_out["calibrated_confidences"]
                .detach()
                .cpu(),
                "clustering_confidences": weak_out["clustering_confidences"]
                .detach()
                .cpu(),
                "z_cal": cal_encoded["embedding"].detach().float().cpu(),
                "indices": batch["index"].detach().long().cpu(),
            }
            if "label" in batch:
                payload["labels"] = batch["label"].detach().long().cpu()
            payload["strong"] = strong.detach().cpu()
        finally:
            self.model.head.clustering_head.train(cluster_was_training)
            self.model.head.calibration_head.train(calibration_was_training)
        return payload

    @staticmethod
    def _append_stats_slice(
        meta_batch: CDCMetaBatch,
        stats: dict[str, torch.Tensor],
        start: int,
        end: int,
    ) -> None:
        meta_batch.append(
            p_cal=stats["p_cal"][start:end],
            pseudo_labels=stats["pseudo_labels"][start:end],
            calibrated_confidences=stats["calibrated_confidences"][start:end],
            clustering_confidences=stats["clustering_confidences"][start:end],
            z_cal=stats["z_cal"][start:end],
            indices=stats["indices"][start:end],
            labels=stats.get("labels", None)[start:end] if "labels" in stats else None,
            strong=stats.get("strong", None)[start:end] if "strong" in stats else None,
        )

    def _empty_step_logs(self, batch_size: int) -> dict[str, Any]:
        return {
            "batch_size": int(batch_size),
            "actual_meta_batch_size": int(batch_size),
            "loss_total": 0.0,
            "loss_ce_mean": 0.0,
            "loss_cal_mean": 0.0,
            "loss_entropy": 0.0,
            "loss_cali_mean": 0.0,
            "selected_pseudo_label_count": 0,
            "reliable_sample_ratio": 0.0,
            "classwise_selected_counts": [
                0 for _ in range(int(self.model.head.n_clusters))
            ],
            "cdc_num_optimization_chunks": 0,
            "cdc_num_sub_batches": 0,
            "skipped_clustering_chunks": 0,
            "skipped_clustering_sub_batches": 0,
            "empty_calibration_mini_clusters": 0,
            "_clustering_confidences": torch.empty(0),
            "_calibrated_confidences": torch.empty(0),
            "_pseudo_labels": torch.empty(0, dtype=torch.long),
        }

    @torch.no_grad()
    def _calibration_targets_from_embeddings(
        self, features_cpu: torch.Tensor, epoch: int
    ) -> tuple[torch.Tensor, int]:
        self.calibration_target_calls += 1
        features = features_cpu.detach().float().cpu()
        was_training = self.model.head.clustering_head.training
        self.model.head.clustering_head.eval()
        try:
            clustering_probs = (
                F.softmax(
                    self.model.head.clustering_head(features.to(self.device)), dim=1
                )
                .detach()
                .cpu()
            )
        finally:
            self.model.head.clustering_head.train(was_training)
        n_samples = int(features.shape[0])
        requested_k = int(self.cdc_config["calibration_k"])
        if requested_k > n_samples:
            raise ValueError(
                f"cdc.calibration_k={requested_k} exceeds actual meta-batch size {n_samples}."
            )
        cluster_num = requested_k
        if cluster_num == 1:
            assignments = torch.zeros(n_samples, dtype=torch.long)
        else:
            kmeans = fit_kmeans(
                F.normalize(features.detach().float().cpu(), p=1, dim=1),
                cluster_num,
                spherical=False,
                init=str(self.cdc_config["kmeans_init"]),
                n_init=int(self.cdc_config["kmeans_n_init"]),
                max_iter=int(self.cdc_config["kmeans_max_iter"]),
                tol=float(self.cdc_config["kmeans_tol"]),
                seed=int(self.config["experiment"]["seed"])
                + int(epoch)
                + int(self.global_step),
                device=torch.device("cpu"),
            )
            assignments = kmeans["assignments"].long()
        targets = []
        mean_target = clustering_probs.mean(dim=0)
        assignments_device = assignments.to(clustering_probs.device)
        empty_mini_clusters = 0
        for cluster_id in range(cluster_num):
            mask = assignments_device == cluster_id
            if not bool(mask.any()):
                empty_mini_clusters += 1
            targets.append(
                clustering_probs[mask].mean(dim=0) if bool(mask.any()) else mean_target
            )
        target_lookup = torch.stack(targets, dim=0)
        return target_lookup[assignments_device].detach().cpu(), int(
            empty_mini_clusters
        )

    @torch.no_grad()
    def extract_deterministic_features(self) -> dict[str, Any]:
        self.model.eval()
        embedding_parts: list[torch.Tensor] = []
        attention_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        index_parts: list[torch.Tensor] = []
        assignment_parts: list[torch.Tensor] = []
        clustering_confidence_parts: list[torch.Tensor] = []
        calibrated_confidence_parts: list[torch.Tensor] = []
        pseudo_label_parts: list[torch.Tensor] = []
        image_ids: list[str] = []
        patch_grid = None
        for batch in self.datamodule.predict_dataloader():
            image = batch["image"].to(self.device)
            encoded = self.model.encode_view(image)
            out = self.model.head(encoded["embedding"])
            embedding_parts.append(encoded["embedding"].detach().cpu())
            if encoded["attention"] is not None:
                attention_parts.append(encoded["attention"].detach().cpu())
            label_parts.append(batch["label"].detach().cpu())
            index_parts.append(batch["index"].detach().cpu())
            assignment_parts.append(out["predictions"].detach().cpu())
            pseudo_label_parts.append(out["clustering_predictions"].detach().cpu())
            clustering_confidence_parts.append(
                out["clustering_confidences"].detach().cpu()
            )
            calibrated_confidence_parts.append(
                out["calibrated_confidences"].detach().cpu()
            )
            image_ids.extend(batch["image_id"])
            patch_grid = encoded["patch_grid"]

        indices = torch.cat(index_parts, dim=0).long()
        order = torch.argsort(indices)
        attention = (
            torch.cat(attention_parts, dim=0)[order].contiguous()
            if attention_parts
            else None
        )
        ordered_ids = [image_ids[i] for i in order.tolist()]
        return {
            "embeddings": torch.cat(embedding_parts, dim=0)[order].float().contiguous(),
            "attention": attention,
            "labels": torch.cat(label_parts, dim=0)[order].long().contiguous(),
            "indices": indices[order].contiguous(),
            "image_ids": ordered_ids,
            "assignments": torch.cat(assignment_parts, dim=0)[order]
            .long()
            .contiguous(),
            "pseudo_labels": torch.cat(pseudo_label_parts, dim=0)[order]
            .long()
            .contiguous(),
            "confidence": torch.cat(clustering_confidence_parts, dim=0)[order]
            .float()
            .contiguous(),
            "calibrated_confidence": torch.cat(calibrated_confidence_parts, dim=0)[
                order
            ]
            .float()
            .contiguous(),
            "patch_grid": patch_grid,
        }

    @torch.no_grad()
    def evaluate_epoch_metrics(self) -> dict[str, float]:
        self.model.eval()
        label_parts: list[torch.Tensor] = []
        assignment_parts: list[torch.Tensor] = []
        index_parts: list[torch.Tensor] = []
        for batch in self.datamodule.predict_dataloader():
            image = batch["image"].to(self.device)
            encoded = self.model.encode_view(image)
            out = self.model.head(encoded["embedding"])
            label_parts.append(batch["label"].detach().cpu())
            assignment_parts.append(out["predictions"].detach().cpu())
            index_parts.append(batch["index"].detach().cpu())

        if not assignment_parts:
            return {"ari": float("nan"), "nmi": float("nan"), "acc": float("nan")}
        indices = torch.cat(index_parts, dim=0).long()
        order = torch.argsort(indices)
        labels = torch.cat(label_parts, dim=0)[order].long().numpy()
        assignments = torch.cat(assignment_parts, dim=0)[order].long().numpy()
        return {
            "ari": float(adjusted_rand_score(labels, assignments)),
            "nmi": float(normalized_mutual_info_score(labels, assignments)),
            "acc": float(clustering_accuracy(labels, assignments)),
        }

    def _batch_views(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "weak" in batch:
            weak = batch["weak"]
            strong = batch["strong"]
            calibration_view = batch["calibration"]
        else:
            weak = batch["views"][0]
            strong = batch["views"][1]
            calibration_view = batch.get("calibration", weak)
        return (
            weak.to(self.device),
            strong.to(self.device),
            calibration_view.to(self.device),
        )

    def _per_class_selected_num(self, batch_size: int) -> int | None:
        raw = self.cdc_config.get("per_class_selected_num")
        if raw is None or str(raw).lower() == "auto":
            return None
        return int(raw)

    def assert_backbone_frozen(self) -> None:
        assert_backbone_frozen(self.model.backbone)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "cdc_logs": self.training_logs(),
        }

    def training_logs(self) -> dict[str, Any]:
        return {
            "loss_clustering_final": self.loss_clustering_final,
            "loss_calibration_final": self.loss_calibration_final,
            "loss_entropy_final": self.loss_entropy_final,
            "loss_total_final": self.loss_total_final,
            "skipped_clustering_updates": int(self.skipped_clustering_updates),
            "epoch_history": self.epoch_history,
            "last_steps": [
                {k: v for k, v in step.items() if not k.startswith("_")}
                for step in self.step_history[-20:]
            ],
            "global_step": int(self.global_step),
            "cdc_physical_batch_size": int(self.physical_batch_size),
            "cdc_meta_batch_size_config": int(self.meta_batch_size),
            "cdc_full_batch_size": int(self.configured_full_batch_size),
            "cdc_sub_batch_size": int(self.sub_batch_size),
            "cdc_meta_batch_drop_last": bool(self.meta_batch_drop_last),
            "calibration_target_calls": int(self.calibration_target_calls),
            "calibration_k": int(self.cdc_config["calibration_k"]),
            "super_cluster_num": int(self.cdc_config["calibration_k"]),
            "per_class_selected_num": self.cdc_config.get("per_class_selected_num"),
            "resource_totals": self.resource_totals,
            "eval_interval": self.eval_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "profile_resources": self.profile_resources,
            "optimizer_cluster_groups": [
                group.get("name", "") for group in self.optimizer_cluster.param_groups
            ],
            "optimizer_calibration_groups": [
                group.get("name", "")
                for group in self.optimizer_calibration.param_groups
            ],
            "gloca_diagnostics_history": self.gloca_diagnostics_history,
            "gloca_lr_multiplier": self.gloca_lr_multiplier,
            "gloca_alpha_lr_multiplier": self.gloca_alpha_lr_multiplier,
            "freeze_gloca": self.freeze_gloca,
            "freeze_gloca_epochs": self.freeze_gloca_epochs,
            "gloca_trainable": self.gloca_trainable,
            **self.init_logs,
        }

    def compute_gloca_diagnostics(self, epoch: int) -> dict[str, Any] | None:
        if self.model.adapter is None or not self.log_gloca_diagnostics:
            return None
        return compute_gloca_diagnostics(
            model=self.model,
            datamodule=self.datamodule,
            device=self.device,
            epoch=epoch,
        )

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

    def _build_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        base_lr = float(self.config["trainer"]["lr"])
        weight_decay = float(self.cdc_config["weight_decay"])
        cluster_groups = [
            {
                "params": list(self.model.head.clustering_head.parameters()),
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "cdc_clustering_head",
            },
        ]
        if self.model.adapter is not None and not self.freeze_gloca:
            cluster_groups.extend(
                gloca_optimizer_groups(
                    self.model.adapter,
                    base_lr=base_lr,
                    weight_decay=weight_decay,
                    gloca_lr_multiplier=self.gloca_lr_multiplier,
                    gloca_alpha_lr_multiplier=self.gloca_alpha_lr_multiplier,
                )
            )
        optimizer_name = str(self.cdc_config["optimizer"]).lower()
        if optimizer_name != "adamw":
            raise ValueError(
                f"Unsupported CDC optimizer '{optimizer_name}'. Only 'adamw' is implemented."
            )
        calibration_groups = [
            {
                "params": list(self.model.head.calibration_head.parameters()),
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "cdc_calibration_head",
            }
        ]
        return torch.optim.AdamW(cluster_groups), torch.optim.AdamW(calibration_groups)

    def _assert_optimizer_ownership(self) -> None:
        self.assert_backbone_frozen()
        cluster_ids = {
            id(parameter)
            for group in self.optimizer_cluster.param_groups
            for parameter in group["params"]
        }
        calibration_ids = {
            id(parameter)
            for group in self.optimizer_calibration.param_groups
            for parameter in group["params"]
        }
        if cluster_ids & calibration_ids:
            raise RuntimeError(
                "CDC cluster and calibration optimizers share parameters."
            )
        backbone_ids = {id(parameter) for parameter in self.model.backbone.parameters()}
        if cluster_ids & backbone_ids or calibration_ids & backbone_ids:
            raise RuntimeError(
                "CDC optimizers must not include DINOv2 backbone parameters."
            )
        calibration_head_ids = {
            id(parameter) for parameter in self.model.head.calibration_head.parameters()
        }
        clustering_head_ids = {
            id(parameter) for parameter in self.model.head.clustering_head.parameters()
        }
        if cluster_ids & calibration_head_ids:
            raise RuntimeError(
                "CDC cluster optimizer must not include calibration-head parameters."
            )
        if calibration_ids & clustering_head_ids:
            raise RuntimeError(
                "CDC calibration optimizer must not include clustering-head parameters."
            )
        if self.model.adapter is not None:
            adapter_ids = {
                id(parameter) for parameter in self.model.adapter.parameters()
            }
            if calibration_ids & adapter_ids:
                raise RuntimeError(
                    "CDC calibration optimizer must not include GLoCA parameters."
                )
