from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import torch

from src.data import ClusteringDataModule
from src.evaluation import compute_clustering_metrics
from src.evaluation.assignment_schema import DEC_IDEC_METRIC_FIELDS
from src.experiments.config import save_experiment_config
from src.experiments.outputs import append_csv_row, write_outputs
from src.features import DINOv2Backbone
from src.models.baselines.dec_idec import DINOCLSDECModel
from src.runners.common import apply_runtime_settings, assert_backbone_frozen, finalize_run, prepare_runner_context
from src.runners.diagnostics import attention_diagnostics, cluster_diagnostics, embedding_diagnostics
from src.runners.embedding_export import extract_deterministic_embeddings
from src.training.checkpointing import resolve_resume_checkpoint
from src.training.dec_idec_trainer import DECIDECTrainer, parse_target_update_interval
from src.utils import ExperimentResult


def run_dec_idec(config: dict[str, Any]) -> ExperimentResult:
    config = deepcopy(config)
    runtime_logs = apply_runtime_settings(config)
    head_name = str(config["head"]["name"]).lower()
    if head_name not in {"dec", "idec"}:
        raise ValueError(f"run_dec_idec only supports head.name in {{'dec', 'idec'}}, got {head_name!r}")
    if bool(config.get("gloca", {}).get("enabled", False)):
        raise ValueError("Standalone DEC/IDEC baselines must not enable GLoCA.")

    ctx = prepare_runner_context(config)
    spec = ctx.spec
    if config["dataset"]["training_views"] == "auto":
        config["dataset"]["training_views"] = 1
    _normalize_baseline_metadata(config)
    _apply_output_metadata_defaults(config, head_name)

    output_dir = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, output_dir / "config.yaml")

    seed = ctx.seed
    datamodule = ClusteringDataModule(
        spec=spec,
        image_size=int(config["backbone"]["image_size"]),
        batch_size=int(config["trainer"]["batch_size"]),
        num_workers=int(config["trainer"]["num_workers"]),
        training_mode="single_view",
        training_views=1,
    )
    backbone = DINOv2Backbone(
        variant=config["backbone"]["variant"],
        image_size=int(config["backbone"]["image_size"]),
        freeze=bool(config["backbone"]["freeze"]),
    )
    assert_backbone_frozen(backbone)

    device = ctx.device
    total_start = time.perf_counter()
    extracted = extract_deterministic_embeddings(
        backbone=backbone,
        adapter=None,
        datamodule=datamodule,
        device=device,
        normalize_cls=bool(config.get("gloca", {}).get("normalize_output", True)),
    )
    x = extracted["embeddings"].float()
    input_dim = int(config.get("baseline", {}).get("input_dim", x.shape[1]))
    hidden_dims = tuple(int(dim) for dim in config.get("baseline", {}).get("hidden_dims", [512, 512, 2048]))
    latent_dim = int(config.get("baseline", {}).get("latent_dim", 64))
    if input_dim != x.shape[1]:
        raise ValueError(f"Configured input_dim={input_dim} does not match cached CLS dim={x.shape[1]}")

    model = DINOCLSDECModel(
        n_clusters=int(config["head"]["n_clusters"]),
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        alpha=float(config.get("baseline", {}).get("alpha", 1.0)),
    )

    pretrain_epochs = int(config.get("baseline", {}).get("pretrain_epochs", 20))
    refine_epochs = int(config.get("baseline", {}).get("refine_epochs", 10))
    pretrain_lr = float(config.get("baseline", {}).get("pretrain_lr", config["trainer"].get("lr", 1.0e-3)))
    refine_lr = float(config.get("baseline", {}).get("refine_lr", config["trainer"].get("lr", 1.0e-4)))
    lambda_recon = float(config.get("baseline", {}).get("lambda_recon", 0.1))
    alpha = float(config.get("baseline", {}).get("alpha", 1.0))
    target_update_mode = str(config["baseline"]["target_update_mode"])
    target_update_interval = config["baseline"]["target_update_interval"]
    checkpoint_dir = output_dir / "checkpoints"
    resume_path = resolve_resume_checkpoint(
        config.get("trainer", {}).get("resume_from_checkpoint", "auto"),
        checkpoint_dir,
    )

    def log_checkpoint_metrics(phase_log: dict[str, Any]) -> None:
        epoch = int(phase_log["epoch"])
        phase = str(phase_log["phase"])
        append_csv_row(
            output_dir / "checkpoint_metrics.csv",
            {
                "experiment": config["experiment"]["name"],
                "head": head_name,
                "backbone": config["backbone"]["variant"],
                "dataset": spec.name,
                "seed": seed,
                "gloca": "disabled",
                "n_clusters": int(config["head"]["n_clusters"]),
                "phase": phase,
                "epoch": epoch,
                "epoch_1based": epoch + 1,
                "checkpoint": f"checkpoints/epoch_{phase}_{epoch + 1:04d}.ckpt",
                "loss": float(phase_log.get("loss", float("nan"))),
                "kl_loss": float(phase_log.get("kl_loss", float("nan"))),
                "recon_loss": float(phase_log.get("recon_loss", float("nan"))),
                "train_epoch_time_s": float(phase_log.get("phase_train_epoch_time_s", 0.0)),
                "checkpoint_save_time_s": float(phase_log.get("checkpoint_save_time_s", 0.0)),
                "eval_time_s": float(phase_log.get("eval_time_s", 0.0)),
                "epoch_total_wall_time_s": float(phase_log.get("phase_epoch_total_wall_time_s", 0.0)),
            },
        )

    trainer = DECIDECTrainer(
        model=model,
        x=x,
        config=config,
        mode=head_name,
        n_clusters=int(config["head"]["n_clusters"]),
        seed=seed,
        device=device,
        labels=extracted["labels"],
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_path,
        checkpoint_metric_logger=log_checkpoint_metrics,
    )

    head_train_start = time.perf_counter()
    trainer.fit()
    head_train_time_s = time.perf_counter() - head_train_start
    trainer_logs = trainer.training_logs()
    pretrain_time_s = sum(
        float(row.get("phase_train_epoch_time_s", 0.0))
        for row in trainer.phase_history
        if row.get("phase") == "pretrain"
    )
    refine_time_s = sum(
        float(row.get("phase_train_epoch_time_s", 0.0))
        for row in trainer.phase_history
        if row.get("phase") == "refine"
    )

    inference_start = time.perf_counter()
    latent, assignments = trainer.predict_all()
    inference_time_s = time.perf_counter() - inference_start
    total_time_s = time.perf_counter() - total_start

    assignment_extras = {
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "hidden_dims": list(hidden_dims),
        "pretrain_epochs": pretrain_epochs,
        "refine_epochs": refine_epochs,
        "pretrain_lr": pretrain_lr,
        "refine_lr": refine_lr,
        "lambda_recon": lambda_recon,
        "alpha": alpha,
        "target_update_mode": target_update_mode,
        "target_update_interval": target_update_interval,
    }
    metrics_extras = {
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "hidden_dims": "-".join(str(dim) for dim in hidden_dims),
        "pretrain_epochs": pretrain_epochs,
        "refine_epochs": refine_epochs,
        "pretrain_lr": pretrain_lr,
        "refine_lr": refine_lr,
        "lambda_recon": lambda_recon,
        "alpha": alpha,
        "target_update_mode": target_update_mode,
        "target_update_interval": target_update_interval,
        "spherical": False,
        "_extra_fields": DEC_IDEC_METRIC_FIELDS,
    }
    logs_extras = {
        "standalone_autoencoder_baseline": True,
        "uses_cached_backbone_features": True,
        "backbone_requires_grad_false": not any(param.requires_grad for param in backbone.parameters()),
        "deterministic_single_view_transform": True,
        "head": head_name,
        "backbone": config["backbone"]["variant"],
        "gloca": "disabled",
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "latent_dim": latent_dim,
        "lambda_recon": lambda_recon,
        "alpha": alpha,
        "pretrain_lr": pretrain_lr,
        "refine_lr": refine_lr,
        "target_update_mode": target_update_mode,
        "target_update_interval": target_update_interval,
        "pretrain_epochs": pretrain_epochs,
        "refine_epochs": refine_epochs,
        **trainer_logs,
        "pretrain_final_loss": trainer.pretrain_losses[-1] if trainer.pretrain_losses else None,
        "refine_final_loss": trainer.refine_losses[-1] if trainer.refine_losses else None,
        "pretrain_time_s": pretrain_time_s,
        "refine_time_s": refine_time_s,
        "head_train_time_s": head_train_time_s,
        "inference_time_s": inference_time_s,
        "total_time_s": total_time_s,
        "backbone_cache_time_s": extracted["cache_time_s"],
        "embedding_shape": list(latent.shape),
        "attention_shape": None,
        "num_workers": int(config["trainer"]["num_workers"]),
        "checkpoint_metrics_path": "checkpoint_metrics.csv",
        **runtime_logs,
    }
    result = finalize_run(
        output_dir=output_dir,
        config=config,
        spec=spec,
        head=head_name,
        image_ids=extracted["image_ids"],
        labels=extracted["labels"],
        assignments=assignments,
        embeddings=latent,
        attention=None,
        patch_grid=extracted["patch_grid"],
        seed=seed,
        backbone_cache_time_s=float(extracted["cache_time_s"]),
        head_train_time_s=head_train_time_s,
        total_time_s=total_time_s,
        inference_time_s=inference_time_s,
        uses_cached_backbone_features=True,
        metrics_extras=metrics_extras,
        logs_extras=logs_extras,
        assignment_extras=assignment_extras,
        metrics_fn=compute_clustering_metrics,
        cluster_diagnostics_fn=cluster_diagnostics,
        embedding_diagnostics_fn=embedding_diagnostics,
        attention_diagnostics_fn=attention_diagnostics,
        writer=write_outputs,
    )
    torch.save({"state_dict": trainer.model.state_dict(), "config": config}, output_dir / "checkpoint.ckpt")
    return result


def _normalize_baseline_metadata(config: dict[str, Any]) -> None:
    baseline = config["baseline"]
    target_update_mode, target_update_interval = parse_target_update_interval(baseline.get("target_update_interval"))
    baseline["target_update_mode"] = target_update_mode
    baseline["target_update_interval"] = target_update_interval


def _apply_output_metadata_defaults(config: dict[str, Any], head_name: str) -> None:
    baseline = config.setdefault("baseline", {})
    baseline["head"] = head_name
    baseline["gloca"] = "disabled"
