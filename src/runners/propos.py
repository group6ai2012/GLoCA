from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import torch

from src.data import ClusteringDataModule
from src.evaluation import compute_clustering_metrics
from src.experiments.config import save_experiment_config, validate_propos_config
from src.experiments.outputs import append_csv_row, write_outputs
from src.features import DINOv2Backbone
from src.models import ClusteringBaseModel, build_adapter
from src.models.clustering import ProPosHead, fit_kmeans
from src.runners.common import (
    apply_runtime_settings,
    assert_backbone_frozen,
    finalize_run,
    prepare_runner_context,
)
from src.runners.diagnostics import attention_diagnostics, cluster_diagnostics, embedding_diagnostics
from src.training.checkpointing import resolve_resume_checkpoint
from src.training.propos_trainer import ProPosTrainer
from src.utils import ExperimentResult, resolve_gloca_name


PROPOS_EXTRA_METRICS = [
    "loss_psa_final",
    "loss_psl_final",
    "loss_total_final",
    "kmeans_interval",
    "warmup_epochs",
    "lambda_psl",
    "sigma",
    "temperature",
    "ema_momentum",
    "ema_momentum_final",
    "projection_dim",
    "n_empty_cluster_batches",
    "n_invalid_psl_batches",
    "kmeans_backend",
    "kmeans_init",
]


def run_propos(config: dict[str, Any]) -> ExperimentResult:
    config = deepcopy(config)
    runtime_logs = apply_runtime_settings(config)
    validate_propos_config(config)
    if config["head"]["name"] != "propos":
        raise ValueError(f"run_propos only supports head.name='propos', got {config['head']['name']!r}")

    ctx = prepare_runner_context(config, expected_head="propos")
    spec = ctx.spec
    config["dataset"]["training_views"] = 2
    config["head"]["training_mode"] = "contrastive_two_view"

    output_dir = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, output_dir / "config.yaml")

    seed = ctx.seed
    device = ctx.device
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    datamodule = ClusteringDataModule(
        spec=spec,
        image_size=int(config["backbone"]["image_size"]),
        batch_size=int(config["trainer"]["batch_size"]),
        num_workers=int(config["trainer"]["num_workers"]),
        training_mode="contrastive_two_view",
        training_views=2,
    )

    backbone = DINOv2Backbone(
        variant=config["backbone"]["variant"],
        image_size=int(config["backbone"]["image_size"]),
        freeze=bool(config["backbone"]["freeze"]),
    )
    assert_backbone_frozen(backbone)

    adapter = build_adapter(config, input_dim=backbone.output_dim)
    gloca_name = resolve_gloca_name(config)
    embedding_dim = backbone.output_dim if adapter is None else int(config["gloca"]["embedding_dim"])
    head = ProPosHead(
        n_clusters=int(config["head"]["n_clusters"]),
        embedding_dim=embedding_dim,
        projection_dim=int(config["propos"]["projection_dim"]),
        projection_hidden_dim=int(config["propos"]["projection_hidden_dim"]),
        predictor_hidden_dim=int(config["propos"]["predictor_hidden_dim"]),
        temperature=float(config["propos"]["temperature"]),
        sigma=float(config["propos"]["sigma"]),
        ema_momentum=float(config["propos"]["ema_momentum"]),
    )
    model = ClusteringBaseModel(
        backbone=backbone,
        adapter=adapter,
        head=head,
        normalize_cls=bool(config["gloca"].get("normalize_output", True)),
    )
    checkpoint_dir = output_dir / "checkpoints"
    resume_path = resolve_resume_checkpoint(
        config.get("trainer", {}).get("resume_from_checkpoint", "auto"),
        checkpoint_dir,
    )

    def log_checkpoint_metrics(epoch_logs: dict[str, Any]) -> None:
        epoch = int(epoch_logs["epoch"])
        append_csv_row(
            output_dir / "checkpoint_metrics.csv",
            {
                "experiment": config["experiment"]["name"],
                "head": "propos",
                "backbone": config["backbone"]["variant"],
                "dataset": spec.name,
                "seed": seed,
                "gloca": gloca_name,
                "n_clusters": int(config["head"]["n_clusters"]),
                "epoch": epoch,
                "epoch_1based": epoch + 1,
                "checkpoint": f"checkpoints/epoch_{epoch + 1:04d}.ckpt",
                "ari": float(epoch_logs.get("ari", float("nan"))),
                "nmi": float(epoch_logs.get("nmi", float("nan"))),
                "acc": float(epoch_logs.get("acc", float("nan"))),
                "loss_total_mean": float(epoch_logs.get("loss_total_mean", float("nan"))),
                "loss_psa_mean": float(epoch_logs.get("loss_psa_mean", float("nan"))),
                "loss_psl_mean": float(epoch_logs.get("loss_psl_mean", float("nan"))),
                "train_epoch_time_s": float(epoch_logs.get("train_epoch_time_s", 0.0)),
                "checkpoint_save_time_s": float(epoch_logs.get("checkpoint_save_time_s", 0.0)),
                "eval_time_s": float(epoch_logs.get("eval_time_s", 0.0)),
                "epoch_total_wall_time_s": float(epoch_logs.get("epoch_total_wall_time_s", 0.0)),
            },
        )

    propos_trainer = ProPosTrainer(
        model=model,
        datamodule=datamodule,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_path,
        checkpoint_metric_logger=log_checkpoint_metrics,
    )

    train_start = time.perf_counter()
    propos_trainer.fit()
    head_train_time_s = time.perf_counter() - train_start
    torch.save(propos_trainer.checkpoint_payload(), output_dir / "checkpoint.ckpt")

    infer_start = time.perf_counter()
    extracted = propos_trainer.extract_deterministic_features()
    final_kmeans = fit_kmeans(
        extracted["embeddings"],
        int(config["head"]["n_clusters"]),
        spherical=True,
        init=str(config["propos"]["kmeans_init"]),
        n_init=int(config["propos"]["kmeans_n_init"]),
        max_iter=int(config["propos"]["kmeans_max_iter"]),
        tol=float(config["propos"]["kmeans_tol"]),
        seed=seed,
        device=device,
    )
    inference_time_s = time.perf_counter() - infer_start
    assignments = final_kmeans["assignments"]

    metrics_extras = {
        "loss_psa_final": propos_trainer.loss_psa_final,
        "loss_psl_final": propos_trainer.loss_psl_final,
        "loss_total_final": propos_trainer.loss_total_final,
        "kmeans_interval": int(config["propos"]["kmeans_interval"]),
        "warmup_epochs": int(config["propos"]["warmup_epochs"]),
        "lambda_psl": float(config["propos"]["lambda_psl"]),
        "sigma": float(config["propos"]["sigma"]),
        "temperature": float(config["propos"]["temperature"]),
        "ema_momentum": float(config["propos"]["ema_momentum"]),
        "ema_momentum_final": float(propos_trainer.ema_momentum_final),
        "projection_dim": int(config["propos"]["projection_dim"]),
        "n_empty_cluster_batches": int(propos_trainer.n_empty_cluster_batches),
        "n_invalid_psl_batches": int(propos_trainer.n_invalid_psl_batches),
        "kmeans_backend": "torch",
        "kmeans_init": str(config["propos"]["kmeans_init"]),
        "_extra_fields": PROPOS_EXTRA_METRICS,
    }
    trainer_logs = propos_trainer.training_logs()
    logs_extras = {
        "uses_cached_backbone_features": False,
        "live_two_view_training": True,
        "deterministic_single_view_prediction": True,
        "backbone_requires_grad_false": not any(param.requires_grad for param in backbone.parameters()),
        **datamodule.data_loading_info(),
        "official_reference_files_inspected": [
            "temp/ProPos/models/propos/byol_wrapper.py",
            "temp/ProPos/models/propos/byol.py",
            "temp/ProPos/config/cifar20_r18_propos.yml",
            "temp/ProPos/utils/__init__.py",
        ],
        "adaptation_notes": [
            "DINOv2 remains frozen and is accessed through ClusteringBaseModel.encode_view for the online branch.",
            "The target branch keeps EMA copies of trainable GLoCA and the projection MLP; no gradients flow through it.",
            "This is a single-GPU GLoCA-compatible ProPos port. It preserves official PSA and PSL method logic but omits distributed gather, queue, shuffled BN, SyncBN, LARS, and large-batch official training infrastructure.",
            "PSL follows the official diagonal replacement and empty-cluster masking behavior.",
            "PSA follows the official -2 cosine alignment on noisy online projections.",
            "Warmup follows the official condition: PSL and latent noise are disabled while epoch <= warmup_epochs.",
        ],
        "head": "propos",
        "backbone": config["backbone"]["variant"],
        "gloca": gloca_name,
        "embedding_shape": list(extracted["embeddings"].shape),
        "attention_shape": None if extracted["attention"] is None else list(extracted["attention"].shape),
        **trainer_logs,
        "final_kmeans_backend": "torch",
        "final_kmeans_algorithm": final_kmeans["logs"]["algorithm"],
        "final_kmeans_logs": final_kmeans["logs"],
        "num_workers": int(config["trainer"]["num_workers"]),
        "head_train_time_s": head_train_time_s,
        "inference_time_s": inference_time_s,
        "checkpoint_metrics_path": "checkpoint_metrics.csv",
        **runtime_logs,
    }
    return finalize_run(
        output_dir=output_dir,
        config=config,
        spec=spec,
        head="propos",
        image_ids=extracted["image_ids"],
        labels=extracted["labels"],
        assignments=assignments,
        embeddings=extracted["embeddings"],
        attention=extracted["attention"],
        patch_grid=extracted["patch_grid"],
        seed=seed,
        backbone_cache_time_s=0.0,
        head_train_time_s=head_train_time_s,
        total_time_s=head_train_time_s + inference_time_s,
        inference_time_s=inference_time_s,
        uses_cached_backbone_features=False,
        metrics_extras=metrics_extras,
        logs_extras=logs_extras,
        metrics_fn=compute_clustering_metrics,
        cluster_diagnostics_fn=cluster_diagnostics,
        embedding_diagnostics_fn=embedding_diagnostics,
        attention_diagnostics_fn=attention_diagnostics,
        writer=write_outputs,
    )
