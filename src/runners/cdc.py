from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import torch

from src.data import ClusteringDataModule
from src.evaluation import compute_clustering_metrics
from src.evaluation.clustering_metrics import calibration_error_metrics
from src.experiments.config import save_experiment_config, validate_cdc_config
from src.experiments.outputs import write_outputs
from src.features import DINOv2Backbone
from src.models import ClusteringBaseModel, build_adapter
from src.models.clustering import CDCHead
from src.runners.common import (
    apply_runtime_settings,
    assert_backbone_frozen,
    finalize_run,
    prepare_runner_context,
)
from src.runners.diagnostics import attention_diagnostics, cluster_diagnostics, embedding_diagnostics
from src.training.checkpointing import resolve_resume_checkpoint
from src.training.cdc_trainer import CDCTrainer
from src.utils import ExperimentResult, resolve_gloca_name


CDC_EXTRA_METRICS = [
    "clustering_confidence_mean",
    "clustering_confidence_std",
    "calibrated_confidence_mean",
    "calibrated_confidence_std",
    "reliable_sample_ratio",
    "calibration_threshold",
    "calibration_ece",
    "calibration_mce",
    "pseudo_label_count",
    "pseudo_label_entropy",
    "cdc_pretrain_epochs",
    "cdc_refine_epochs",
    "cdc_init_mode",
]


def run_cdc(config: dict[str, Any]) -> ExperimentResult:
    config = deepcopy(config)
    runtime_logs = apply_runtime_settings(config)
    validate_cdc_config(config)
    if config["head"]["name"] != "cdc":
        raise ValueError(f"run_cdc only supports head.name='cdc', got {config['head']['name']!r}")

    ctx = prepare_runner_context(config, expected_head="cdc")
    spec = ctx.spec
    config["dataset"]["training_views"] = str(config["cdc"]["training_views"])
    config["head"]["training_mode"] = "cdc"

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
        training_mode="cdc",
        training_views=str(config["cdc"]["training_views"]),
        cdc_augment_config=config["cdc"].get("augment", {}),
        pin_memory=config["trainer"].get("pin_memory"),
        persistent_workers=config["trainer"].get("persistent_workers"),
        prefetch_factor=config["trainer"].get("prefetch_factor"),
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
    head = CDCHead(
        input_dim=embedding_dim,
        n_clusters=int(config["head"]["n_clusters"]),
        hidden_dim=int(config["cdc"]["hidden_dim"]),
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
    trainer = CDCTrainer(
        model=model,
        datamodule=datamodule,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_path,
    )

    train_start = time.perf_counter()
    trainer.fit()
    head_train_time_s = time.perf_counter() - train_start
    torch.save(trainer.checkpoint_payload(), output_dir / "checkpoint.ckpt")

    infer_start = time.perf_counter()
    extracted = trainer.extract_deterministic_features()
    inference_time_s = time.perf_counter() - infer_start
    assignments = extracted["assignments"]
    labels_np = extracted["labels"].numpy()

    calibration = calibration_error_metrics(
        labels_np,
        assignments.numpy(),
        extracted["calibrated_confidence"].numpy(),
    )
    latest_epoch = trainer.epoch_history[-1] if trainer.epoch_history else {}

    metrics_extras = {
        "clustering_confidence_mean": float(extracted["confidence"].mean().item()),
        "clustering_confidence_std": float(extracted["confidence"].std(unbiased=False).item()),
        "calibrated_confidence_mean": float(extracted["calibrated_confidence"].mean().item()),
        "calibrated_confidence_std": float(extracted["calibrated_confidence"].std(unbiased=False).item()),
        "reliable_sample_ratio": float(latest_epoch.get("reliable_sample_ratio", 0.0)),
        "calibration_threshold": "",
        "calibration_ece": calibration["calibration_ece"],
        "calibration_mce": calibration["calibration_mce"],
        "pseudo_label_count": int(latest_epoch.get("selected_pseudo_label_count", 0)),
        "pseudo_label_entropy": float(latest_epoch.get("pseudo_label_entropy", 0.0)),
        "cdc_pretrain_epochs": int(config["cdc"].get("pretrain_epochs", 0)),
        "cdc_refine_epochs": int(config["trainer"]["max_epochs"]),
        "cdc_init_mode": str(trainer.init_logs.get("cdc_init_mode", "random")),
        "_extra_fields": CDC_EXTRA_METRICS,
    }
    trainer_logs = trainer.training_logs()
    logs_extras = {
        "uses_cached_backbone_features": False,
        "live_image_training": True,
        "cdc_named_views": True,
        "deterministic_single_view_prediction": True,
        "final_prediction_head": "calibration",
        "backbone_requires_grad_false": not any(param.requires_grad for param in backbone.parameters()),
        **datamodule.data_loading_info(),
        "official_reference_files_inspected": [
            "temp/CDC/cdc/backbones/models.py",
            "temp/CDC/cdc/methods/calibrate_train.py",
            "temp/CDC/cdc/args.py",
            "temp/CDC/cdc/data/augment.py",
            "temp/CDC/cdc/data/custom_dataset.py",
        ],
        "adaptation_notes": [
            "CDC consumes only final embeddings from ClusteringBaseModel.encode_view; raw DINO patch tokens are not read by CDC.",
            "DINOv2 remains frozen; optional GLoCA parameters are the only trainable shared-path parameters.",
            "The original CDC standard/augment/val sample contract is ported as weak/strong/calibration views.",
            "Reliable sample selection and calibration mini-cluster targets are computed once per virtual CDC meta-batch; optimizer updates then run in sub-batch chunks.",
            "Strong views are generated once by the physical dataloader and stored on CPU inside the virtual CDC meta-batch.",
            "The calibration loss uses detached calibration embeddings and detached clustering-head targets, so it updates only the calibration head.",
            "Prototype initialization uses local torch K-Means when enough deterministic embeddings exist; otherwise logs cdc_init_mode=random with the fallback reason.",
            "The original 2000-epoch orth_train step is exposed as a config flag but is not run in this first local port.",
        ],
        "head": "cdc",
        "backbone": config["backbone"]["variant"],
        "gloca": gloca_name,
        "embedding_shape": list(extracted["embeddings"].shape),
        "attention_shape": None if extracted["attention"] is None else list(extracted["attention"].shape),
        **trainer_logs,
        **calibration,
        "num_workers": int(config["trainer"]["num_workers"]),
        "head_train_time_s": head_train_time_s,
        "inference_time_s": inference_time_s,
        **runtime_logs,
    }
    result = finalize_run(
        output_dir=output_dir,
        config=config,
        spec=spec,
        head="cdc",
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
    torch.save(extracted["confidence"].cpu(), output_dir / "confidence.pt")
    torch.save(extracted["calibrated_confidence"].cpu(), output_dir / "calibrated_confidence.pt")
    torch.save(extracted["pseudo_labels"].cpu(), output_dir / "pseudo_labels.pt")
    return result
