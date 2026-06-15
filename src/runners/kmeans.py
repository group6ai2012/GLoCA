from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import torch

from src.data import ClusteringDataModule
from src.experiments.config import save_experiment_config
from src.experiments.outputs import write_outputs
from src.features import DINOv2Backbone
from src.models import build_adapter
from src.models.clustering.kmeans import fit_kmeans
from src.runners.common import (
    apply_runtime_settings,
    assert_backbone_frozen,
    finalize_run,
    prepare_runner_context,
)
from src.runners.diagnostics import attention_diagnostics, cluster_diagnostics, embedding_diagnostics
from src.runners.embedding_export import extract_deterministic_embeddings
from src.evaluation import compute_clustering_metrics
from src.utils import ExperimentResult, resolve_gloca_name


def run_kmeans(config: dict[str, Any]) -> ExperimentResult:
    config = deepcopy(config)
    runtime_logs = apply_runtime_settings(config)
    ctx = prepare_runner_context(config, expected_head="kmeans")
    spec = ctx.spec
    output_dir = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, output_dir / "config.yaml")

    seed = ctx.seed
    if config["dataset"]["training_views"] == "auto":
        config["dataset"]["training_views"] = 1
    datamodule = ClusteringDataModule(
        spec=spec,
        image_size=int(config["backbone"]["image_size"]),
        batch_size=int(config["trainer"]["batch_size"]),
        num_workers=int(config["trainer"]["num_workers"]),
        training_mode="single_view",
        training_views=config["dataset"]["training_views"],
    )
    backbone = DINOv2Backbone(
        variant=config["backbone"]["variant"],
        image_size=int(config["backbone"]["image_size"]),
        freeze=bool(config["backbone"]["freeze"]),
    )
    assert_backbone_frozen(backbone)
    adapter = build_adapter(config, input_dim=backbone.output_dim)
    gloca_name = resolve_gloca_name(config)
    baseline = config["baseline"]
    spherical = bool(baseline["spherical"])
    effective_head = "spherical_kmeans" if spherical else "kmeans"
    normalize_cls = bool(config.get("gloca", {}).get("normalize_output", True))
    device = ctx.device
    total_start = time.perf_counter()

    extracted = extract_deterministic_embeddings(
        backbone=backbone,
        adapter=adapter,
        datamodule=datamodule,
        device=device,
        normalize_cls=normalize_cls,
    )
    embeddings = extracted["embeddings"]

    fit_start = time.perf_counter()
    kmeans_result = fit_kmeans(
        embeddings,
        int(config["head"]["n_clusters"]),
        spherical=spherical,
        init=str(baseline["kmeans_init"]),
        n_init=int(baseline["kmeans_n_init"]),
        max_iter=int(baseline["kmeans_max_iter"]),
        tol=float(baseline["kmeans_tol"]),
        seed=seed,
        device=device,
    )
    head_train_time_s = time.perf_counter() - fit_start
    total_time_s = time.perf_counter() - total_start
    assignments = kmeans_result["assignments"]
    clustered_embeddings = embeddings.float()

    logs_extras = {
        "non_trainable_baseline": True,
        "checkpoint_written": False,
        "uses_cached_backbone_features": True,
        "backbone_requires_grad_false": not any(param.requires_grad for param in backbone.parameters()),
        "deterministic_single_view_transform": True,
        "spherical": spherical,
        "cls_normalized": adapter is None and normalize_cls,
        "gloca": gloca_name,
        "head": effective_head,
        "configured_head": "kmeans",
        "backbone": config["backbone"]["variant"],
        "embedding_shape": list(clustered_embeddings.shape),
        "attention_shape": None if extracted["attention"] is None else list(extracted["attention"].shape),
        "num_workers": int(config["trainer"]["num_workers"]),
        "backbone_cache_time_s": extracted["cache_time_s"],
        "head_train_time_s": head_train_time_s,
        "total_time_s": total_time_s,
        "inference_time_s": 0.0,
        "kmeans_backend": "torch",
        "kmeans_algorithm": kmeans_result["logs"]["algorithm"],
        "kmeans_init": str(baseline["kmeans_init"]),
        "kmeans_n_init": int(baseline["kmeans_n_init"]),
        "kmeans_max_iter": int(baseline["kmeans_max_iter"]),
        "kmeans_tol": float(baseline["kmeans_tol"]),
        "kmeans_logs": kmeans_result["logs"],
        **runtime_logs,
    }
    return finalize_run(
        output_dir=output_dir,
        config=config,
        spec=spec,
        head=effective_head,
        image_ids=extracted["image_ids"],
        labels=extracted["labels"],
        assignments=assignments,
        embeddings=clustered_embeddings,
        attention=extracted["attention"],
        patch_grid=extracted["patch_grid"],
        seed=seed,
        backbone_cache_time_s=float(extracted["cache_time_s"]),
        head_train_time_s=head_train_time_s,
        total_time_s=total_time_s,
        inference_time_s=0.0,
        uses_cached_backbone_features=True,
        logs_extras=logs_extras,
        metrics_fn=compute_clustering_metrics,
        cluster_diagnostics_fn=cluster_diagnostics,
        embedding_diagnostics_fn=embedding_diagnostics,
        attention_diagnostics_fn=attention_diagnostics,
        writer=write_outputs,
    )
