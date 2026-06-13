from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import torch

from src.data import ClusteringDataModule
from src.evaluation import compute_clustering_metrics
from src.experiments.config import save_experiment_config
from src.experiments.outputs import write_outputs
from src.features import DINOv2Backbone
from src.models import build_adapter
from src.models.clustering.kmeans import fit_kmeans
from src.runners.common import (
    apply_runtime_settings,
    assert_backbone_frozen,
    build_assignment_payload,
    get_peak_gpu_mb,
    prepare_runner_context,
)
from src.runners.diagnostics import attention_diagnostics, cluster_diagnostics, embedding_diagnostics
from src.runners.embedding_export import extract_deterministic_embeddings
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
    assignments_np = assignments.numpy()
    clustered_embeddings = embeddings.float()

    metrics = compute_clustering_metrics(
        extracted["labels"].numpy(),
        assignments_np,
        clustered_embeddings.numpy(),
    )
    cluster_stats = cluster_diagnostics(assignments_np, int(config["head"]["n_clusters"]))
    embedding_stats = embedding_diagnostics(clustered_embeddings)
    attention_stats = attention_diagnostics(extracted["attention"])
    peak_gpu_mb = get_peak_gpu_mb()

    assignments_payload = build_assignment_payload(
        config=config,
        spec=spec,
        head=effective_head,
        image_ids=extracted["image_ids"],
        labels=extracted["labels"],
        assignments=assignments,
        patch_grid=extracted["patch_grid"],
    )
    metrics_row = {
        "experiment": config["experiment"]["name"],
        "head": effective_head,
        "backbone": config["backbone"]["variant"],
        "dataset": spec.name,
        "seed": seed,
        "gloca": gloca_name,
        "n_clusters": int(config["head"]["n_clusters"]),
        "n_images": int(clustered_embeddings.shape[0]),
        "ari": metrics["ari"],
        "nmi": metrics["nmi"],
        "acc": metrics["acc"],
        "silhouette": metrics["silhouette"],
        **cluster_stats,
        **embedding_stats,
        **attention_stats,
        "backbone_cache_time_s": extracted["cache_time_s"],
        "head_train_time_s": head_train_time_s,
        "total_time_s": total_time_s,
        "inference_time_s": 0.0,
        "peak_gpu_mb": peak_gpu_mb,
        "uses_cached_backbone_features": True,
    }
    logs = {
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
        **cluster_stats,
        **embedding_stats,
        **attention_stats,
    }
    write_outputs(
        output_dir=output_dir,
        config=config,
        assignments_payload=assignments_payload,
        metrics_row=metrics_row,
        embeddings=clustered_embeddings,
        attention=extracted["attention"],
        logs=logs,
    )
    return ExperimentResult(output_dir=str(output_dir), metrics=metrics_row)
