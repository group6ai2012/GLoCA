from __future__ import annotations

from src.experiments.reproducibility import record_run

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data import ClusteringDataModule
from src.evaluation import compute_clustering_metrics
from src.evaluation.assignment_schema import CACHED_CLS_METRIC_FIELDS
from src.experiments.outputs import write_outputs
from src.features import DINOv2Backbone
from src.models.baselines.cached_cls import fit_gaussian_mixture, fit_hdbscan
from src.runners.common import (
    apply_runtime_settings,
    assert_backbone_frozen,
    finalize_run,
    prepare_runner_context,
)
from src.runners.diagnostics import attention_diagnostics, cluster_diagnostics, embedding_diagnostics
from src.runners.embedding_export import extract_deterministic_embeddings
from src.training.cached_cls_trainer import train_cached_cls_mlp
from src.utils import ExperimentResult, resolve_gloca_name


def extract_cached_cls_embeddings(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    runtime_logs = apply_runtime_settings(config)
    if bool(config.get("gloca", {}).get("enabled", False)):
        raise ValueError("Cached CLS baselines require gloca.enabled=false.")
    spec_config = config["dataset"]
    from src.data import get_dataset_spec

    spec = get_dataset_spec(spec_config)
    datamodule = ClusteringDataModule(
        spec=spec,
        image_size=int(config["backbone"]["image_size"]),
        batch_size=int(config["trainer"]["batch_size"]),
        num_workers=int(config["trainer"]["num_workers"]),
        seed=int(config["experiment"]["seed"]),
        deterministic=bool(config["trainer"].get("deterministic", True)),
        training_mode="single_view",
        training_views=1,
    )
    backbone = DINOv2Backbone(
        variant=config["backbone"]["variant"],
        image_size=int(config["backbone"]["image_size"]),
        freeze=bool(config["backbone"]["freeze"]),
        revision=config["backbone"].get("revision"),
    )
    assert_backbone_frozen(backbone)
    extracted = extract_deterministic_embeddings(
        backbone=backbone,
        adapter=None,
        datamodule=datamodule,
        device=torch.device("cuda" if torch.cuda.is_available() and config["trainer"].get("accelerator", "auto") in {"auto", "gpu", "cuda"} else "cpu"),
        normalize_cls=bool(config.get("gloca", {}).get("normalize_output", True)),
    )
    extracted["runtime_logs"] = runtime_logs
    extracted["spec"] = spec
    extracted["backbone"] = backbone
    from src.experiments.reproducibility import feature_cache_key
    extracted["feature_key"] = feature_cache_key(config, "cls")
    return extracted


@record_run
def run_cached_cls_gmm(config: dict[str, Any], cache: dict[str, Any]) -> ExperimentResult:
    config = deepcopy(config)
    runtime_logs = cache.get("runtime_logs", {})
    ctx = prepare_runner_context(config, expected_head="gmm")
    output_dir = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    result = fit_gaussian_mixture(
        cache["embeddings"],
        n_components=ctx.n_clusters,
        covariance_type=str(config["cached_baseline"]["gmm_covariance_type"]),
        init_params=str(config["cached_baseline"]["gmm_init_params"]),
        n_init=int(config["cached_baseline"]["gmm_n_init"]),
        max_iter=int(config["cached_baseline"]["gmm_max_iter"]),
        tol=float(config["cached_baseline"]["gmm_tol"]),
        reg_covar=float(config["cached_baseline"]["gmm_reg_covar"]),
        seed=ctx.seed,
    )
    head_time = time.perf_counter() - start
    logs = {
        "non_trainable_baseline": True,
        "backbone_requires_grad_false": True,
        "cached_cls_shared": True,
        "gloca": "disabled",
        "head": "gmm",
        "backbone": config["backbone"]["variant"],
        "uses_cached_backbone_features": True,
        "gmm": result["logs"],
        "probability_mean": float(result["probabilities"].mean().item()),
        "backbone_cache_time_s": float(cache["cache_time_s"]),
        "head_train_time_s": float(head_time),
        "total_time_s": float(time.perf_counter() - start + cache["cache_time_s"]),
        **runtime_logs,
    }
    return _finalize_cached_run(config, ctx, cache, result["assignments"], cache["embeddings"], None, head_time, logs, {})


@record_run
def run_cached_cls_hdbscan(config: dict[str, Any], cache: dict[str, Any]) -> ExperimentResult:
    config = deepcopy(config)
    runtime_logs = cache.get("runtime_logs", {})
    config["head"]["n_clusters"] = "auto"
    ctx = prepare_runner_context(config, expected_head="hdbscan")
    output_dir = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    baseline = config["cached_baseline"]
    result = fit_hdbscan(
        cache["embeddings"],
        min_cluster_size=int(baseline["hdbscan_min_cluster_size"]),
        min_samples=None if baseline.get("hdbscan_min_samples") in {None, "none", "null"} else int(baseline["hdbscan_min_samples"]),
        cluster_selection_epsilon=float(baseline["hdbscan_cluster_selection_epsilon"]),
        metric=str(baseline["hdbscan_metric"]),
        algorithm=str(baseline["hdbscan_algorithm"]),
        leaf_size=int(baseline["hdbscan_leaf_size"]),
        n_jobs=None if baseline.get("hdbscan_n_jobs") is None else int(baseline["hdbscan_n_jobs"]),
        cluster_selection_method=str(baseline["hdbscan_cluster_selection_method"]),
        allow_single_cluster=bool(baseline["hdbscan_allow_single_cluster"]),
    )
    config["head"]["n_clusters"] = int(result["logs"]["n_output_clusters"])
    ctx.n_clusters = int(result["logs"]["n_output_clusters"])
    head_time = time.perf_counter() - start
    logs = {
        "non_trainable_baseline": True,
        "backbone_requires_grad_false": True,
        "cached_cls_shared": True,
        "gloca": "disabled",
        "head": "hdbscan",
        "backbone": config["backbone"]["variant"],
        "uses_cached_backbone_features": True,
        "hdbscan": result["logs"],
        "mean_membership_probability": float(result["probabilities"].mean()),
        "reference_n_classes": int(cache["labels"].unique().numel()),
        "backbone_cache_time_s": float(cache["cache_time_s"]),
        "head_train_time_s": float(head_time),
        "total_time_s": float(time.perf_counter() - start + cache["cache_time_s"]),
        **runtime_logs,
    }
    metrics_extras = {
        "reference_n_classes": int(cache["labels"].unique().numel()),
        "n_discovered_clusters": int(result["logs"]["n_discovered_clusters"]),
        "noise_count": int(result["logs"]["noise_count"]),
        "noise_fraction": float(result["logs"]["noise_fraction"]),
        "mean_membership_probability": float(result["probabilities"].mean()),
        "_extra_fields": CACHED_CLS_METRIC_FIELDS,
    }
    return _finalize_cached_run(config, ctx, cache, result["assignments"], cache["embeddings"], None, head_time, logs, metrics_extras)


@record_run
def run_cached_cls_mlp(config: dict[str, Any], cache: dict[str, Any]) -> ExperimentResult:
    config = deepcopy(config)
    runtime_logs = cache.get("runtime_logs", {})
    ctx = prepare_runner_context(config, expected_head="supervised_mlp")
    output_dir = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    baseline = config["cached_baseline"]
    trained = train_cached_cls_mlp(
        cache["embeddings"],
        cache["labels"],
        n_classes=ctx.n_clusters,
        hidden_dims=tuple(int(value) for value in baseline["mlp_hidden_dims"]),
        seed=ctx.seed,
        max_epochs=int(config["trainer"]["max_epochs"]),
        batch_size=int(config["trainer"]["batch_size"]),
        learning_rate=float(config["trainer"]["lr"]),
        weight_decay=float(baseline["mlp_weight_decay"]),
        output_dir=output_dir,
        checkpoint_interval=int(config["trainer"].get("checkpoint_interval", 0)),
        resume_from_checkpoint=config["trainer"].get("resume_from_checkpoint"),
        split_ratios=tuple(float(value) for value in baseline["mlp_split_ratios"]),
        config=config,
    )
    test_labels = trained["labels"]
    test_predictions = trained["assignments"]
    from sklearn.metrics import balanced_accuracy_score, f1_score

    direct_accuracy = float((test_predictions == test_labels).float().mean().item())
    metrics_extras = {
        "classification_accuracy": direct_accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(test_labels.numpy(), test_predictions.numpy())),
        "macro_f1": float(f1_score(test_labels.numpy(), test_predictions.numpy(), average="macro", zero_division=0)),
        "test_loss": float(trained["test_loss"]),
        "best_epoch": int(trained["best_epoch"]),
        "best_val_accuracy": float(trained["best_val_accuracy"]),
        "best_val_loss": float(trained["best_val_loss"]),
        "train_n_images": int(trained["splits"].train.numel()),
        "validation_n_images": int(trained["splits"].validation.numel()),
        "test_n_images": int(trained["splits"].test.numel()),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in trained["model"].parameters())),
        "evaluation_split": "test",
        "_extra_fields": CACHED_CLS_METRIC_FIELDS,
    }
    logs = {
        "trainable_baseline": True,
        "backbone_requires_grad_false": True,
        "cached_cls_shared": True,
        "gloca": "disabled",
        "head": "supervised_mlp",
        "backbone": config["backbone"]["variant"],
        "uses_cached_backbone_features": True,
        "evaluation_split": "test",
        "split_indices": trained["splits"].as_dict(),
        "class_weights": trained["class_weights"].tolist(),
        "history": trained["history"],
        "mlp_hidden_dims": [int(value) for value in baseline["mlp_hidden_dims"]],
        "trainable_parameter_count": metrics_extras["trainable_parameter_count"],
        "backbone_cache_time_s": float(cache["cache_time_s"]),
        "head_train_time_s": float(time.perf_counter() - start),
        "total_time_s": float(time.perf_counter() - start + cache["cache_time_s"]),
        **runtime_logs,
    }
    test_indices = trained["splits"].test.tolist()
    test_ids = [cache["image_ids"][index] for index in test_indices]
    return _finalize_cached_run(config, ctx, cache, test_predictions, trained["embeddings"], None, float(time.perf_counter() - start), logs, metrics_extras, labels=test_labels, image_ids=test_ids)


def _finalize_cached_run(
    config: dict[str, Any],
    ctx: Any,
    cache: dict[str, Any],
    assignments: torch.Tensor,
    embeddings: torch.Tensor,
    attention: torch.Tensor | None,
    head_time: float,
    logs: dict[str, Any],
    metrics_extras: dict[str, Any],
    *,
    labels: torch.Tensor | None = None,
    image_ids: list[str] | None = None,
) -> ExperimentResult:
    config["gloca"]["enabled"] = False
    config["gloca"]["name"] = "disabled"
    labels = cache["labels"] if labels is None else labels
    image_ids = cache["image_ids"] if image_ids is None else image_ids
    return finalize_run(
        output_dir=ctx.output_dir,
        config=config,
        spec=cache["spec"],
        head=str(config["head"]["name"]),
        image_ids=image_ids,
        labels=labels,
        assignments=assignments,
        embeddings=embeddings,
        attention=attention,
        patch_grid=cache["patch_grid"],
        seed=ctx.seed,
        backbone_cache_time_s=float(cache["cache_time_s"]),
        head_train_time_s=float(head_time),
        total_time_s=float(logs.get("total_time_s", head_time + cache["cache_time_s"])),
        inference_time_s=0.0,
        uses_cached_backbone_features=True,
        metrics_extras=metrics_extras,
        logs_extras=logs,
        metrics_fn=compute_clustering_metrics,
        cluster_diagnostics_fn=cluster_diagnostics,
        embedding_diagnostics_fn=embedding_diagnostics,
        attention_diagnostics_fn=attention_diagnostics,
        writer=write_outputs,
    )
