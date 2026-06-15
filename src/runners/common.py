from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from pathlib import Path

from src.data import FolderImageDataset, get_dataset_spec
from src.data.folder_dataset import DatasetSpec
from src.data.transforms import build_eval_transform
from src.evaluation import compute_clustering_metrics
from src.experiments.outputs import write_outputs
from src.runners.diagnostics import attention_diagnostics, cluster_diagnostics, embedding_diagnostics
from src.utils import ExperimentResult, resolve_device, resolve_gloca_name, resolve_output_dir, seed_everything


@dataclass
class RunnerContext:
    config: dict[str, Any]
    spec: DatasetSpec
    output_dir: Path
    seed: int
    device: torch.device
    n_clusters: int


def apply_runtime_settings(config: dict[str, Any]) -> dict[str, Any]:
    trainer_config = config.get("trainer") or {}
    if not isinstance(trainer_config, dict):
        raise ValueError("trainer must be a mapping when applying runtime settings.")
    matmul_precision = trainer_config.get("matmul_precision")
    if matmul_precision is not None:
        allowed = {"highest", "high", "medium"}
        if matmul_precision not in allowed:
            raise ValueError(
                "trainer.matmul_precision must be one of "
                f"{sorted(allowed)} or null, got {matmul_precision!r}"
            )
        torch.set_float32_matmul_precision(matmul_precision)
    return {
        "matmul_precision": matmul_precision,
        "runtime_settings_applied": matmul_precision is not None,
    }


def prepare_runner_context(config: dict[str, Any], expected_head: str | None = None) -> RunnerContext:
    if expected_head is not None and str(config["head"]["name"]).lower() != expected_head:
        raise ValueError(
            f"Runner expected head.name={expected_head!r}, got {str(config['head']['name']).lower()!r}"
        )
    spec = get_dataset_spec(config["dataset"])
    n_clusters = resolve_n_clusters(config, spec)
    config["head"]["n_clusters"] = n_clusters
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    return RunnerContext(
        config=config,
        spec=spec,
        output_dir=resolve_output_dir(config),
        seed=seed,
        device=resolve_device(config),
        n_clusters=n_clusters,
    )


def resolve_n_clusters(config: dict[str, Any], spec: DatasetSpec) -> int:
    raw = config["head"]["n_clusters"]
    if raw != "auto":
        return int(raw)
    eval_dataset = FolderImageDataset(
        spec,
        transform=build_eval_transform(int(config["backbone"]["image_size"])),
        training_views=1,
    )
    return int(eval_dataset.n_classes)


def assert_backbone_frozen(backbone: torch.nn.Module) -> None:
    if any(param.requires_grad for param in backbone.parameters()):
        raise RuntimeError("All DINOv2 parameters must have requires_grad=False")


def get_peak_gpu_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return 0.0


def build_assignment_payload(
    *,
    config: dict[str, Any],
    spec: DatasetSpec,
    head: str,
    image_ids: list[str],
    labels: torch.Tensor,
    assignments: torch.Tensor,
    patch_grid: tuple[int, int] | list[int],
) -> dict[str, Any]:
    return {
        "head": head,
        "backbone": config["backbone"]["variant"],
        "gloca": resolve_gloca_name(config),
        "dataset": spec.name,
        "seed": int(config["experiment"]["seed"]),
        "n_clusters": int(config["head"]["n_clusters"]),
        "image_ids": image_ids,
        "labels": labels.tolist(),
        "assignments": assignments.tolist(),
        "patch_grid": list(patch_grid),
    }


def finalize_run(
    *,
    output_dir: Path,
    config: dict[str, Any],
    spec: DatasetSpec,
    head: str,
    image_ids: list[str],
    labels: torch.Tensor,
    assignments: torch.Tensor,
    embeddings: torch.Tensor,
    attention: torch.Tensor | None,
    patch_grid: tuple[int, int] | list[int],
    seed: int,
    backbone_cache_time_s: float,
    head_train_time_s: float,
    total_time_s: float,
    inference_time_s: float,
    uses_cached_backbone_features: bool,
    metrics_extras: dict[str, Any] | None = None,
    logs_extras: dict[str, Any] | None = None,
    assignment_extras: dict[str, Any] | None = None,
    metrics_fn: Callable[[Any, Any, Any], dict[str, float]] = compute_clustering_metrics,
    cluster_diagnostics_fn: Callable[[Any, int], dict[str, Any]] = cluster_diagnostics,
    embedding_diagnostics_fn: Callable[[torch.Tensor], dict[str, Any]] = embedding_diagnostics,
    attention_diagnostics_fn: Callable[[torch.Tensor | None], dict[str, Any]] = attention_diagnostics,
    writer: Callable[..., None] = write_outputs,
) -> ExperimentResult:
    assignments_np = assignments.detach().cpu().numpy()
    embeddings_cpu = embeddings.detach().cpu().float()
    labels_cpu = labels.detach().cpu().long()
    n_clusters = int(config["head"]["n_clusters"])

    metrics = metrics_fn(labels_cpu.numpy(), assignments_np, embeddings_cpu.numpy())
    cluster_stats = cluster_diagnostics_fn(assignments_np, n_clusters)
    embedding_stats = embedding_diagnostics_fn(embeddings_cpu)
    attention_stats = attention_diagnostics_fn(attention)
    peak_gpu_mb = get_peak_gpu_mb()

    assignments_payload = build_assignment_payload(
        config=config,
        spec=spec,
        head=head,
        image_ids=image_ids,
        labels=labels_cpu,
        assignments=assignments.detach().cpu().long(),
        patch_grid=patch_grid,
    )
    if assignment_extras:
        assignments_payload.update(assignment_extras)

    metrics_row = {
        "experiment": config["experiment"]["name"],
        "head": head,
        "backbone": config["backbone"]["variant"],
        "dataset": spec.name,
        "seed": int(seed),
        "gloca": resolve_gloca_name(config),
        "n_clusters": n_clusters,
        "n_images": int(embeddings_cpu.shape[0]),
        "ari": metrics["ari"],
        "nmi": metrics["nmi"],
        "acc": metrics["acc"],
        "silhouette": metrics["silhouette"],
        **cluster_stats,
        **embedding_stats,
        **attention_stats,
        "backbone_cache_time_s": float(backbone_cache_time_s),
        "head_train_time_s": float(head_train_time_s),
        "total_time_s": float(total_time_s),
        "inference_time_s": float(inference_time_s),
        "peak_gpu_mb": peak_gpu_mb,
        "uses_cached_backbone_features": bool(uses_cached_backbone_features),
    }
    if metrics_extras:
        metrics_row.update(metrics_extras)

    logs = {
        "uses_cached_backbone_features": bool(uses_cached_backbone_features),
        "head": head,
        "backbone": config["backbone"]["variant"],
        "gloca": resolve_gloca_name(config),
        "embedding_shape": list(embeddings_cpu.shape),
        "attention_shape": None if attention is None else list(attention.shape),
        "num_workers": int(config["trainer"]["num_workers"]),
        "backbone_cache_time_s": float(backbone_cache_time_s),
        "head_train_time_s": float(head_train_time_s),
        "total_time_s": float(total_time_s),
        "inference_time_s": float(inference_time_s),
        **cluster_stats,
        **embedding_stats,
        **attention_stats,
    }
    if logs_extras:
        logs.update(logs_extras)

    writer(
        output_dir=output_dir,
        config=config,
        assignments_payload=assignments_payload,
        metrics_row=metrics_row,
        embeddings=embeddings_cpu,
        attention=attention,
        logs=logs,
    )
    return ExperimentResult(output_dir=str(output_dir), metrics=metrics_row)
