from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from pathlib import Path

from src.data import FolderImageDataset, get_dataset_spec
from src.data.folder_dataset import DatasetSpec
from src.data.transforms import build_eval_transform
from src.utils import resolve_device, resolve_gloca_name, resolve_output_dir, seed_everything


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
