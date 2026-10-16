from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.experiments import baselines as sweep_script
from src.experiments.config import (
    load_experiment_config,
    validate_global_local_kmeans_config,
    validate_global_local_sweep_config,
)
from src.models.baselines.global_local import FrozenCLSMeanPatchConcat
from src.runners import global_local as global_local_runner


ROOT = Path(__file__).resolve().parents[1]


def _config(output_dir: Path) -> dict[str, Any]:
    return {
        "experiment": {"name": "global_local_test", "seed": 7, "output_dir": str(output_dir)},
        "dataset": {"name": "tiny", "training_views": "auto"},
        "backbone": {
            "family": "dinov2",
            "variant": "facebook/dinov2-small",
            "freeze": True,
            "image_size": 224,
        },
        "gloca": {
            "enabled": False,
            "name": "disabled",
            "embedding_dim": 384,
            "normalize_output": True,
        },
        "head": {"name": "kmeans", "n_clusters": 2, "training_mode": "single_view"},
        "baseline": {
            "embedding_source": "cls_mean_patch_concat",
            "normalize_embedding": True,
            "spherical": False,
            "kmeans_init": "kmeans++",
            "kmeans_n_init": 2,
            "kmeans_max_iter": 25,
            "kmeans_tol": 1.0e-4,
        },
        "prediction": {
            "deterministic": True,
            "export_embeddings": True,
            "export_attention": False,
        },
        "trainer": {
            "max_epochs": 1,
            "batch_size": 4,
            "lr": 1.0e-4,
            "accelerator": "cpu",
            "precision": 32,
            "matmul_precision": "high",
            "num_workers": 0,
        },
    }


def test_frozen_global_local_feature_matches_normalized_concat() -> None:
    cls = torch.tensor([[3.0, 4.0], [1.0, -1.0]])
    patches = torch.tensor(
        [
            [[1.0, 3.0], [3.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    )
    module = FrozenCLSMeanPatchConcat(normalize_output=True)

    output = module(cls=cls, patch_tokens=patches, patch_grid=(1, 2))
    expected = F.normalize(torch.cat((cls, patches.mean(dim=1)), dim=-1), dim=-1)

    assert torch.allclose(output["embedding"], expected)
    assert output["embedding"].shape == (2, 4)
    assert torch.allclose(output["embedding"].norm(dim=-1), torch.ones(2))
    assert output["attention"] is None
    assert output["patch_grid"] == (1, 2)
    assert sum(parameter.numel() for parameter in module.parameters()) == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["gloca"].update(enabled=True), "gloca.enabled=false"),
        (lambda config: config["backbone"].update(freeze=False), "backbone.freeze=true"),
        (lambda config: config["baseline"].update(spherical=True), "spherical=false"),
        (
            lambda config: config["baseline"].update(embedding_source="cls"),
            "cls_mean_patch_concat",
        ),
        (
            lambda config: config["baseline"].update(normalize_embedding=False),
            "normalize_embedding=true",
        ),
    ],
)
def test_global_local_config_rejects_semantic_drift(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    config = _config(tmp_path)
    mutate(config)
    with pytest.raises(ValueError, match=message):
        validate_global_local_kmeans_config(config)


def test_full_sweep_expands_two_datasets_and_three_seeds() -> None:
    path = ROOT / "configs/baselines/global_local_full.yaml"
    config = load_experiment_config(path)
    validate_global_local_sweep_config(config)

    plans = sweep_script._build_plans(config)

    assert len(plans) == 2
    assert {plan["dataset"] for plan in plans} == {"plantvillage", "plantseg"}
    assert {plan["seed"] for plan in plans} == {42}
    assert {
        Path(plan["config"]["experiment"]["output_dir"]) for plan in plans
    } == {Path("outputs/global_local_baseline")}
    assert all(plan["expected"]["embedding_source"] == "cls_mean_patch_concat" for plan in plans)


def test_global_local_runner_writes_canonical_outputs_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    output_dir = tmp_path / "global_local_test" / "kmeans" / "tiny" / "seed_7"
    spec = SimpleNamespace(name="tiny")
    backbone = nn.Linear(1, 1)
    backbone.requires_grad_(False)
    feature_builder = FrozenCLSMeanPatchConcat(normalize_output=True)
    cache = {
        "embeddings": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.9, 0.1],
            ]
        ),
        "labels": torch.tensor([0, 0, 1, 1]),
        "image_ids": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "patch_grid": (16, 16),
        "cache_time_s": 0.25,
        "runtime_logs": {"matmul_precision": "high", "runtime_settings_applied": True},
        "spec": spec,
        "backbone": backbone,
        "feature_builder": feature_builder,
    }
    captured: dict[str, Any] = {}

    def fake_fit(embeddings: torch.Tensor, n_clusters: int, **kwargs: Any) -> dict[str, Any]:
        captured.update({"embeddings": embeddings, "n_clusters": n_clusters, **kwargs})
        return {
            "assignments": torch.tensor([0, 0, 1, 1]),
            "logs": {"algorithm": "kmeans"},
        }

    monkeypatch.setattr(
        global_local_runner,
        "prepare_runner_context",
        lambda *_args, **_kwargs: SimpleNamespace(
            spec=spec,
            output_dir=output_dir,
            seed=7,
            device=torch.device("cpu"),
            n_clusters=2,
        ),
    )
    monkeypatch.setattr(global_local_runner, "fit_kmeans", fake_fit)
    monkeypatch.setattr(
        global_local_runner,
        "compute_clustering_metrics",
        lambda *_args: {"ari": 1.0, "nmi": 1.0, "acc": 1.0, "silhouette": 0.5},
    )
    monkeypatch.setattr(
        global_local_runner,
        "cluster_diagnostics",
        lambda *_args: {
            "n_nonempty_clusters": 2,
            "cluster_size_min": 2,
            "cluster_size_max": 2,
            "cluster_size_entropy": 1.0,
        },
    )
    monkeypatch.setattr(
        global_local_runner,
        "embedding_diagnostics",
        lambda *_args: {
            "embedding_variance_mean": 0.2,
            "embedding_norm_mean": 1.0,
            "embedding_norm_std": 0.0,
        },
    )
    monkeypatch.setattr(
        global_local_runner,
        "attention_diagnostics",
        lambda *_args: {
            "attention_entropy": None,
            "attention_max": None,
            "attention_top5_mass": None,
            "attention_variance": None,
        },
    )

    result = global_local_runner.run_cached_global_local_kmeans.__wrapped__(config, cache)

    assert result["output_dir"] == str(output_dir)
    assert captured["spherical"] is False
    assert captured["n_clusters"] == 2
    assert torch.equal(captured["embeddings"], cache["embeddings"])
    assert {path.name for path in output_dir.iterdir()} == {
        "assignments.json",
        "config.yaml",
        "embeddings.pt",
        "logs.json",
        "metrics.csv",
    }
    assignments = json.loads((output_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["head"] == "kmeans"
    assert assignments["gloca"] == "disabled"
    assert assignments["embedding_source"] == "cls_mean_patch_concat"
    with (output_dir / "metrics.csv").open("r", encoding="utf-8") as handle:
        metrics = next(csv.DictReader(handle))
    assert metrics["embedding_source"] == "cls_mean_patch_concat"
    assert metrics["embedding_dim"] == "4"
    assert metrics["trainable_parameter_count"] == "0"
    logs = json.loads((output_dir / "logs.json").read_text(encoding="utf-8"))
    assert logs["learned_adapter"] is False
    assert logs["final_embedding_normalized"] is True
