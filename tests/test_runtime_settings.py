from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from src.runners import common
from src.runners import kmeans as kmeans_runner
from src.runners import propos as propos_runner


def test_apply_runtime_settings_accepts_supported_values(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(common.torch, "set_float32_matmul_precision", calls.append)

    for value in ["highest", "high", "medium"]:
        logs = common.apply_runtime_settings({"trainer": {"matmul_precision": value}})
        assert logs["matmul_precision"] == value
        assert logs["runtime_settings_applied"] is True
        assert logs["deterministic"] is True
        assert logs["persistent_workers"] is False

    assert calls == ["highest", "high", "medium"]


def test_apply_runtime_settings_accepts_null_and_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(common.torch, "set_float32_matmul_precision", calls.append)

    for config in [{"trainer": {"matmul_precision": None}}, {"trainer": {}}]:
        logs = common.apply_runtime_settings(config)
        assert logs["matmul_precision"] == "highest"
        assert logs["runtime_settings_applied"] is True
        assert logs["deterministic"] is True

    assert calls == ["highest", "highest"]


def test_apply_runtime_settings_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="trainer\\.matmul_precision"):
        common.apply_runtime_settings({"trainer": {"matmul_precision": "fast"}})


def test_run_propos_writes_matmul_precision_to_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "propos_out"
    _patch_propos_runner(monkeypatch, output_dir)

    result = propos_runner.run_propos.__wrapped__(_propos_config(tmp_path))
    logs = json.loads((Path(result.output_dir) / "logs.json").read_text(encoding="utf-8"))

    assert logs["matmul_precision"] == "high"
    assert logs["runtime_settings_applied"] is True


def test_run_kmeans_writes_matmul_precision_to_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "kmeans_out"
    _patch_kmeans_runner(monkeypatch, output_dir)

    result = kmeans_runner.run_kmeans.__wrapped__(_kmeans_config(tmp_path))
    logs = json.loads((Path(result.output_dir) / "logs.json").read_text(encoding="utf-8"))

    assert logs["matmul_precision"] == "high"
    assert logs["runtime_settings_applied"] is True


def _write_logs(output_dir: Path, logs: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs.json").write_text(json.dumps(logs), encoding="utf-8")


class _DummyBackbone(nn.Module):
    output_dim = 4

    def __init__(self, **_: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, image: torch.Tensor) -> dict[str, Any]:
        return {"cls": torch.zeros(image.shape[0], 4), "patch_tokens": torch.empty(image.shape[0], 0, 4), "patch_grid": (1, 1)}


class _DummyDataModule:
    def __init__(self, **_: Any) -> None:
        pass

    def data_loading_info(self) -> dict[str, Any]:
        return {}


class _DummyProPosHead(nn.Module):
    def __init__(self, **_: Any) -> None:
        super().__init__()


class _DummyProPosTrainer:
    loss_psa_final = 0.1
    loss_psl_final = 0.2
    loss_total_final = 0.3
    ema_momentum_final = 0.996
    n_empty_cluster_batches = 0
    n_invalid_psl_batches = 0

    def __init__(self, **_: Any) -> None:
        pass

    def fit(self) -> None:
        pass

    def checkpoint_payload(self) -> dict[str, Any]:
        return {"state_dict": {}}

    def extract_deterministic_features(self) -> dict[str, Any]:
        return _extracted()

    def training_logs(self) -> dict[str, Any]:
        return {"epoch_history": []}


def _patch_propos_runner(monkeypatch: pytest.MonkeyPatch, output_dir: Path) -> None:
    ctx = SimpleNamespace(spec=SimpleNamespace(name="tiny"), output_dir=output_dir, seed=7, device=torch.device("cpu"), n_clusters=2)
    monkeypatch.setattr(propos_runner, "prepare_runner_context", lambda config, expected_head=None: ctx)
    monkeypatch.setattr(propos_runner, "ClusteringDataModule", _DummyDataModule)
    monkeypatch.setattr(propos_runner, "DINOv2Backbone", _DummyBackbone)
    monkeypatch.setattr(propos_runner, "build_adapter", lambda config, input_dim: None)
    monkeypatch.setattr(propos_runner, "ProPosHead", _DummyProPosHead)
    monkeypatch.setattr(propos_runner, "ProPosTrainer", _DummyProPosTrainer)
    monkeypatch.setattr(propos_runner, "fit_kmeans", _fit_kmeans)
    monkeypatch.setattr(propos_runner, "compute_clustering_metrics", _metrics)
    monkeypatch.setattr(propos_runner, "cluster_diagnostics", _cluster_stats)
    monkeypatch.setattr(propos_runner, "embedding_diagnostics", _embedding_stats)
    monkeypatch.setattr(propos_runner, "attention_diagnostics", _attention_stats)
    monkeypatch.setattr(propos_runner, "write_outputs", lambda output_dir, logs, **kwargs: _write_logs(output_dir, logs))


def _patch_kmeans_runner(monkeypatch: pytest.MonkeyPatch, output_dir: Path) -> None:
    ctx = SimpleNamespace(spec=SimpleNamespace(name="tiny"), output_dir=output_dir, seed=7, device=torch.device("cpu"), n_clusters=2)
    monkeypatch.setattr(kmeans_runner, "prepare_runner_context", lambda config, expected_head=None: ctx)
    monkeypatch.setattr(kmeans_runner, "ClusteringDataModule", _DummyDataModule)
    monkeypatch.setattr(kmeans_runner, "DINOv2Backbone", _DummyBackbone)
    monkeypatch.setattr(kmeans_runner, "build_adapter", lambda config, input_dim: None)
    monkeypatch.setattr(kmeans_runner, "extract_deterministic_embeddings", lambda **kwargs: _extracted(cache_time_s=0.01))
    monkeypatch.setattr(kmeans_runner, "fit_kmeans", _fit_kmeans)
    monkeypatch.setattr(kmeans_runner, "compute_clustering_metrics", _metrics)
    monkeypatch.setattr(kmeans_runner, "cluster_diagnostics", _cluster_stats)
    monkeypatch.setattr(kmeans_runner, "embedding_diagnostics", _embedding_stats)
    monkeypatch.setattr(kmeans_runner, "attention_diagnostics", _attention_stats)
    monkeypatch.setattr(kmeans_runner, "write_outputs", lambda output_dir, logs, **kwargs: _write_logs(output_dir, logs))


def _extracted(cache_time_s: float = 0.0) -> dict[str, Any]:
    return {
        "embeddings": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        "labels": torch.tensor([0, 1]),
        "image_ids": ["a.jpg", "b.jpg"],
        "attention": None,
        "patch_grid": (1, 1),
        "cache_time_s": cache_time_s,
    }


def _fit_kmeans(*_: Any, **__: Any) -> dict[str, Any]:
    return {"assignments": torch.tensor([0, 1]), "logs": {"algorithm": "spherical_kmeans"}}


def _metrics(*_: Any, **__: Any) -> dict[str, float]:
    return {"ari": 1.0, "nmi": 1.0, "acc": 1.0, "silhouette": 0.0}


def _cluster_stats(*_: Any, **__: Any) -> dict[str, float]:
    return {
        "n_nonempty_clusters": 2,
        "cluster_size_min": 1,
        "cluster_size_max": 1,
        "cluster_size_entropy": 1.0,
    }


def _embedding_stats(*_: Any, **__: Any) -> dict[str, float]:
    return {
        "embedding_variance_mean": 0.25,
        "embedding_norm_mean": 1.0,
        "embedding_norm_std": 0.0,
    }


def _attention_stats(*_: Any, **__: Any) -> dict[str, Any]:
    return {
        "attention_entropy": None,
        "attention_max": None,
        "attention_top5_mass": None,
        "attention_variance": None,
    }


def _common_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "experiment": {"name": "runtime_test", "seed": 7, "output_dir": str(tmp_path)},
        "dataset": {"name": "tiny", "training_views": "auto"},
        "backbone": {"family": "dinov2", "variant": "facebook/dinov2-small", "freeze": True, "image_size": 224},
        "gloca": {"enabled": False, "name": "disabled", "embedding_dim": 4, "normalize_output": True},
        "prediction": {"deterministic": True, "export_embeddings": True, "export_attention": True},
        "trainer": {
            "max_epochs": 1,
            "batch_size": 2,
            "lr": 1.0e-3,
            "accelerator": "cpu",
            "precision": 32,
            "matmul_precision": "high",
            "num_workers": 0,
        },
    }


def _propos_config(tmp_path: Path) -> dict[str, Any]:
    config = _common_config(tmp_path)
    config.update(
        {
            "head": {"name": "propos", "n_clusters": 2, "training_mode": "contrastive_two_view"},
            "propos": {
                "embedding_dim": 4,
                "projection_dim": 2,
                "projection_hidden_dim": 4,
                "predictor_hidden_dim": 4,
                "temperature": 0.5,
                "sigma": 0.001,
                "lambda_psl": 0.1,
                "ema_momentum": 0.996,
                "ema_momentum_max": 1.0,
                "ema_momentum_increase": True,
                "kmeans_interval": 1,
                "kmeans_init": "kmeans++",
                "kmeans_n_init": 1,
                "kmeans_max_iter": 2,
                "kmeans_tol": 1.0e-4,
                "warmup_epochs": 0,
                "predictor_lr_multiplier": 1.0,
                "weight_decay": 0.0,
                "optimizer": "adamw",
                "symmetric_loss": True,
                "profile_batches": False,
                "gloca_lr_multiplier": 1.0,
                "gloca_alpha_lr_multiplier": 1.0,
                "freeze_gloca": False,
                "freeze_gloca_epochs": 0,
                "log_gloca_diagnostics": False,
            },
        }
    )
    return config


def _kmeans_config(tmp_path: Path) -> dict[str, Any]:
    config = _common_config(tmp_path)
    config.update(
        {
            "head": {"name": "kmeans", "n_clusters": 2, "training_mode": "single_view"},
            "baseline": {
                "spherical": False,
                "kmeans_init": "kmeans++",
                "kmeans_n_init": 1,
                "kmeans_max_iter": 2,
                "kmeans_tol": 1.0e-4,
            },
        }
    )
    return config
