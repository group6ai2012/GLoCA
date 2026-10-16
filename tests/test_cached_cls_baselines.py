from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.baselines.cached_cls import CachedCLSMLPClassifier, remap_hdbscan_labels
from src.training.cached_cls_trainer import build_cached_cls_splits, train_cached_cls_mlp


def test_cached_mlp_is_plain_and_capacity_is_stable() -> None:
    model = CachedCLSMLPClassifier(384, (896, 384), 38)
    assert [layer.__class__.__name__ for layer in model.feature_layers] == ["Linear", "GELU", "Linear", "GELU"]
    assert not hasattr(model, "alpha")
    assert sum(parameter.numel() for parameter in model.parameters()) == 704038
    output = model(torch.randn(3, 384))
    assert output["embedding"].shape == (3, 384)
    assert output["logits"].shape == (3, 38)


def test_hdbscan_noise_is_remapped_to_nonnegative_cluster() -> None:
    assignments, logs = remap_hdbscan_labels(torch.tensor([4, -1, 4, 8, -1]))
    assert assignments.tolist() == [0, 2, 0, 1, 2]
    assert logs["n_discovered_clusters"] == 2
    assert logs["n_output_clusters"] == 3
    assert logs["noise_count"] == 2


def test_cached_cls_splits_are_deterministic_disjoint_and_class_complete() -> None:
    labels = torch.arange(5).repeat_interleave(7)
    first = build_cached_cls_splits(labels, seed=42)
    second = build_cached_cls_splits(labels, seed=42)
    assert first.as_dict() == second.as_dict()
    assert set(first.train.tolist()).isdisjoint(first.validation.tolist())
    assert set(first.train.tolist()).isdisjoint(first.test.tolist())
    assert set(first.validation.tolist()).isdisjoint(first.test.tolist())
    for split in [first.train, first.validation, first.test]:
        assert torch.unique(labels[split]).numel() == 5


def test_cached_mlp_training_writes_best_checkpoint_and_test_only_outputs(tmp_path: Path) -> None:
    torch.manual_seed(0)
    features = torch.randn(24, 8)
    labels = torch.arange(3).repeat_interleave(8)
    result = train_cached_cls_mlp(
        features,
        labels,
        n_classes=3,
        hidden_dims=(10, 6),
        seed=3,
        max_epochs=2,
        batch_size=6,
        learning_rate=1.0e-2,
        weight_decay=1.0e-4,
        output_dir=tmp_path,
        checkpoint_interval=1,
        resume_from_checkpoint=None,
    )
    assert (tmp_path / "checkpoint.ckpt").exists()
    assert (tmp_path / "checkpoints" / "latest.ckpt").exists()
    assert result["labels"].shape[0] == result["splits"].test.shape[0]
    assert result["embeddings"].shape[1] == 6
    assert torch.isfinite(result["embeddings"]).all()


def test_cached_cls_split_rejects_too_small_class() -> None:
    with pytest.raises(ValueError, match="at least three samples"):
        build_cached_cls_splits(torch.tensor([0, 0, 1]), seed=0)
