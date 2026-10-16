from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.baselines.cached_cls import CachedCLSMLPClassifier
from src.experiments.reproducibility import checkpoint_identity, validate_checkpoint
from src.training.checkpointing import capture_rng_state, restore_rng_state, atomic_torch_save, copy_as_latest, resolve_resume_checkpoint, should_save_epoch_checkpoint


@dataclass(frozen=True)
class CachedCLSSplits:
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": self.train.tolist(),
            "validation": self.validation.tolist(),
            "test": self.test.tolist(),
        }


def build_cached_cls_splits(
    labels: torch.Tensor,
    *,
    seed: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> CachedCLSSplits:
    labels = labels.detach().cpu().long()
    ratios = np.asarray([train_ratio, validation_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios <= 0) or not np.isclose(float(ratios.sum()), 1.0):
        raise ValueError("train/validation/test ratios must be positive and sum to 1.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    train_parts: list[torch.Tensor] = []
    validation_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    for class_id in torch.unique(labels, sorted=True).tolist():
        class_indices = torch.nonzero(labels == int(class_id), as_tuple=False).flatten()
        if class_indices.numel() < 3:
            raise ValueError(
                f"Class {int(class_id)} has only {int(class_indices.numel())} samples; "
                "cached MLP requires at least three samples per class."
            )
        shuffled = class_indices[torch.randperm(class_indices.numel(), generator=generator)]
        n_validation = max(1, int(round(float(class_indices.numel()) * validation_ratio)))
        n_test = max(1, int(round(float(class_indices.numel()) * test_ratio)))
        while n_validation + n_test >= int(class_indices.numel()):
            if n_validation >= n_test and n_validation > 1:
                n_validation -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                raise ValueError(f"Unable to allocate a training sample for class {int(class_id)}.")
        validation_parts.append(shuffled[:n_validation])
        test_parts.append(shuffled[n_validation : n_validation + n_test])
        train_parts.append(shuffled[n_validation + n_test :])

    return CachedCLSSplits(
        train=torch.cat(train_parts).sort().values,
        validation=torch.cat(validation_parts).sort().values,
        test=torch.cat(test_parts).sort().values,
    )


def train_cached_cls_mlp(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_classes: int,
    hidden_dims: tuple[int, int] | list[int],
    seed: int,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    output_dir: Path,
    checkpoint_interval: int,
    resume_from_checkpoint: Any = None,
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    config: dict | None = None,
) -> dict[str, Any]:
    config = config or {"experiment": {"seed": seed}, "trainer": {"max_epochs": max_epochs}, "hidden_dims": list(hidden_dims), "batch_size": batch_size, "lr": learning_rate, "weight_decay": weight_decay, "split_ratios": list(split_ratios)}
    features = features.detach().cpu().float().contiguous()
    labels = labels.detach().cpu().long().contiguous()
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("Cached MLP features and labels have incompatible shapes.")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Cached MLP features contain non-finite values.")

    splits = build_cached_cls_splits(labels, seed=seed, train_ratio=split_ratios[0], validation_ratio=split_ratios[1], test_ratio=split_ratios[2])
    model = CachedCLSMLPClassifier(features.shape[1], hidden_dims, n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    train_labels = labels[splits.train]
    counts = torch.bincount(train_labels, minlength=int(n_classes)).float()
    if bool((counts <= 0).any()):
        raise ValueError("Every class must be represented in the cached MLP training split.")
    class_weights = counts.sum() / (float(n_classes) * counts)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    plain_criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(features[splits.train], labels[splits.train]),
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = resolve_resume_checkpoint(resume_from_checkpoint, checkpoint_dir)
    start_epoch = 0
    best_epoch = 0
    best_val_accuracy = -float("inf")
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    if resume_path is not None and resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_checkpoint(payload, config)
        if payload.get("splits") != splits.as_dict():
            raise ValueError("Checkpoint MLP splits differ from current splits")
        restore_rng_state(payload.get("rng_state"))
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload.get("epoch", 0))
        best_epoch = int(payload.get("best_epoch", 0))
        best_val_accuracy = float(payload.get("best_val_accuracy", best_val_accuracy))
        best_val_loss = float(payload.get("best_val_loss", best_val_loss))
        best_state = {key: value.detach().clone() for key, value in payload.get("best_state", model.state_dict()).items()}
        history = list(payload.get("history", []))

    for epoch in range(start_epoch, int(max_epochs)):
        loader.generator.manual_seed((int(seed) + epoch) % 2**32)
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features)["logits"], batch_labels)
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = _evaluate(model, features[splits.validation], labels[splits.validation], plain_criterion)
        history.append({"epoch": float(epoch + 1), "validation_loss": val_loss, "validation_accuracy": val_accuracy})
        improved = val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and val_loss < best_val_loss)
        if improved:
            best_epoch = epoch + 1
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        payload = {
            "reproducibility": checkpoint_identity(config),
            "rng_state": capture_rng_state(),
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_state": best_state or model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "history": history,
            "splits": splits.as_dict(),
            "class_weights": class_weights,
        }
        if should_save_epoch_checkpoint(epoch, checkpoint_interval):
            path = checkpoint_dir / f"epoch_{epoch + 1:04d}.ckpt"
            atomic_torch_save(payload, path)
            copy_as_latest(path, checkpoint_dir / "latest.ckpt")

    if best_state is None:
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        best_epoch = int(max_epochs)
    model.load_state_dict(best_state)
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "history": history,
            "splits": splits.as_dict(),
            "class_weights": class_weights,
            "input_dim": int(features.shape[1]),
            "hidden_dims": list(hidden_dims),
            "n_classes": int(n_classes),
        },
        output_dir / "checkpoint.ckpt",
    )
    test_loss, _ = _evaluate(model, features[splits.test], labels[splits.test], plain_criterion)
    with torch.no_grad():
        test_output = model(features[splits.test])
        test_embedding = test_output["embedding"].detach().cpu()
        test_logits = test_output["logits"].detach().cpu()
        assignments = test_logits.argmax(dim=1).long()
    test_labels = labels[splits.test]
    direct_accuracy = float((assignments == test_labels).float().mean().item())
    return {
        "assignments": assignments,
        "labels": test_labels,
        "embeddings": test_embedding,
        "splits": splits,
        "test_loss": float(test_loss),
        "classification_accuracy": direct_accuracy,
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_loss": float(best_val_loss),
        "class_weights": class_weights,
        "history": history,
        "model": model,
    }


def _evaluate(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features)["logits"]
        loss = float(criterion(logits, labels).item())
        accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())
    return loss, accuracy
