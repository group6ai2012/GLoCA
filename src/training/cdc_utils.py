from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def select_reliable_samples(
    calibration_probabilities: torch.Tensor,
    per_class_selected_num: int | None = None,
) -> dict[str, torch.Tensor | int | float | list[int]]:
    if calibration_probabilities.ndim != 2:
        raise ValueError(
            f"Expected probabilities [B, C], got {tuple(calibration_probabilities.shape)}"
        )
    batch_size, n_clusters = calibration_probabilities.shape
    top_count = (
        int(per_class_selected_num)
        if per_class_selected_num is not None and int(per_class_selected_num) > 0
        else max(1, int(math.floor(batch_size / max(1, n_clusters))))
    )
    top_count = min(top_count, batch_size)
    confidences, pseudo_labels = calibration_probabilities.max(dim=1)
    selected = torch.zeros(
        batch_size, dtype=torch.bool, device=calibration_probabilities.device
    )
    classwise_counts: list[int] = []
    for cluster_id in range(n_clusters):
        sorted_indices = torch.argsort(
            calibration_probabilities[:, cluster_id], descending=True
        )[:top_count]
        selected_count = int(
            torch.floor(confidences[sorted_indices].mean() * top_count)
            .detach()
            .cpu()
            .item()
        )
        selected_count = max(0, min(top_count, selected_count))
        if selected_count > 0:
            selected[sorted_indices[:selected_count]] = True
        classwise_counts.append(selected_count)
    selected_count = int(selected.sum().detach().cpu().item())
    return {
        "selected_mask": selected,
        "pseudo_labels": pseudo_labels,
        "selected_count": selected_count,
        "reliable_sample_ratio": float(selected_count / max(1, batch_size)),
        "per_class_selected_num": int(top_count),
        "classwise_selected_counts": classwise_counts,
    }


def calibration_target_loss(
    calibration_logits: torch.Tensor,
    target_distribution: torch.Tensor,
    *,
    entropy_weight: float,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(calibration_logits, dim=1)
    target = target_distribution.detach().clamp_min(0.0)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)
    loss_cal = -(target * log_probs).sum(dim=1).mean()
    mean_probs = F.softmax(calibration_logits, dim=1).mean(dim=0).clamp_min(eps)
    loss_entropy = torch.sum(mean_probs * torch.log(mean_probs))
    return loss_cal + float(entropy_weight) * loss_entropy, loss_cal, loss_entropy


def pseudo_label_entropy(labels: torch.Tensor, n_clusters: int) -> float:
    if labels.numel() == 0:
        return 0.0
    counts = torch.bincount(
        labels.detach().long().cpu(), minlength=int(n_clusters)
    ).float()
    probs = counts / counts.sum().clamp_min(1.0)
    valid = probs > 0
    return float(-(probs[valid] * probs[valid].log()).sum().item())


def confidence_stats(values: torch.Tensor, prefix: str) -> dict[str, float]:
    vals = values.detach().float().cpu()
    if vals.numel() == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(vals.mean().item()),
        f"{prefix}_std": float(vals.std(unbiased=False).item()),
        f"{prefix}_min": float(vals.min().item()),
        f"{prefix}_max": float(vals.max().item()),
    }


def split_sub_batch_indices(
    batch_size: int, sub_batch_size: int, device: torch.device | None = None
) -> list[torch.Tensor]:
    batch_size = int(batch_size)
    sub_batch_size = int(sub_batch_size)
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}")
    if sub_batch_size <= 0:
        raise ValueError(f"sub_batch_size must be positive, got {sub_batch_size}")
    indices = torch.arange(batch_size, device=device)
    return list(indices.split(sub_batch_size))
