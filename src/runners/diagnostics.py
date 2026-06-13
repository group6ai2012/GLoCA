from __future__ import annotations

import math

import numpy as np
import torch


def cluster_diagnostics(assignments: np.ndarray | torch.Tensor, n_clusters: int) -> dict[str, float | int]:
    assignments_np = np.asarray(assignments, dtype=np.int64)
    counts = np.bincount(assignments_np, minlength=int(n_clusters))
    nonempty = counts[counts > 0]
    proportions = nonempty / max(1, assignments_np.shape[0])
    entropy = 0.0
    if proportions.size > 0 and n_clusters > 1:
        entropy = float(-(proportions * np.log(proportions + 1e-12)).sum() / math.log(n_clusters))
    return {
        "n_nonempty_clusters": int(nonempty.size),
        "cluster_size_min": int(nonempty.min()) if nonempty.size else 0,
        "cluster_size_max": int(nonempty.max()) if nonempty.size else 0,
        "cluster_size_entropy": entropy,
    }


def embedding_diagnostics(embeddings: torch.Tensor) -> dict[str, float]:
    embeddings = embeddings.detach().float().cpu()
    norms = embeddings.norm(dim=1)
    return {
        "embedding_variance_mean": float(embeddings.var(dim=0, unbiased=False).mean().item()),
        "embedding_norm_mean": float(norms.mean().item()),
        "embedding_norm_std": float(norms.std(unbiased=False).item()),
    }


def attention_diagnostics(attention: torch.Tensor | None) -> dict[str, float]:
    if attention is None:
        return {}
    attention = attention.detach().float().cpu()
    safe_attention = attention.clamp_min(1e-8)
    topk = min(5, attention.shape[1])
    return {
        "attention_entropy": float((-(attention * safe_attention.log()).sum(dim=1)).mean().item()),
        "attention_max": float(attention.max(dim=1).values.mean().item()),
        "attention_top5_mass": float(attention.topk(topk, dim=1).values.sum(dim=1).mean().item()),
        "attention_variance": float(attention.var(dim=1, unbiased=False).mean().item()),
    }
