from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


SUPPORTED_KMEANS_INITS = {"random", "kmeans++"}


class TorchKMeans(nn.Module):
    """Torch K-Means, optionally using spherical/cosine updates."""

    def __init__(
        self,
        n_clusters: int,
        *,
        spherical: bool = False,
        init: str = "kmeans++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1.0e-4,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.spherical = bool(spherical)
        self.init = str(init)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.device = device

    @torch.no_grad()
    def fit_predict(self, features: torch.Tensor) -> dict[str, Any]:
        return fit_kmeans(
            features,
            self.n_clusters,
            spherical=self.spherical,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            seed=self.seed,
            device=self.device,
        )


class TorchSphericalKMeans(TorchKMeans):
    """Cosine/spherical K-Means."""

    def __init__(
        self,
        n_clusters: int,
        *,
        init: str = "kmeans++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1.0e-4,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            n_clusters,
            spherical=True,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            device=device,
        )


@torch.no_grad()
def fit_kmeans(
    features: torch.Tensor,
    n_clusters: int,
    *,
    spherical: bool,
    init: str,
    n_init: int,
    max_iter: int,
    tol: float,
    seed: int,
    device: torch.device | None = None,
) -> dict[str, Any]:
    _validate_inputs(features, n_clusters=n_clusters, init=init)
    run_device = device or features.device
    clean_features = torch.nan_to_num(features.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).to(run_device)
    if spherical:
        labels, centers, logs = _run_kmeans(
            F.normalize(clean_features, dim=-1),
            n_clusters=int(n_clusters),
            spherical=True,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
    else:
        labels, centers, logs = _run_kmeans(
            clean_features,
            n_clusters=int(n_clusters),
            spherical=False,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )

    labels_cpu = labels.detach().cpu().long()
    centers_cpu = torch.nan_to_num(centers.detach().cpu().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(centers_cpu).all():
        raise RuntimeError("K-Means produced non-finite centers.")
    if labels_cpu.shape != (int(features.shape[0]),):
        raise RuntimeError(f"K-Means produced labels with invalid shape {tuple(labels_cpu.shape)}.")
    if labels_cpu.numel() and (int(labels_cpu.min()) < 0 or int(labels_cpu.max()) >= int(n_clusters)):
        raise RuntimeError("K-Means produced assignments outside the cluster range.")
    logs["device"] = str(run_device)
    return {"assignments": labels_cpu, "centers": centers_cpu, "logs": logs}


@torch.no_grad()
def torch_kmeans(
    features: torch.Tensor,
    n_clusters: int,
    *,
    init: str = "kmeans++",
    n_init: int = 10,
    max_iter: int = 300,
    seed: int = 0,
    tol: float = 1.0e-4,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    result = fit_kmeans(
        features,
        n_clusters,
        spherical=False,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        device=device,
    )
    return result["assignments"], result["centers"], result["logs"]


@torch.no_grad()
def torch_spherical_kmeans(
    features: torch.Tensor,
    n_clusters: int,
    *,
    init: str = "kmeans++",
    n_init: int = 10,
    max_iter: int = 300,
    seed: int = 0,
    tol: float = 1.0e-4,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    result = fit_kmeans(
        features,
        n_clusters,
        spherical=True,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        device=device,
    )
    return result["assignments"], result["centers"], result["logs"]


def _run_kmeans(
    features: torch.Tensor,
    *,
    n_clusters: int,
    spherical: bool,
    init: str,
    n_init: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    n_samples = int(features.shape[0])
    generator = _make_generator(features.device, int(seed))
    best_labels: torch.Tensor | None = None
    best_centers: torch.Tensor | None = None
    best_objective = -float("inf") if spherical else float("inf")
    best_n_iter = 0
    empty_reinitializations = 0

    for init_idx in range(max(1, int(n_init))):
        centers = _initialize_centers(
            features,
            n_clusters=n_clusters,
            init=init,
            spherical=spherical,
            generator=generator,
        )
        labels = torch.zeros(n_samples, dtype=torch.long, device=features.device)
        previous_labels: torch.Tensor | None = None
        previous_objective: float | None = None
        n_iter_run = 0

        for iteration in range(max(1, int(max_iter))):
            labels, objective = _assign(features, centers, spherical=spherical)
            n_iter_run = iteration + 1
            centers, n_empty = _update_centers(
                features,
                labels,
                n_clusters=n_clusters,
                spherical=spherical,
                generator=generator,
            )
            empty_reinitializations += n_empty

            labels_unchanged = previous_labels is not None and bool(torch.equal(labels, previous_labels))
            objective_delta = float("inf") if previous_objective is None else abs(objective - previous_objective)
            if labels_unchanged or objective_delta <= float(tol):
                break
            previous_labels = labels.detach().clone()
            previous_objective = objective

        final_labels, final_objective = _assign(features, centers, spherical=spherical)
        better = final_objective > best_objective if spherical else final_objective < best_objective
        if better or init_idx == 0:
            best_objective = final_objective
            best_labels = final_labels.detach().clone()
            best_centers = centers.detach().clone()
            best_n_iter = n_iter_run

    if best_labels is None or best_centers is None:
        raise RuntimeError("Torch K-Means failed to produce assignments.")
    if spherical:
        best_centers = F.normalize(best_centers, dim=-1)
    logs = {
        "backend": "torch",
        "algorithm": "spherical_kmeans" if spherical else "kmeans",
        "init": init,
        "n_init": int(max(1, int(n_init))),
        "max_iter": int(max(1, int(max_iter))),
        "tol": float(tol),
        "best_objective": float(best_objective),
        "best_n_iter": int(best_n_iter),
        "empty_reinitializations": int(empty_reinitializations),
    }
    return best_labels, best_centers, logs


def _assign(features: torch.Tensor, centers: torch.Tensor, *, spherical: bool) -> tuple[torch.Tensor, float]:
    if spherical:
        scores = features @ centers.T
        max_scores, labels = scores.max(dim=1)
        return labels, float(max_scores.mean().detach().cpu())
    distances = torch.cdist(features, centers, p=2).pow(2)
    min_distances, labels = distances.min(dim=1)
    return labels, float(min_distances.mean().detach().cpu())


def _initialize_centers(
    features: torch.Tensor,
    *,
    n_clusters: int,
    init: str,
    spherical: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    if init == "random":
        indices = torch.randperm(features.shape[0], generator=generator, device=features.device)[:n_clusters]
        centers = features[indices].clone()
        return F.normalize(centers, dim=-1) if spherical else centers
    return _initialize_kmeans_plus_plus(
        features,
        n_clusters=n_clusters,
        spherical=spherical,
        generator=generator,
    )


def _initialize_kmeans_plus_plus(
    features: torch.Tensor,
    *,
    n_clusters: int,
    spherical: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    n_samples = int(features.shape[0])
    selected = torch.zeros(n_samples, dtype=torch.bool, device=features.device)
    first = torch.randint(n_samples, (1,), generator=generator, device=features.device).item()
    selected[first] = True
    centers = [features[first].clone()]

    for _ in range(1, n_clusters):
        current_centers = torch.stack(centers, dim=0)
        if spherical:
            max_similarity = (features @ current_centers.T).max(dim=1).values
            distances = torch.clamp(1.0 - max_similarity, min=0.0)
        else:
            distances = torch.cdist(features, current_centers, p=2).pow(2).min(dim=1).values
        distances = torch.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)
        distances[selected] = 0.0
        distance_sum = distances.sum()
        if (not bool(torch.isfinite(distance_sum))) or float(distance_sum.detach().cpu()) <= 0.0:
            next_index = _random_unselected_index(selected, generator=generator)
        else:
            next_index = torch.multinomial(distances / distance_sum, 1, generator=generator).item()
        selected[next_index] = True
        centers.append(features[next_index].clone())

    stacked = torch.stack(centers, dim=0)
    return F.normalize(stacked, dim=-1) if spherical else stacked


def _update_centers(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_clusters: int,
    spherical: bool,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    centers = []
    n_empty = 0
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if bool(mask.any()):
            center = features[mask].mean(dim=0)
        else:
            n_empty += 1
            replacement = torch.randint(features.shape[0], (1,), generator=generator, device=features.device).item()
            center = features[replacement]
        centers.append(center)
    stacked = torch.nan_to_num(torch.stack(centers, dim=0), nan=0.0, posinf=0.0, neginf=0.0)
    return (F.normalize(stacked, dim=-1) if spherical else stacked), n_empty


def _random_unselected_index(selected: torch.Tensor, *, generator: torch.Generator) -> int:
    candidates = torch.nonzero(~selected, as_tuple=False).flatten()
    if candidates.numel() == 0:
        candidates = torch.arange(selected.shape[0], device=selected.device)
    offset = torch.randint(candidates.numel(), (1,), generator=generator, device=selected.device).item()
    return int(candidates[offset].item())


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type not in {"mps"} else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def _validate_inputs(features: torch.Tensor, *, n_clusters: int, init: str) -> None:
    if features.ndim != 2:
        raise ValueError(f"features must be a 2D tensor, got shape {tuple(features.shape)}")
    if init not in SUPPORTED_KMEANS_INITS:
        raise ValueError(f"Unsupported K-Means init {init!r}; expected one of {sorted(SUPPORTED_KMEANS_INITS)}")
    n_samples = int(features.shape[0])
    n_clusters = int(n_clusters)
    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")
    if n_samples < n_clusters:
        raise ValueError(f"n_clusters={n_clusters} cannot exceed n_samples={n_samples}")
