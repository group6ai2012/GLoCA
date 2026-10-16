from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


MIN_SKLEARN_HDBSCAN_VERSION = (1, 3)


class CachedCLSMLPClassifier(nn.Module):
    """Plain MLP classifier over cached DINO CLS embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] | list[int],
        n_classes: int,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(int(value) for value in hidden_dims)
        if len(hidden_dims) != 2:
            raise ValueError(f"hidden_dims must contain exactly two values, got {hidden_dims}")
        if min(int(input_dim), *hidden_dims, int(n_classes)) <= 0:
            raise ValueError("MLP dimensions must all be positive.")
        self.input_dim = int(input_dim)
        self.hidden_dims = hidden_dims
        self.n_classes = int(n_classes)
        self.feature_layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dims[1], self.n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_layers(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encode(x)
        return {"embedding": embedding, "logits": self.classifier(embedding)}


def fit_gaussian_mixture(
    features: torch.Tensor,
    *,
    n_components: int,
    covariance_type: str,
    init_params: str,
    n_init: int,
    max_iter: int,
    tol: float,
    reg_covar: float,
    seed: int,
) -> dict[str, Any]:
    _validate_features(features)
    from sklearn.mixture import GaussianMixture

    estimator = GaussianMixture(
        n_components=int(n_components),
        covariance_type=str(covariance_type),
        init_params=str(init_params),
        n_init=int(n_init),
        max_iter=int(max_iter),
        tol=float(tol),
        reg_covar=float(reg_covar),
        random_state=int(seed),
    )
    array = features.detach().cpu().float().numpy()
    assignments = estimator.fit_predict(array)
    return {
        "assignments": torch.from_numpy(assignments.astype(np.int64, copy=False)),
        "probabilities": torch.from_numpy(estimator.predict_proba(array).astype(np.float32, copy=False)),
        "logs": {
            "backend": "sklearn",
            "algorithm": "gaussian_mixture",
            "converged": bool(estimator.converged_),
            "n_iter": int(estimator.n_iter_),
            "lower_bound": float(estimator.lower_bound_),
            "aic": float(estimator.aic(array)),
            "bic": float(estimator.bic(array)),
        },
    }


def fit_hdbscan(
    features: torch.Tensor,
    *,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_epsilon: float,
    metric: str,
    algorithm: str,
    leaf_size: int,
    n_jobs: int | None,
    cluster_selection_method: str,
    allow_single_cluster: bool,
) -> dict[str, Any]:
    _validate_features(features)
    _require_sklearn_hdbscan()
    from sklearn.cluster import HDBSCAN

    estimator = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        metric=str(metric),
        algorithm=str(algorithm),
        leaf_size=int(leaf_size),
        n_jobs=None if n_jobs is None else int(n_jobs),
        cluster_selection_method=str(cluster_selection_method),
        allow_single_cluster=bool(allow_single_cluster),
    )
    raw = np.asarray(estimator.fit_predict(features.detach().cpu().float().numpy()), dtype=np.int64)
    assignments, relabel_logs = remap_hdbscan_labels(raw)
    probabilities = np.asarray(estimator.probabilities_, dtype=np.float32)
    return {
        "assignments": torch.from_numpy(assignments),
        "raw_assignments": torch.from_numpy(raw),
        "probabilities": torch.from_numpy(probabilities),
        "logs": {
            "backend": "sklearn",
            "algorithm": "hdbscan",
            **relabel_logs,
        },
    }


def remap_hdbscan_labels(raw_labels: np.ndarray | torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_labels, dtype=np.int64)
    if raw.ndim != 1:
        raise ValueError(f"HDBSCAN labels must be one-dimensional, got shape {raw.shape}")
    non_noise = sorted(int(value) for value in np.unique(raw) if int(value) >= 0)
    mapping = {old: new for new, old in enumerate(non_noise)}
    remapped = np.empty(raw.shape, dtype=np.int64)
    for old, new in mapping.items():
        remapped[raw == old] = new
    noise_mask = raw < 0
    if bool(noise_mask.any()):
        remapped[noise_mask] = len(non_noise)
    n_output_clusters = len(non_noise) + int(bool(noise_mask.any()))
    return remapped, {
        "n_discovered_clusters": len(non_noise),
        "n_output_clusters": n_output_clusters,
        "noise_count": int(noise_mask.sum()),
        "noise_fraction": float(noise_mask.mean()) if raw.size else 0.0,
        "noise_assignment_id": len(non_noise) if bool(noise_mask.any()) else None,
    }


def _validate_features(features: torch.Tensor) -> None:
    if features.ndim != 2:
        raise ValueError(f"Cached CLS features must be 2D, got shape {tuple(features.shape)}")
    if features.shape[0] == 0:
        raise ValueError("Cached CLS features must not be empty.")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Cached CLS features contain non-finite values.")


def _require_sklearn_hdbscan() -> None:
    try:
        import sklearn
        from sklearn.cluster import HDBSCAN as _HDBSCAN  # noqa: F401
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "HDBSCAN requires scikit-learn>=1.3; install a compatible scikit-learn version."
        ) from exc
    version = _numeric_version_tuple(sklearn.__version__)
    if version < MIN_SKLEARN_HDBSCAN_VERSION:
        raise RuntimeError(
            "HDBSCAN requires scikit-learn>=1.3; "
            f"found scikit-learn {sklearn.__version__}."
        )


def _numeric_version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for item in str(value).split("."):
        digits = "".join(character for character in item if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)
