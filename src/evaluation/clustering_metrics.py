from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def clustering_accuracy(labels: np.ndarray, assignments: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    assignments = assignments.astype(np.int64)
    matrix, row_ind, col_ind = _hungarian_alignment(labels, assignments)
    return float(matrix[row_ind, col_ind].sum() / max(1, labels.shape[0]))


def compute_clustering_metrics(labels, assignments, embeddings) -> dict[str, float]:
    labels_np = np.asarray(labels)
    assignments_np = np.asarray(assignments)
    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    metrics = {
        "ari": float(adjusted_rand_score(labels_np, assignments_np)),
        "nmi": float(normalized_mutual_info_score(labels_np, assignments_np)),
        "acc": clustering_accuracy(labels_np, assignments_np),
        "silhouette": float("nan"),
    }
    unique_assignments = np.unique(assignments_np)
    if 1 < unique_assignments.shape[0] < embeddings_np.shape[0]:
        sample_size = min(2000, embeddings_np.shape[0])
        metrics["silhouette"] = float(
            silhouette_score(
                embeddings_np,
                assignments_np,
                sample_size=sample_size,
                random_state=0,
            )
        )
    return metrics


def calibration_error_metrics(
    labels: np.ndarray,
    assignments: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    assignments = np.asarray(assignments, dtype=np.int64)
    confidences = np.asarray(confidences, dtype=np.float64)
    if labels.size == 0:
        return {"calibration_ece": float("nan"), "calibration_mce": float("nan")}

    _, row_ind, col_ind = _hungarian_alignment(labels, assignments)
    mapping = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    aligned = np.asarray(
        [mapping.get(int(pred), int(pred)) for pred in assignments],
        dtype=np.int64,
    )
    correct = (aligned == labels).astype(np.float64)
    ece = 0.0
    mce = 0.0
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (confidences > start) & (confidences <= end)
        if start == 0.0:
            mask = (confidences >= start) & (confidences <= end)
        if not np.any(mask):
            continue
        accuracy = float(correct[mask].mean())
        confidence = float(confidences[mask].mean())
        gap = abs(accuracy - confidence)
        ece += float(mask.mean()) * gap
        mce = max(mce, gap)
    return {"calibration_ece": float(ece), "calibration_mce": float(mce)}


def _hungarian_alignment(
    labels: np.ndarray,
    assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    assignments = np.asarray(assignments, dtype=np.int64)
    n = max(labels.max(initial=0), assignments.max(initial=0)) + 1
    matrix = np.zeros((n, n), dtype=np.int64)
    for y, pred in zip(labels, assignments):
        matrix[pred, y] += 1
    row_ind, col_ind = linear_sum_assignment(matrix.max() - matrix)
    return matrix, row_ind, col_ind
