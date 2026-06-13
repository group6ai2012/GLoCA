from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def clustering_accuracy(labels: np.ndarray, assignments: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    assignments = assignments.astype(np.int64)
    n = max(labels.max(initial=0), assignments.max(initial=0)) + 1
    matrix = np.zeros((n, n), dtype=np.int64)
    for y, pred in zip(labels, assignments):
        matrix[pred, y] += 1
    row_ind, col_ind = linear_sum_assignment(matrix.max() - matrix)
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

