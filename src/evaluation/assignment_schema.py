from __future__ import annotations

from typing import Any


ASSIGNMENT_REQUIRED_FIELDS = [
    "head",
    "backbone",
    "gloca",
    "dataset",
    "seed",
    "n_clusters",
    "image_ids",
    "labels",
    "assignments",
    "patch_grid",
]

ASSIGNMENT_ARRAY_FIELDS = ["image_ids", "labels", "assignments"]

METRIC_FIELDS = [
    "experiment",
    "head",
    "backbone",
    "dataset",
    "seed",
    "gloca",
    "n_clusters",
    "n_images",
    "ari",
    "nmi",
    "acc",
    "silhouette",
    "n_nonempty_clusters",
    "cluster_size_min",
    "cluster_size_max",
    "cluster_size_entropy",
    "embedding_variance_mean",
    "embedding_norm_mean",
    "embedding_norm_std",
    "attention_entropy",
    "attention_max",
    "attention_top5_mass",
    "attention_variance",
    "backbone_cache_time_s",
    "head_train_time_s",
    "total_time_s",
    "inference_time_s",
    "peak_gpu_mb",
    "uses_cached_backbone_features",
]

DEC_IDEC_METRIC_FIELDS = [
    "input_dim",
    "hidden_dims",
    "latent_dim",
    "pretrain_epochs",
    "refine_epochs",
    "pretrain_lr",
    "refine_lr",
    "lambda_recon",
    "alpha",
    "target_update_mode",
    "target_update_interval",
]

AGGREGATE_METRIC_FIELDS = [
    "backbone",
    "gloca",
    "head",
    "dataset",
    "n_runs",
    "seeds",
    "ari_mean",
    "ari_std",
    "nmi_mean",
    "nmi_std",
    "acc_mean",
    "acc_std",
    "silhouette_mean",
    "silhouette_std",
    "total_time_s_mean",
    "total_time_s_std",
    "peak_gpu_mb",
]


def validate_assignment_payload(payload: dict) -> None:
    missing = [field for field in ASSIGNMENT_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Assignment payload is missing required fields: {missing}")
    lengths = {field: len(payload.get(field, [])) for field in ASSIGNMENT_ARRAY_FIELDS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Assignment length mismatch: {lengths}")


def ordered_metrics_row(row: dict[str, Any], extra_fields: list[str] | None = None) -> dict[str, Any]:
    fields = list(METRIC_FIELDS)
    for field in extra_fields or []:
        if field not in fields:
            fields.append(field)
    return {field: row.get(field, "") for field in fields}
