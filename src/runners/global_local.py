from __future__ import annotations

from src.experiments.reproducibility import record_run

import time
from copy import deepcopy
from typing import Any

import torch

from src.data import ClusteringDataModule, get_dataset_spec
from src.evaluation import compute_clustering_metrics
from src.evaluation.assignment_schema import GLOBAL_LOCAL_METRIC_FIELDS
from src.experiments.config import validate_global_local_kmeans_config
from src.experiments.outputs import write_outputs
from src.models.baselines.global_local import FrozenCLSMeanPatchConcat
from src.models.clustering.kmeans import fit_kmeans
from src.runners.common import (
    apply_runtime_settings,
    assert_backbone_frozen,
    finalize_run,
    prepare_runner_context,
)
from src.runners.diagnostics import (
    attention_diagnostics,
    cluster_diagnostics,
    embedding_diagnostics,
)
from src.utils import ExperimentResult


EMBEDDING_SOURCE = "cls_mean_patch_concat"


def extract_cached_global_local_embeddings(config: dict[str, Any]) -> dict[str, Any]:
    from src.features import DINOv2Backbone
    from src.runners.embedding_export import extract_deterministic_embeddings

    config = deepcopy(config)
    validate_global_local_kmeans_config(config)
    runtime_logs = apply_runtime_settings(config)
    spec = get_dataset_spec(config["dataset"])
    datamodule = ClusteringDataModule(
        spec=spec,
        image_size=int(config["backbone"]["image_size"]),
        batch_size=int(config["trainer"]["batch_size"]),
        num_workers=int(config["trainer"]["num_workers"]),
        seed=int(config["experiment"]["seed"]),
        deterministic=bool(config["trainer"].get("deterministic", True)),
        training_mode="single_view",
        training_views=1,
    )
    backbone = DINOv2Backbone(
        variant=config["backbone"]["variant"],
        image_size=int(config["backbone"]["image_size"]),
        freeze=bool(config["backbone"]["freeze"]),
        revision=config["backbone"].get("revision"),
    )
    assert_backbone_frozen(backbone)
    feature_builder = FrozenCLSMeanPatchConcat(
        normalize_output=bool(config["baseline"]["normalize_embedding"]),
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and config["trainer"].get("accelerator", "auto") in {"auto", "gpu", "cuda"}
        else "cpu"
    )
    extracted = extract_deterministic_embeddings(
        backbone=backbone,
        adapter=feature_builder,
        datamodule=datamodule,
        device=device,
        normalize_cls=False,
    )
    extracted["runtime_logs"] = runtime_logs
    extracted["spec"] = spec
    extracted["backbone"] = backbone
    extracted["feature_builder"] = feature_builder
    from src.experiments.reproducibility import feature_cache_key
    extracted["feature_key"] = feature_cache_key(config, EMBEDDING_SOURCE)
    return extracted


@record_run
def run_cached_global_local_kmeans(
    config: dict[str, Any],
    cache: dict[str, Any],
) -> ExperimentResult:
    config = deepcopy(config)
    validate_global_local_kmeans_config(config)
    ctx = prepare_runner_context(config, expected_head="kmeans")
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = config["baseline"]

    fit_start = time.perf_counter()
    kmeans_result = fit_kmeans(
        cache["embeddings"],
        ctx.n_clusters,
        spherical=False,
        init=str(baseline["kmeans_init"]),
        n_init=int(baseline["kmeans_n_init"]),
        max_iter=int(baseline["kmeans_max_iter"]),
        tol=float(baseline["kmeans_tol"]),
        seed=ctx.seed,
        device=ctx.device,
    )
    head_train_time_s = time.perf_counter() - fit_start
    total_time_s = float(cache["cache_time_s"]) + head_train_time_s
    embeddings = cache["embeddings"].float()
    embedding_dim = int(embeddings.shape[1])
    trainable_parameter_count = int(
        sum(parameter.numel() for parameter in cache["feature_builder"].parameters())
    )

    metrics_extras = {
        "embedding_source": EMBEDDING_SOURCE,
        "patch_pooling": "mean",
        "feature_fusion": "concat",
        "final_embedding_normalized": True,
        "embedding_dim": embedding_dim,
        "trainable_parameter_count": trainable_parameter_count,
        "_extra_fields": GLOBAL_LOCAL_METRIC_FIELDS,
    }
    logs_extras = {
        "non_trainable_baseline": True,
        "checkpoint_written": False,
        "uses_cached_backbone_features": True,
        "backbone_requires_grad_false": not any(
            parameter.requires_grad for parameter in cache["backbone"].parameters()
        ),
        "deterministic_single_view_transform": True,
        "embedding_source": EMBEDDING_SOURCE,
        "patch_pooling": "mean",
        "feature_fusion": "concat",
        "final_embedding_normalized": True,
        "embedding_dim": embedding_dim,
        "trainable_parameter_count": trainable_parameter_count,
        "learned_adapter": False,
        "spherical": False,
        "kmeans_backend": "torch",
        "kmeans_algorithm": kmeans_result["logs"]["algorithm"],
        "kmeans_init": str(baseline["kmeans_init"]),
        "kmeans_n_init": int(baseline["kmeans_n_init"]),
        "kmeans_max_iter": int(baseline["kmeans_max_iter"]),
        "kmeans_tol": float(baseline["kmeans_tol"]),
        "kmeans_logs": kmeans_result["logs"],
        "backbone_cache_time_s": float(cache["cache_time_s"]),
        "head_train_time_s": head_train_time_s,
        "total_time_s": total_time_s,
        **cache.get("runtime_logs", {}),
    }
    return finalize_run(
        output_dir=ctx.output_dir,
        config=config,
        spec=cache["spec"],
        head="kmeans",
        image_ids=cache["image_ids"],
        labels=cache["labels"],
        assignments=kmeans_result["assignments"],
        embeddings=embeddings,
        attention=None,
        patch_grid=cache["patch_grid"],
        seed=ctx.seed,
        backbone_cache_time_s=float(cache["cache_time_s"]),
        head_train_time_s=head_train_time_s,
        total_time_s=total_time_s,
        inference_time_s=0.0,
        uses_cached_backbone_features=True,
        metrics_extras=metrics_extras,
        logs_extras=logs_extras,
        assignment_extras={"embedding_source": EMBEDDING_SOURCE},
        metrics_fn=compute_clustering_metrics,
        cluster_diagnostics_fn=cluster_diagnostics,
        embedding_diagnostics_fn=embedding_diagnostics,
        attention_diagnostics_fn=attention_diagnostics,
        writer=write_outputs,
    )
