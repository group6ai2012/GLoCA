from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import warnings

import yaml


def load_experiment_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return config


def save_experiment_config(config: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(deepcopy(config), handle, sort_keys=False)


def require_keys(mapping: dict[str, Any], keys: list[str], prefix: str = "") -> None:
    if not isinstance(mapping, dict):
        label = prefix or "config"
        raise ValueError(f"{label} must be a mapping.")
    missing = [key for key in keys if key not in mapping]
    if missing:
        label = prefix or "config"
        raise ValueError(f"Missing required config key(s) in {label}: {', '.join(missing)}")


def validate_common_config(config: dict[str, Any]) -> None:
    from src.experiments.cli import seed_value
    require_keys(config, ["experiment", "dataset", "backbone", "gloca", "head", "prediction", "trainer"])
    require_keys(config["experiment"], ["name", "seed", "output_dir"], "experiment")
    require_keys(config["dataset"], ["name", "training_views"], "dataset")
    require_keys(config["backbone"], ["family", "variant", "freeze", "image_size"], "backbone")
    require_keys(config["gloca"], ["enabled", "name", "embedding_dim", "normalize_output"], "gloca")
    require_keys(config["head"], ["name", "n_clusters", "training_mode"], "head")
    require_keys(config["prediction"], ["deterministic", "export_embeddings", "export_attention"], "prediction")
    require_keys(config["trainer"], ["max_epochs", "batch_size", "lr", "precision", "num_workers"], "trainer")
    seed_value(config["experiment"]["seed"])
    if not isinstance(config["trainer"].get("deterministic", True), bool):
        raise ValueError("trainer.deterministic must be a boolean")
    if not config["backbone"]["freeze"]:
        raise ValueError("DINOv2 requires backbone.freeze=true")


def validate_kmeans_config(config: dict[str, Any]) -> None:
    validate_common_config(config)
    require_keys(config, ["baseline"])
    require_keys(
        config["baseline"],
        ["spherical", "kmeans_init", "kmeans_n_init", "kmeans_max_iter", "kmeans_tol"],
        "baseline",
    )


def validate_global_local_kmeans_config(config: dict[str, Any]) -> None:
    validate_common_config(config)
    require_keys(config, ["baseline"])
    require_keys(
        config["baseline"],
        [
            "embedding_source",
            "normalize_embedding",
            "spherical",
            "kmeans_init",
            "kmeans_n_init",
            "kmeans_max_iter",
            "kmeans_tol",
        ],
        "baseline",
    )
    if str(config["head"]["name"]).lower() != "kmeans":
        raise ValueError("Global-local baseline requires head.name='kmeans'.")
    if str(config["head"]["training_mode"]).lower() != "single_view":
        raise ValueError("Global-local baseline requires head.training_mode='single_view'.")
    if bool(config["gloca"]["enabled"]):
        raise ValueError("Global-local baseline requires gloca.enabled=false.")
    if not bool(config["backbone"]["freeze"]):
        raise ValueError("Global-local baseline requires backbone.freeze=true.")
    if str(config["baseline"]["embedding_source"]).lower() != "cls_mean_patch_concat":
        raise ValueError(
            "Global-local baseline requires "
            "baseline.embedding_source='cls_mean_patch_concat'."
        )
    if not bool(config["baseline"]["normalize_embedding"]):
        raise ValueError("Global-local baseline requires baseline.normalize_embedding=true.")
    if bool(config["baseline"]["spherical"]):
        raise ValueError("Global-local baseline uses ordinary K-means and requires spherical=false.")
    if not bool(config["prediction"]["deterministic"]):
        raise ValueError("Global-local baseline requires prediction.deterministic=true.")


def validate_global_local_sweep_config(config: dict[str, Any]) -> None:
    validate_baseline_sweep_config(config)
    for run in config["sweep"]["runs"]:
        if run["runner"] not in ['global_local_kmeans']:
            raise ValueError(f"Unsupported runner: {run['runner']!r}")
    from src.experiments.baselines import _build_plans
    _build_plans(config)


def validate_dec_idec_config(config: dict[str, Any]) -> None:
    validate_common_config(config)
    require_keys(config, ["baseline"])
    require_keys(
        config["baseline"],
        [
            "kmeans_init",
            "kmeans_n_init",
            "kmeans_max_iter",
            "kmeans_tol",
            "input_dim",
            "hidden_dims",
            "latent_dim",
            "pretrain_epochs",
            "refine_epochs",
            "pretrain_lr",
            "refine_lr",
            "lambda_recon",
            "alpha",
            "target_update_interval",
        ],
        "baseline",
    )


def validate_cached_cls_config(config: dict[str, Any]) -> None:
    validate_common_config(config)
    if config["gloca"]["enabled"] or not config["gloca"]["normalize_output"]:
        raise ValueError("Cached CLS reference baselines require normalized CLS with GLoCA disabled")
    require_keys(config, ["cached_baseline"])
    head_name = str(config["head"]["name"]).lower()
    baseline = config["cached_baseline"]
    if head_name == "gmm":
        require_keys(
            baseline,
            [
                "gmm_covariance_type",
                "gmm_init_params",
                "gmm_n_init",
                "gmm_max_iter",
                "gmm_tol",
                "gmm_reg_covar",
            ],
            "cached_baseline",
        )
    elif head_name == "hdbscan":
        require_keys(
            baseline,
            [
                "hdbscan_min_cluster_size",
                "hdbscan_min_samples",
                "hdbscan_cluster_selection_epsilon",
                "hdbscan_metric",
                "hdbscan_algorithm",
                "hdbscan_leaf_size",
                "hdbscan_n_jobs",
                "hdbscan_cluster_selection_method",
                "hdbscan_allow_single_cluster",
            ],
            "cached_baseline",
        )
    elif head_name == "supervised_mlp":
        require_keys(
            baseline,
            ["mlp_hidden_dims", "mlp_weight_decay", "mlp_split_ratios"],
            "cached_baseline",
        )
        if len(baseline["mlp_hidden_dims"]) != 2:
            raise ValueError("cached_baseline.mlp_hidden_dims must contain exactly two values.")
        if len(baseline["mlp_split_ratios"]) != 3 or abs(sum(float(value) for value in baseline["mlp_split_ratios"]) - 1.0) > 1e-6:
            raise ValueError("cached_baseline.mlp_split_ratios must contain three values summing to 1.")
    else:
        raise ValueError(
            "Cached CLS runner expects head.name to be one of 'gmm', 'hdbscan', or 'supervised_mlp'."
        )


def validate_cached_cls_sweep_config(config: dict[str, Any]) -> None:
    validate_baseline_sweep_config(config)
    for run in config["sweep"]["runs"]:
        if run["runner"] not in ['cached_cls_gmm', 'cached_cls_hdbscan', 'cached_cls_mlp']:
            raise ValueError(f"Unsupported runner: {run['runner']!r}")
    from src.experiments.baselines import _build_plans
    _build_plans(config)


def validate_propos_config(config: dict[str, Any]) -> None:
    validate_common_config(config)
    require_keys(config, ["propos"])
    require_keys(
        config["propos"],
        [
            "embedding_dim",
            "projection_dim",
            "projection_hidden_dim",
            "predictor_hidden_dim",
            "temperature",
            "sigma",
            "lambda_psl",
            "ema_momentum",
            "ema_momentum_max",
            "ema_momentum_increase",
            "kmeans_interval",
            "kmeans_init",
            "kmeans_n_init",
            "kmeans_max_iter",
            "kmeans_tol",
            "warmup_epochs",
            "predictor_lr_multiplier",
            "weight_decay",
            "optimizer",
            "symmetric_loss",
            "profile_batches",
            "gloca_lr_multiplier",
            "gloca_alpha_lr_multiplier",
            "freeze_gloca",
            "freeze_gloca_epochs",
            "log_gloca_diagnostics",
        ],
        "propos",
    )


def validate_cdc_config(config: dict[str, Any]) -> None:
    validate_common_config(config)
    require_keys(config, ["cdc"])
    require_keys(
        config["cdc"],
        [
            "embedding_dim",
            "hidden_dim",
            "training_views",
            "meta_batch_size",
            "sub_batch_size",
            "per_class_selected_num",
            "calibration_k",
            "meta_batch_drop_last",
            "w_en",
            "prototype_init",
            "orthogonalize_init",
            "kmeans_init",
            "kmeans_n_init",
            "kmeans_max_iter",
            "kmeans_tol",
            "weight_decay",
            "optimizer",
            "gloca_lr_multiplier",
            "gloca_alpha_lr_multiplier",
            "freeze_gloca",
            "freeze_gloca_epochs",
            "log_gloca_diagnostics",
            "augment",
        ],
        "cdc",
    )
    batch_size = int(config["trainer"]["batch_size"])
    if batch_size <= 0:
        raise ValueError(f"trainer.batch_size must be positive for CDC, got {batch_size}")
    meta_batch_size = int(config["cdc"]["meta_batch_size"])
    sub_batch_size = int(config["cdc"]["sub_batch_size"])
    calibration_k = int(config["cdc"]["calibration_k"])
    if meta_batch_size < batch_size:
        raise ValueError(
            f"cdc.meta_batch_size must be >= trainer.batch_size, got {meta_batch_size} < {batch_size}"
        )
    if sub_batch_size <= 0:
        raise ValueError(f"cdc.sub_batch_size must be positive, got {sub_batch_size}")
    if sub_batch_size > meta_batch_size:
        raise ValueError(
            f"cdc.sub_batch_size must be <= cdc.meta_batch_size, got {sub_batch_size} > {meta_batch_size}"
        )
    if calibration_k > meta_batch_size:
        raise ValueError(
            f"cdc.calibration_k must be <= cdc.meta_batch_size, got {calibration_k} > {meta_batch_size}"
        )
    raw_clusters = config["head"]["n_clusters"]
    if raw_clusters != "auto" and meta_batch_size < int(raw_clusters):
        warnings.warn(
            "CDC meta_batch_size is smaller than head.n_clusters; reliable sample selection may be sparse.",
            stacklevel=2,
        )


def validate_baseline_sweep_config(config: dict[str, Any]) -> None:
    require_keys(config, ["experiment", "base_config", "sweep"])
    require_keys(config["experiment"], ["output_dir"], "experiment")
    require_keys(config["sweep"], ["datasets", "seeds", "skip_existing", "fail_fast", "runs"], "sweep")
    from src.experiments.cli import seed_value
    seeds = config["sweep"]["seeds"]
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(map(seed_value, seeds))):
        raise ValueError("sweep.seeds must be a non-empty list of unique seeds")
    if not isinstance(config["sweep"]["datasets"], list) or not config["sweep"]["datasets"]:
        raise ValueError("sweep.datasets must be a non-empty list")
    if not isinstance(config["sweep"]["runs"], list) or not config["sweep"]["runs"]:
        raise ValueError("sweep.runs must be a non-empty list.")
    names = [run.get("name") for run in config["sweep"]["runs"] if isinstance(run, dict)]
    if len(names) != len(set(names)):
        raise ValueError("sweep run names must be unique")
    for index, run in enumerate(config["sweep"]["runs"]):
        prefix = f"sweep.runs[{index}]"
        require_keys(run, ["name", "runner", "expected", "config"], prefix)
        require_keys(run["expected"], ["head", "gloca"], f"{prefix}.expected")
        require_keys(run["config"], ["experiment"], f"{prefix}.config")
        require_keys(run["config"]["experiment"], ["name"], f"{prefix}.config.experiment")


def validate_propos_diagnostics_config(config: dict[str, Any]) -> None:
    require_keys(config, ["experiment", "base_config", "diagnostics"])
    require_keys(config["experiment"], ["output_dir"], "experiment")
    require_keys(config["diagnostics"], ["force", "runs"], "diagnostics")
    if not isinstance(config["diagnostics"]["runs"], list) or not config["diagnostics"]["runs"]:
        raise ValueError("diagnostics.runs must be a non-empty list.")
    for index, run in enumerate(config["diagnostics"]["runs"]):
        prefix = f"diagnostics.runs[{index}]"
        require_keys(run, ["id", "name", "config"], prefix)
        require_keys(run["config"], ["experiment"], f"{prefix}.config")
        require_keys(run["config"]["experiment"], ["name"], f"{prefix}.config.experiment")
