from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

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
    require_keys(config, ["experiment", "dataset", "backbone", "gloca", "head", "prediction", "trainer"])
    require_keys(config["experiment"], ["name", "seed", "output_dir"], "experiment")
    require_keys(config["dataset"], ["name", "training_views"], "dataset")
    require_keys(config["backbone"], ["family", "variant", "freeze", "image_size"], "backbone")
    require_keys(config["gloca"], ["enabled", "name", "embedding_dim", "normalize_output"], "gloca")
    require_keys(config["head"], ["name", "n_clusters", "training_mode"], "head")
    require_keys(config["prediction"], ["deterministic", "export_embeddings", "export_attention"], "prediction")
    require_keys(config["trainer"], ["max_epochs", "batch_size", "lr", "precision", "num_workers"], "trainer")


def validate_kmeans_config(config: dict[str, Any]) -> None:
    validate_common_config(config)
    require_keys(config, ["baseline"])
    require_keys(
        config["baseline"],
        ["spherical", "kmeans_init", "kmeans_n_init", "kmeans_max_iter", "kmeans_tol"],
        "baseline",
    )


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


def validate_baseline_sweep_config(config: dict[str, Any]) -> None:
    require_keys(config, ["experiment", "base_config", "sweep"])
    require_keys(config["experiment"], ["output_dir"], "experiment")
    require_keys(config["sweep"], ["datasets", "seeds", "skip_existing", "fail_fast", "runs"], "sweep")
    if not isinstance(config["sweep"]["runs"], list) or not config["sweep"]["runs"]:
        raise ValueError("sweep.runs must be a non-empty list.")
    for index, run in enumerate(config["sweep"]["runs"]):
        prefix = f"sweep.runs[{index}]"
        require_keys(run, ["name", "runner", "expected", "config"], prefix)
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
