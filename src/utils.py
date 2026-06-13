from __future__ import annotations

import random
from pathlib import Path
from typing import Any


class ExperimentResult(dict):
    @property
    def output_dir(self) -> Path:
        return Path(self["output_dir"])


def resolve_output_dir(config: dict[str, Any]) -> Path:
    base = Path(config["experiment"]["output_dir"])
    return (
        base
        / config["experiment"]["name"]
        / config["head"]["name"]
        / config["dataset"]["name"]
        / f"seed_{int(config['experiment']['seed'])}"
    )


def resolve_device(config: dict[str, Any]) -> torch.device:
    import torch

    requested = config["trainer"].get("accelerator", "auto")
    if requested in {"auto", "gpu", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_gloca_name(config: dict[str, Any]) -> str:
    gloca_config = config.get("gloca", {})
    if not bool(gloca_config.get("enabled", False)):
        return "disabled"
    return str(gloca_config.get("name") or gloca_config.get("variation") or "disabled")


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
