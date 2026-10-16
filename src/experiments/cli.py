"""Shared experiment arguments and process settings (no numerical imports)."""
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy


def bootstrap(*, reexec: bool = False) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        if reexec:
            os.execv(sys.executable, [sys.executable, *sys.argv])


def seed_value(value: str | int) -> int:
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if isinstance(value, bool) or str(seed) != str(value).strip() or not 0 <= seed < 2**32:
        raise argparse.ArgumentTypeError("seed must be an integer between 0 and 4294967295")
    return seed


def add_seed_arguments(parser: argparse.ArgumentParser, *, sweep: bool = False) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=seed_value, help="Override the YAML seed.")
    if sweep:
        group.add_argument("--seeds", type=seed_value, nargs="+", help="Replace the YAML sweep seeds.")


def override_seeds(config: dict, seed: int | None = None, seeds: list[int] | None = None) -> dict:
    result = deepcopy(config)
    if seed is not None and seeds is not None:
        raise ValueError("--seed and --seeds are mutually exclusive")
    values = seeds if seeds is not None else ([seed] if seed is not None else None)
    if values is not None:
        values = [seed_value(value) for value in values]
        if not values or len(values) != len(set(values)):
            raise ValueError("sweep seeds must be non-empty and unique")
        if "sweep" in result:
            result["sweep"]["seeds"] = values
        elif seeds is not None:
            raise ValueError("--seeds is supported only for baseline sweeps")
        elif "diagnostics" in result:
            result["base_config"]["experiment"]["seed"] = seed
            for run in result["diagnostics"]["runs"]:
                run["config"].setdefault("experiment", {})["seed"] = seed
        else:
            result["experiment"]["seed"] = seed
    return result
