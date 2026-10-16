from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import statistics
import math
from typing import Any


from src.evaluation.assignment_schema import AGGREGATE_METRIC_FIELDS, DEC_IDEC_METRIC_FIELDS, GLOBAL_LOCAL_METRIC_FIELDS, METRIC_FIELDS
from src.experiments.config import (
    load_experiment_config,
    validate_baseline_sweep_config,
)
from src.utils import resolve_output_dir, resolve_gloca_name
from src.experiments.cli import add_seed_arguments, override_seeds
from src.experiments.reproducibility import is_complete, verification_session
EXTRA_FIELDS = [
    "reference_n_classes",
    "n_discovered_clusters",
    "noise_count",
    "noise_fraction",
    "mean_membership_probability",
    "classification_accuracy",
    "balanced_accuracy",
    "macro_f1",
    "test_loss",
    "best_epoch",
    "best_val_accuracy",
    "best_val_loss",
    "train_n_images",
    "validation_n_images",
    "test_n_images",
    "trainable_parameter_count",
    "evaluation_split",
]



SUMMARY_FIELDS = [
    "status",
    "error",
    "baseline_name",
    *METRIC_FIELDS,
    *DEC_IDEC_METRIC_FIELDS,
    *[f for f in GLOBAL_LOCAL_METRIC_FIELDS if f not in METRIC_FIELDS],
    *[f for f in EXTRA_FIELDS if f not in METRIC_FIELDS and f not in GLOBAL_LOCAL_METRIC_FIELDS],
    "output_dir",
]

GROUP_FIELDS = ["baseline_name", "embedding_source", "evaluation_split", "backbone", "gloca", "head", "dataset"]
SUMMARY_FIELDS = list(dict.fromkeys(SUMMARY_FIELDS))
AGG_FIELDS = list(dict.fromkeys([*GROUP_FIELDS, *AGGREGATE_METRIC_FIELDS, *[f"{metric}_{stat}" for metric in ["classification_accuracy", "balanced_accuracy", "macro_f1", "test_loss"] for stat in ["mean", "std"]]]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YAML-defined clustering baseline sweeps.")
    parser.add_argument("config_path", type=Path)
    add_seed_arguments(parser, sweep=True)
    return parser.parse_args()


@verification_session()
def main() -> None:
    args = parse_args()
    config = override_seeds(load_experiment_config(args.config_path), args.seed, args.seeds)
    validate_baseline_sweep_config(config)
    plans = _build_plans(config)

    rows: list[dict[str, Any]] = []
    output_root = Path(config["experiment"]["output_dir"])
    fail_fast = bool(config["sweep"]["fail_fast"])
    skip_existing = bool(config["sweep"]["skip_existing"])
    caches = {}
    for index, plan in enumerate(plans, start=1):
        print(
            f"[{index}/{len(plans)}] dataset={plan['dataset']} seed={plan['seed']} "
            f"baseline={plan['baseline_name']}",
            flush=True,
        )
        try:
            output_dir = resolve_output_dir(plan["config"])
            if skip_existing and is_complete(plan["config"]):
                row = _summary_row_from_output(plan, output_dir, status="skipped", error="")
            else:
                result = _run_plan(plan, caches)
                row = _summary_row_from_result(plan, result, status="success", error="")
        except Exception as exc:
            row = _failed_row(plan, exc)
            print(f"FAILED: {exc}", flush=True)
        rows.append(row)
        _write_summaries(output_root, rows)
        if fail_fast and row["status"] == "failed":
            raise SystemExit(1)

    print(f"Wrote summary to {output_root / 'baseline_summary.csv'}")
    print(f"Wrote aggregate summary to {output_root / 'baseline_summary_agg.csv'}")
    if any(row["status"] == "failed" for row in rows):
        raise SystemExit(1)


def _build_plans(config: dict[str, Any]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    output_root = Path(config["experiment"]["output_dir"])
    for dataset in config["sweep"]["datasets"]:
        for seed in config["sweep"]["seeds"]:
            for run in config["sweep"]["runs"]:
                run_config = deepcopy(config["base_config"])
                _deep_update(run_config, deepcopy(run["config"]))
                _deep_update(
                    run_config,
                    {
                        "experiment": {
                            "name": run["config"]["experiment"]["name"],
                            "seed": int(seed),
                            "output_dir": str(output_root / str(dataset) / f"seed_{int(seed)}") if run["runner"] in {"kmeans", "dec_idec"} else str(output_root),
                        },
                        "dataset": {"name": str(dataset)},
                    },
                )
                if str(dataset) in {"plantseg", "plantvillage"} and not run_config["dataset"].get("root"):
                    run_config["dataset"].setdefault("manifest", f"data/manifests/{dataset}.json")
                _validate_run_config(run["runner"], run_config)
                effective_head = str(run_config["head"]["name"])
                if run["runner"] == "kmeans" and run_config["baseline"]["spherical"]:
                    effective_head = "spherical_kmeans"
                if run["expected"]["head"] != effective_head or run["expected"]["gloca"] != resolve_gloca_name(run_config):
                    raise ValueError(f"Expected head/GLoCA metadata differs from config for {run['name']}")
                if "embedding_source" in run["expected"] and run["expected"]["embedding_source"] != run_config.get("baseline", {}).get("embedding_source"):
                    raise ValueError(f"Expected embedding source differs from config for {run['name']}")
                plans.append(
                    {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "baseline_name": str(run["name"]),
                        "runner": str(run["runner"]),
                        "expected": dict(run["expected"]),
                        "config": run_config,
                    }
                )
    destinations = [resolve_output_dir(plan["config"]).resolve() for plan in plans]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Sweep contains duplicate output destinations")
    return plans


RUNNERS = {
    "kmeans": ("kmeans", "run_kmeans", "validate_kmeans_config"),
    "dec_idec": ("dec_idec", "run_dec_idec", "validate_dec_idec_config"),
    "cached_cls_gmm": ("cached_cls", "run_cached_cls_gmm", "validate_cached_cls_config"),
    "cached_cls_hdbscan": ("cached_cls", "run_cached_cls_hdbscan", "validate_cached_cls_config"),
    "cached_cls_mlp": ("cached_cls", "run_cached_cls_mlp", "validate_cached_cls_config"),
    "global_local_kmeans": ("global_local", "run_cached_global_local_kmeans", "validate_global_local_kmeans_config"),
}


def _validate_run_config(runner: str, config: dict[str, Any]) -> None:
    from src.experiments import config as validation
    if runner not in RUNNERS:
        raise ValueError(f"Unsupported baseline runner: {runner!r}")
    getattr(validation, RUNNERS[runner][2])(config)
    expected_head = {"cached_cls_gmm": "gmm", "cached_cls_hdbscan": "hdbscan", "cached_cls_mlp": "supervised_mlp"}.get(runner)
    if expected_head and config["head"]["name"] != expected_head:
        raise ValueError(f"{runner} requires head.name={expected_head}")


def _run_plan(plan: dict, caches: dict):
    from src.experiments.reproducibility import feature_cache_key
    from src.runners.cached_cls import extract_cached_cls_embeddings
    from src.runners.global_local import extract_cached_global_local_embeddings
    config = plan["config"]
    if config["gloca"]["enabled"]:
        return _load_runner(plan["runner"])(config)
    source = "cls_mean_patch_concat" if plan["runner"] == "global_local_kmeans" else "cls"
    key = feature_cache_key(config, source)
    if key not in caches:
        try:
            extractor = extract_cached_global_local_embeddings if source != "cls" else extract_cached_cls_embeddings
            caches[key] = extractor(config)
        except Exception as exc:
            caches[key] = exc
    if isinstance(caches[key], Exception):
        raise caches[key]
    return _load_runner(plan["runner"])(config, caches[key])


def _summary_row_from_result(plan: dict[str, Any], result: dict[str, Any], status: str, error: str) -> dict[str, Any]:
    output_dir = Path(result["output_dir"])
    return _summary_row(plan, output_dir, dict(result.get("metrics", {})), status, error)


def _summary_row_from_output(plan: dict[str, Any], output_dir: Path, status: str, error: str) -> dict[str, Any]:
    return _summary_row(plan, output_dir, _read_metrics(output_dir / "metrics.csv"), status, error)


def _summary_row(
    plan: dict[str, Any],
    output_dir: Path,
    metrics: dict[str, Any],
    status: str,
    error: str,
) -> dict[str, Any]:
    logs = _read_json(output_dir / "logs.json")
    expected = plan["expected"]
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "status": status,
            "error": error,
            "baseline_name": plan["baseline_name"],
            "output_dir": str(output_dir),
        }
    )
    for field in SUMMARY_FIELDS:
        if field in row and row[field] != "":
            continue
        row[field] = metrics.get(field, logs.get(field, expected.get(field, "")))
    row["dataset"] = row["dataset"] or plan["dataset"]
    row["seed"] = row["seed"] or plan["seed"]
    row["backbone"] = row["backbone"] or plan["config"]["backbone"]["variant"]
    row["head"] = row["head"] or expected["head"]
    row["gloca"] = row["gloca"] or expected["gloca"]
    return row


def _failed_row(plan: dict[str, Any], exc: Exception) -> dict[str, Any]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "status": "failed",
            "error": str(exc),
            "dataset": plan["dataset"],
            "seed": plan["seed"],
            "baseline_name": plan["baseline_name"],
            "output_dir": str(resolve_output_dir(plan["config"])),
            "backbone": plan["config"]["backbone"]["variant"],
            **plan["expected"],
        }
    )
    return row


def _load_runner(name: str):
    from importlib import import_module
    if name not in RUNNERS:
        raise ValueError(f"Unknown runner: {name}")
    module, symbol, _ = RUNNERS[name]
    return getattr(import_module("src.runners." + module), symbol)


def _write_summaries(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "baseline_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "baseline_summary_agg.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGG_FIELDS)
        writer.writeheader()
        writer.writerows(_aggregate_rows(rows))


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in GROUP_FIELDS)
        grouped.setdefault(key, []).append(row)

    aggregated = []
    for key, group_rows in sorted(grouped.items()):
        success_rows = [row for row in group_rows if row.get("status") in {"success", "skipped"}]
        out = {field: "" for field in AGG_FIELDS}
        out.update(dict(zip(GROUP_FIELDS, key)))
        out["n_runs"] = len(success_rows)
        out["seeds"] = " ".join(str(row["seed"]) for row in success_rows)
        for metric in ["ari", "nmi", "acc", "silhouette", "classification_accuracy", "balanced_accuracy", "macro_f1", "test_loss"]:
            values = [_to_float(row.get(metric)) for row in success_rows]
            values = [value for value in values if value is not None]
            out[f"{metric}_mean"] = _mean(values)
            out[f"{metric}_std"] = _std(values)
        total_times = [_to_float(row.get("total_time_s")) for row in success_rows]
        total_times = [value for value in total_times if value is not None]
        out["total_time_s_mean"] = _mean(total_times)
        out["total_time_s_std"] = _std(total_times)
        peak_gpu_values = [_to_float(row.get("peak_gpu_mb")) for row in success_rows]
        peak_gpu_values = [value for value in peak_gpu_values if value is not None]
        out["peak_gpu_mb"] = "" if not peak_gpu_values else str(max(peak_gpu_values))
        aggregated.append(out)
    return aggregated


def _mean(values: list[float]) -> str:
    return "" if not values else str(statistics.mean(values))


def _std(values: list[float]) -> str:
    return "" if len(values) < 2 else str(statistics.stdev(values))


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value) if math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def _read_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


if __name__ == "__main__":
    main()
