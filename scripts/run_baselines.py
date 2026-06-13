from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.assignment_schema import AGGREGATE_METRIC_FIELDS, DEC_IDEC_METRIC_FIELDS, METRIC_FIELDS
from src.experiments.config import (
    load_experiment_config,
    validate_baseline_sweep_config,
    validate_dec_idec_config,
    validate_kmeans_config,
)
from src.utils import resolve_output_dir


SUMMARY_FIELDS = [
    "status",
    "error",
    "baseline_name",
    *METRIC_FIELDS,
    *DEC_IDEC_METRIC_FIELDS,
    "output_dir",
]

AGG_FIELDS = AGGREGATE_METRIC_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YAML-defined clustering baseline sweeps.")
    parser.add_argument("config_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config_path)
    validate_baseline_sweep_config(config)
    plans = _build_plans(config)

    rows: list[dict[str, Any]] = []
    output_root = Path(config["experiment"]["output_dir"])
    fail_fast = bool(config["sweep"]["fail_fast"])
    skip_existing = bool(config["sweep"]["skip_existing"])
    for index, plan in enumerate(plans, start=1):
        print(
            f"[{index}/{len(plans)}] dataset={plan['dataset']} seed={plan['seed']} "
            f"baseline={plan['baseline_name']}",
            flush=True,
        )
        try:
            output_dir = resolve_output_dir(plan["config"])
            if skip_existing and (output_dir / "metrics.csv").exists():
                row = _summary_row_from_output(plan, output_dir, status="skipped", error="")
            else:
                result = _load_runner(plan["runner"])(plan["config"])
                row = _summary_row_from_result(plan, result, status="success", error="")
        except Exception as exc:
            if fail_fast:
                raise
            row = _failed_row(plan, exc)
            print(f"FAILED: {exc}", flush=True)
        rows.append(row)
        _write_summaries(output_root, rows)

    print(f"Wrote summary to {output_root / 'baseline_summary.csv'}")
    print(f"Wrote aggregate summary to {output_root / 'baseline_summary_agg.csv'}")


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
                            "output_dir": str(output_root / str(dataset) / f"seed_{int(seed)}"),
                        },
                        "dataset": {"name": str(dataset)},
                    },
                )
                _validate_run_config(run["runner"], run_config)
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
    return plans


def _validate_run_config(runner: str, config: dict[str, Any]) -> None:
    if runner == "kmeans":
        validate_kmeans_config(config)
        return
    if runner == "dec_idec":
        validate_dec_idec_config(config)
        return
    raise ValueError(f"Unsupported baseline runner in YAML: {runner!r}")


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
    if name == "kmeans":
        from src.runners import run_kmeans

        return run_kmeans
    if name == "dec_idec":
        from src.runners import run_dec_idec

        return run_dec_idec
    raise ValueError(f"Unknown runner: {name}")


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
        key = tuple(row.get(field, "") for field in ["backbone", "gloca", "head", "dataset"])
        grouped.setdefault(key, []).append(row)

    aggregated = []
    for key, group_rows in sorted(grouped.items()):
        success_rows = [row for row in group_rows if row.get("status") in {"success", "skipped"}]
        out = {field: "" for field in AGG_FIELDS}
        out.update(dict(zip(["backbone", "gloca", "head", "dataset"], key)))
        out["n_runs"] = len(success_rows)
        out["seeds"] = " ".join(str(row["seed"]) for row in success_rows)
        for metric in ["ari", "nmi", "acc", "silhouette"]:
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
        return float(value)
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
