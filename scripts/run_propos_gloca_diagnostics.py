from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.config import (
    load_experiment_config,
    validate_propos_config,
    validate_propos_diagnostics_config,
)
from src.runners import run_propos


SUMMARY_FIELDS = [
    "run_id",
    "status",
    "error",
    "experiment",
    "head",
    "dataset",
    "seed",
    "gloca",
    "freeze_gloca",
    "freeze_gloca_epochs",
    "gloca_lr_multiplier",
    "gloca_alpha_lr_multiplier",
    "max_epochs",
    "warmup_epochs",
    "kmeans_backend",
    "kmeans_init",
    "ari",
    "nmi",
    "acc",
    "silhouette",
    "n_nonempty_clusters",
    "cluster_size_entropy",
    "loss_psa_final",
    "loss_psl_final",
    "loss_total_final",
    "gloca_alpha_initial",
    "gloca_alpha_final",
    "gloca_alpha_value_last",
    "gloca_alpha_grad_norm_last",
    "gloca_param_grad_norm_last",
    "gloca_delta_norm_mean_last",
    "gloca_delta_norm_std_last",
    "gloca_embedding_cls_cosine_mean_last",
    "gloca_embedding_cls_cosine_std_last",
    "gloca_attention_entropy_mean_last",
    "gloca_attention_max_mean_last",
    "total_time_s",
    "peak_gpu_mb",
    "elapsed_wall_time_s",
    "output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a YAML-defined ProPos/GLoCA diagnostic matrix.")
    parser.add_argument("config_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config_path)
    validate_propos_diagnostics_config(config)
    plans = _build_plans(config)
    report_root = _report_root(config)
    report_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, plan in enumerate(plans, start=1):
        print(f"Running {index}/{len(plans)}: {plan['run_id']} - {plan['name']}", flush=True)
        row = run_or_collect(plan, force=bool(config["diagnostics"]["force"]))
        rows.append(row)
        print_run_summary(row)
    csv_path = report_root / "propos_gloca_diagnostic_summary.csv"
    md_path = report_root / "propos_gloca_diagnostic_report.md"
    write_summary_csv(csv_path, rows)
    write_markdown_report(md_path, rows)
    print(f"Wrote summary CSV: {csv_path}", flush=True)
    print(f"Wrote Markdown report: {md_path}", flush=True)


def _build_plans(config: dict[str, Any]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    report_root = _report_root(config)
    for run in config["diagnostics"]["runs"]:
        run_config = deepcopy(config["base_config"])
        _deep_update(run_config, deepcopy(run["config"]))
        _deep_update(
            run_config,
            {
                "experiment": {
                    "name": run["config"]["experiment"]["name"],
                    "output_dir": str(report_root),
                }
            },
        )
        validate_propos_config(run_config)
        plans.append(
            {
                "run_id": str(run["id"]),
                "name": str(run["name"]),
                "config": run_config,
                "output_dir": expected_output_dir(run_config),
            }
        )
    return plans


def _report_root(config: dict[str, Any]) -> Path:
    base_config = config["base_config"]
    dataset = str(base_config["dataset"]["name"])
    seed = int(base_config["experiment"]["seed"])
    return Path(config["experiment"]["output_dir"]) / dataset / f"seed_{seed}"


def run_or_collect(plan: dict[str, Any], *, force: bool) -> dict[str, Any]:
    output_dir = plan["output_dir"]
    metrics_path = output_dir / "metrics.csv"
    logs_path = output_dir / "logs.json"
    start = time.perf_counter()
    status = "completed"
    error = ""
    try:
        if metrics_path.exists() and logs_path.exists() and not force:
            status = "skipped"
        else:
            result = run_propos(plan["config"])
            output_dir = Path(result.output_dir)
            metrics_path = output_dir / "metrics.csv"
            logs_path = output_dir / "logs.json"
        row = collect_row(plan["run_id"], status, error, output_dir, metrics_path, logs_path)
    except Exception as exc:  # noqa: BLE001 - report should survive a failed run.
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        row = empty_row(plan["run_id"], status, error, output_dir)
    row["elapsed_wall_time_s"] = f"{time.perf_counter() - start:.3f}"
    return row


def expected_output_dir(config: dict[str, Any]) -> Path:
    return (
        Path(config["experiment"]["output_dir"])
        / config["experiment"]["name"]
        / config["head"]["name"]
        / config["dataset"]["name"]
        / f"seed_{int(config['experiment']['seed'])}"
    )


def collect_row(
    run_id: str,
    status: str,
    error: str,
    output_dir: Path,
    metrics_path: Path,
    logs_path: Path,
) -> dict[str, Any]:
    row = empty_row(run_id, status, error, output_dir)
    metrics = read_csv_row(metrics_path)
    logs = read_json(logs_path)
    row.update({field: metrics.get(field, row.get(field, "")) for field in SUMMARY_FIELDS if field in metrics})
    row.update(
        {
            "experiment": metrics.get("experiment", ""),
            "head": metrics.get("head", ""),
            "dataset": metrics.get("dataset", ""),
            "seed": metrics.get("seed", ""),
            "gloca": metrics.get("gloca", ""),
            "freeze_gloca": logs.get("freeze_gloca", ""),
            "freeze_gloca_epochs": logs.get("freeze_gloca_epochs", ""),
            "gloca_lr_multiplier": logs.get("gloca_lr_multiplier", ""),
            "gloca_alpha_lr_multiplier": logs.get("gloca_alpha_lr_multiplier", ""),
            "max_epochs": len(logs.get("epoch_history", [])) if logs.get("epoch_history") else "",
            "warmup_epochs": metrics.get("warmup_epochs", ""),
            "kmeans_backend": metrics.get("kmeans_backend", logs.get("kmeans_backend", "")),
            "kmeans_init": metrics.get("kmeans_init", logs.get("kmeans_init", "")),
            "loss_psa_final": metrics.get("loss_psa_final", logs.get("loss_psa_final", "")),
            "loss_psl_final": metrics.get("loss_psl_final", logs.get("loss_psl_final", "")),
            "loss_total_final": metrics.get("loss_total_final", logs.get("loss_total_final", "")),
            "gloca_alpha_initial": logs.get("gloca_alpha_initial", ""),
            "gloca_alpha_final": logs.get("gloca_alpha_final", ""),
            "output_dir": str(output_dir),
        }
    )
    for key, value in last_gloca_diagnostic(logs).items():
        row[f"{key}_last"] = value
    return normalize_row(row)


def empty_row(run_id: str, status: str, error: str, output_dir: Path) -> dict[str, Any]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update({"run_id": run_id, "status": status, "error": error, "output_dir": str(output_dir)})
    return row


def read_csv_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def last_gloca_diagnostic(logs: dict[str, Any]) -> dict[str, Any]:
    history = logs.get("gloca_diagnostics_history") or []
    if not history:
        return {}
    latest = history[-1]
    return {
        "gloca_alpha_value": latest.get("gloca_alpha_value", ""),
        "gloca_alpha_grad_norm": latest.get("gloca_alpha_grad_norm", ""),
        "gloca_param_grad_norm": latest.get("gloca_param_grad_norm", ""),
        "gloca_delta_norm_mean": latest.get("gloca_delta_norm_mean", ""),
        "gloca_delta_norm_std": latest.get("gloca_delta_norm_std", ""),
        "gloca_embedding_cls_cosine_mean": latest.get("gloca_embedding_cls_cosine_mean", ""),
        "gloca_embedding_cls_cosine_std": latest.get("gloca_embedding_cls_cosine_std", ""),
        "gloca_attention_entropy_mean": latest.get("gloca_attention_entropy_mean", ""),
        "gloca_attention_max_mean": latest.get("gloca_attention_max_mean", ""),
    }


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in SUMMARY_FIELDS}


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ProPos GLoCA Diagnostic Report",
        "",
        "## Final Metrics",
        "",
        markdown_table(rows, ["run_id", "status", "experiment", "gloca", "ari", "nmi", "acc", "silhouette"]),
        "",
        "## GLoCA Diagnostics",
        "",
        markdown_table(
            [row for row in rows if row.get("gloca") and row.get("gloca") != "disabled"],
            [
                "run_id",
                "freeze_gloca",
                "gloca_lr_multiplier",
                "gloca_alpha_lr_multiplier",
                "gloca_alpha_final",
                "gloca_alpha_grad_norm_last",
                "gloca_param_grad_norm_last",
                "gloca_embedding_cls_cosine_mean_last",
                "gloca_attention_entropy_mean_last",
                "gloca_attention_max_mean_last",
            ],
        ),
        "",
        "## Interpretation",
        "",
        *[f"- {line}" for line in interpret(rows)],
        "",
        "## Output Directories",
        "",
        markdown_table(rows, ["run_id", "status", "output_dir", "error"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = ["| " + " | ".join(format_cell(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text.replace("|", "\\|")
    if number != number:
        return "nan"
    return f"{number:.4f}"


def interpret(rows: list[dict[str, Any]]) -> list[str]:
    by_id = {row["run_id"]: row for row in rows if row.get("status") in {"completed", "skipped"}}
    notes = []
    a = by_id.get("A")
    b = by_id.get("B")
    c = by_id.get("C")
    d = by_id.get("D")
    e = by_id.get("E")
    if a and b:
        if is_much_worse(b, a):
            notes.append("Frozen GLoCA is much worse than CLS; inspect adapter path or normalization.")
        else:
            notes.append("Frozen GLoCA path is consistent with CLS; conservative initialization is likely working.")
    if b and c:
        alpha_c = abs(to_float(c.get("gloca_alpha_final")))
        if not is_meaningfully_better(c, b) and alpha_c < 0.01:
            notes.append("Trainable GLoCA with alpha LR 1.0 is barely affecting the embedding.")
    if c and d and is_meaningfully_better(d, c):
        notes.append("Higher alpha LR helps; alpha movement was likely a bottleneck.")
    if d and e:
        cosine_e = to_float(e.get("gloca_embedding_cls_cosine_mean_last"))
        if is_much_worse(e, d) and cosine_e < 0.995:
            notes.append("Full GLoCA body LR may be too aggressive.")
    if a and any(is_meaningfully_better(row, a) for run_id, row in by_id.items() if run_id in {"C", "D", "E"}):
        notes.append("A trainable GLoCA run beats CLS by the simple threshold; the correction is promising here.")
    if not notes:
        notes.append("No strong automatic conclusion from the configured thresholds.")
    notes.append("Thresholds: >=0.02 ARI/ACC or >=0.01 NMI is treated as meaningful; CLS cosine below 0.995 indicates noticeable movement.")
    return notes


def is_meaningfully_better(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        to_float(candidate.get("ari")) - to_float(baseline.get("ari")) >= 0.02
        or to_float(candidate.get("acc")) - to_float(baseline.get("acc")) >= 0.02
        or to_float(candidate.get("nmi")) - to_float(baseline.get("nmi")) >= 0.01
    )


def is_much_worse(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        to_float(baseline.get("ari")) - to_float(candidate.get("ari")) >= 0.02
        or to_float(baseline.get("acc")) - to_float(candidate.get("acc")) >= 0.02
        or to_float(baseline.get("nmi")) - to_float(candidate.get("nmi")) >= 0.01
    )


def to_float(value: Any) -> float:
    try:
        if value in {"", None}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def print_run_summary(row: dict[str, Any]) -> None:
    print(f"  output_dir: {row.get('output_dir', '')}", flush=True)
    print(
        "  final: "
        f"status={row.get('status')} "
        f"ari={format_cell(row.get('ari', ''))} "
        f"nmi={format_cell(row.get('nmi', ''))} "
        f"acc={format_cell(row.get('acc', ''))} "
        f"alpha_final={format_cell(row.get('gloca_alpha_final', ''))} "
        f"cls_cosine_last={format_cell(row.get('gloca_embedding_cls_cosine_mean_last', ''))} "
        f"elapsed={row.get('elapsed_wall_time_s', '')}s",
        flush=True,
    )
    if row.get("error"):
        print(f"  error: {row['error']}", flush=True)


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


if __name__ == "__main__":
    main()
