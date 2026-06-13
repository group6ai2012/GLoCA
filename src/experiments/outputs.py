from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch

from src.evaluation.assignment_schema import ordered_metrics_row, validate_assignment_payload
from src.experiments.config import save_experiment_config


def write_outputs(
    output_dir: Path,
    config: dict,
    assignments_payload: dict[str, Any],
    metrics_row: dict[str, Any],
    embeddings: torch.Tensor,
    attention: torch.Tensor | None,
    logs: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, output_dir / "config.yaml")
    validate_assignment_payload(assignments_payload)
    metrics_row = ordered_metrics_row(
        metrics_row,
        extra_fields=metrics_row.get("_extra_fields"),
    )
    metrics_row.pop("_extra_fields", None)
    with (output_dir / "assignments.json").open("w", encoding="utf-8") as handle:
        json.dump(assignments_payload, handle, indent=2)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_row.keys()))
        writer.writeheader()
        writer.writerow(metrics_row)
    torch.save(embeddings.cpu(), output_dir / "embeddings.pt")
    if attention is not None:
        torch.save(attention.cpu(), output_dir / "attention.pt")
    with (output_dir / "logs.json").open("w", encoding="utf-8") as handle:
        json.dump(logs, handle, indent=2)
