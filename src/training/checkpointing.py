from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import os
import random
import shutil
import time

import numpy as np
import torch


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def copy_as_latest(src: Path, latest_path: Path) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = latest_path.with_suffix(latest_path.suffix + ".tmp")
    shutil.copyfile(src, tmp_path)
    os.replace(tmp_path, latest_path)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    latest = Path(checkpoint_dir) / "latest.ckpt"
    return latest if latest.exists() else None


def resolve_resume_checkpoint(raw_value: Any, checkpoint_dir: Path) -> Path | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if value.lower() in {"", "none", "null", "false", "no"}:
        return None
    if value.lower() == "auto":
        return find_latest_checkpoint(checkpoint_dir)
    return Path(value)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(_rng_byte_tensor(state["torch"]))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [_rng_byte_tensor(cuda_state) for cuda_state in state["cuda"]]
        )


def _rng_byte_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype=torch.uint8).contiguous()
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()


def should_save_epoch_checkpoint(epoch: int, interval: int) -> bool:
    interval = int(interval)
    return interval > 0 and (int(epoch) + 1) % interval == 0


def should_run_eval(
    *,
    epoch: int,
    max_epochs: int,
    eval_interval: Any,
    checkpoint_interval: int,
) -> bool:
    is_final = (int(epoch) + 1) >= int(max_epochs)
    if eval_interval is None:
        return is_final

    value = str(eval_interval).strip().lower()
    if value in {"", "none", "null", "false", "no", "final_only"}:
        return is_final
    if value == "every_epoch":
        return True
    if value == "checkpoint":
        return is_final or should_save_epoch_checkpoint(epoch, checkpoint_interval)

    interval = int(value)
    if interval <= 0:
        return is_final
    return is_final or ((int(epoch) + 1) % interval == 0)


@contextmanager
def timed_section(target: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        target[key] = target.get(key, 0.0) + (time.perf_counter() - start)


def empty_resource_totals() -> dict[str, float]:
    return {
        "total_train_time_s": 0.0,
        "total_checkpoint_save_time_s": 0.0,
        "total_eval_time_s": 0.0,
        "total_epoch_wall_time_s": 0.0,
    }


def update_resource_totals(
    totals: dict[str, float], epoch_timing: dict[str, float]
) -> None:
    totals["total_train_time_s"] = totals.get("total_train_time_s", 0.0) + float(
        epoch_timing.get("train_epoch_time_s", 0.0)
    )
    totals["total_checkpoint_save_time_s"] = totals.get(
        "total_checkpoint_save_time_s", 0.0
    ) + float(epoch_timing.get("checkpoint_save_time_s", 0.0))
    totals["total_eval_time_s"] = totals.get("total_eval_time_s", 0.0) + float(
        epoch_timing.get("eval_time_s", 0.0)
    )
    totals["total_epoch_wall_time_s"] = totals.get(
        "total_epoch_wall_time_s", 0.0
    ) + float(epoch_timing.get("epoch_total_wall_time_s", 0.0))
