from __future__ import annotations

from pathlib import Path

import torch

from src.training.checkpointing import (
    atomic_torch_save,
    copy_as_latest,
    find_latest_checkpoint,
    restore_rng_state,
    resolve_resume_checkpoint,
    should_run_eval,
)
from src.experiments.outputs import append_csv_row


def test_atomic_torch_save_writes_loadable_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "epoch_0001.ckpt"
    atomic_torch_save({"value": torch.tensor([1, 2, 3])}, path)

    loaded = torch.load(path, weights_only=False)

    assert loaded["value"].tolist() == [1, 2, 3]


def test_latest_checkpoint_is_real_file_not_symlink(tmp_path: Path) -> None:
    src = tmp_path / "epoch_0001.ckpt"
    latest = tmp_path / "latest.ckpt"
    atomic_torch_save({"epoch": 1}, src)
    copy_as_latest(src, latest)

    assert latest.exists()
    assert latest.is_file()
    assert not latest.is_symlink()
    assert torch.load(latest, weights_only=False)["epoch"] == 1


def test_append_csv_row_appends_without_rewriting(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint_metrics.csv"
    append_csv_row(path, {"epoch": 1, "nmi": 0.1})
    append_csv_row(path, {"epoch": 2, "nmi": 0.2})

    assert path.read_text(encoding="utf-8").splitlines() == [
        "epoch,nmi",
        "1,0.1",
        "2,0.2",
    ]


def test_find_and_resolve_latest_checkpoint(tmp_path: Path) -> None:
    latest = tmp_path / "latest.ckpt"
    atomic_torch_save({"epoch": 2}, latest)

    assert find_latest_checkpoint(tmp_path) == latest
    assert resolve_resume_checkpoint("auto", tmp_path) == latest
    assert resolve_resume_checkpoint(None, tmp_path) is None
    assert resolve_resume_checkpoint("null", tmp_path) is None
    atomic_torch_save({"epoch": 1}, tmp_path / "other.ckpt")
    assert resolve_resume_checkpoint(tmp_path / "other.ckpt", tmp_path) == tmp_path / "other.ckpt"


def test_should_run_eval_schedules() -> None:
    assert should_run_eval(epoch=9, max_epochs=30, eval_interval="checkpoint", checkpoint_interval=10)
    assert not should_run_eval(epoch=8, max_epochs=30, eval_interval="checkpoint", checkpoint_interval=10)
    assert should_run_eval(epoch=29, max_epochs=30, eval_interval="checkpoint", checkpoint_interval=10)
    assert should_run_eval(epoch=0, max_epochs=3, eval_interval="every_epoch", checkpoint_interval=10)
    assert not should_run_eval(epoch=0, max_epochs=3, eval_interval="final_only", checkpoint_interval=10)
    assert should_run_eval(epoch=2, max_epochs=3, eval_interval="final_only", checkpoint_interval=10)
    assert should_run_eval(epoch=4, max_epochs=12, eval_interval=5, checkpoint_interval=10)
    assert not should_run_eval(epoch=3, max_epochs=12, eval_interval=5, checkpoint_interval=10)


def test_restore_rng_state_accepts_non_cpu_byte_tensor_forms() -> None:
    original = torch.get_rng_state()
    torch.manual_seed(123)

    restore_rng_state({"torch": original.tolist()})

    assert torch.equal(torch.get_rng_state(), original)
