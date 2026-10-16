from __future__ import annotations

from pathlib import Path

import torch

from src.models.baselines.dec_idec import DINOCLSDECModel
from src.training.dec_idec_trainer import DECIDECTrainer


def _dec_config(pretrain_epochs: int = 2, refine_epochs: int = 2) -> dict:
    return {
        "trainer": {
            "batch_size": 4,
            "lr": 1.0e-3,
            "checkpoint_interval": 1,
            "eval_interval": "checkpoint",
            "profile_resources": True,
        },
        "baseline": {
            "kmeans_init": "kmeans++",
            "kmeans_n_init": 1,
            "kmeans_max_iter": 5,
            "kmeans_tol": 1.0e-4,
            "pretrain_epochs": pretrain_epochs,
            "refine_epochs": refine_epochs,
            "pretrain_lr": 1.0e-3,
            "refine_lr": 1.0e-3,
            "lambda_recon": 0.1,
            "target_update_interval": None,
        },
        "gloca": {"enabled": False},
    }


def _make_dec_trainer(
    tmp_path: Path,
    *,
    resume_from_checkpoint: Path | None = None,
    pretrain_epochs: int = 2,
    refine_epochs: int = 2,
) -> DECIDECTrainer:
    model = DINOCLSDECModel(
        n_clusters=2,
        input_dim=4,
        hidden_dims=(8,),
        latent_dim=2,
        alpha=1.0,
    )
    return DECIDECTrainer(
        model=model,
        x=torch.randn(8, 4),
        config=_dec_config(pretrain_epochs=pretrain_epochs, refine_epochs=refine_epochs),
        mode="dec",
        n_clusters=2,
        seed=0,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def test_dec_resume_from_pretrain_checkpoint_continues_pretrain(tmp_path: Path) -> None:
    source = _make_dec_trainer(tmp_path)
    source.pretrain(start_epoch=0)
    latest = tmp_path / "latest.ckpt"

    resumed = _make_dec_trainer(tmp_path, resume_from_checkpoint=latest)
    resumed.fit()

    assert resumed.pretrain_start_epoch == 2
    assert len(resumed.pretrain_losses) == 2
    assert resumed.cluster_centers_initialized
    assert len(resumed.refine_losses) == 2


def test_dec_resume_from_refine_skips_pretrain_and_cluster_init(tmp_path: Path) -> None:
    source = _make_dec_trainer(tmp_path, refine_epochs=2)
    source.pretrain(start_epoch=0)
    source.initialize_cluster_centers()
    source.refresh_target_distribution()
    source.refine(start_epoch=0)
    latest = tmp_path / "latest.ckpt"

    resumed = _make_dec_trainer(tmp_path, resume_from_checkpoint=latest, refine_epochs=2)

    def fail_pretrain(*args, **kwargs):
        raise AssertionError("pretrain should be skipped when resuming refine")

    def fail_initialize(*args, **kwargs):
        raise AssertionError("cluster initialization should be skipped when restored")

    resumed.pretrain = fail_pretrain
    resumed.initialize_cluster_centers = fail_initialize
    resumed.fit()

    assert resumed.refine_start_epoch == 2
    assert resumed.cluster_centers_initialized


def test_dec_target_distribution_and_optimizers_restore(tmp_path: Path) -> None:
    source = _make_dec_trainer(tmp_path)
    source.pretrain(start_epoch=0)
    source.initialize_cluster_centers()
    source.refresh_target_distribution()
    source.refine(start_epoch=0)
    latest = tmp_path / "latest.ckpt"

    resumed = _make_dec_trainer(tmp_path, resume_from_checkpoint=latest)

    assert resumed.target_distribution is not None
    assert torch.allclose(
        resumed.target_distribution.cpu(), source.target_distribution.cpu()
    )
    assert resumed.pretrain_optimizer.state_dict()["state"]
    assert resumed.refine_optimizer.state_dict()["state"]
    assert resumed.resource_totals["total_train_time_s"] > 0.0


def test_dec_complete_checkpoint_causes_training_skip(tmp_path: Path) -> None:
    source = _make_dec_trainer(tmp_path, pretrain_epochs=1, refine_epochs=1)
    source.fit()
    resumed = _make_dec_trainer(
        tmp_path,
        resume_from_checkpoint=tmp_path / "latest.ckpt",
        pretrain_epochs=1,
        refine_epochs=1,
    )

    def fail_training(*args, **kwargs):
        raise AssertionError("complete checkpoint should skip training")

    resumed.pretrain = fail_training
    resumed.refine = fail_training
    resumed.fit()

    assert resumed.resume_phase == "complete"
