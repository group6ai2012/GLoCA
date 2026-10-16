from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from src.data.datamodule import ClusteringDataModule
from src.data.folder_dataset import DatasetSpec
from src.models.baselines.dec_idec import DINOCLSDECModel
from src.training.dec_idec_trainer import DECIDECTrainer, parse_target_update_interval


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=(100, 150, 200)).save(path)


def test_clustering_datamodule_is_plain_pytorch_helper(tmp_path: Path) -> None:
    _write_image(tmp_path / "a" / "0.jpg")
    _write_image(tmp_path / "b" / "0.jpg")
    spec = DatasetSpec(name="tiny", root=tmp_path)

    loaders = ClusteringDataModule(
        spec=spec,
        image_size=8,
        batch_size=2,
        num_workers=0,
        training_mode="contrastive_two_view",
        training_views="auto",
    )

    assert type(loaders).__bases__ == (object,)
    train_batch = next(iter(loaders.train_dataloader()))
    predict_batch = next(iter(loaders.predict_dataloader()))
    assert set(["index", "label", "label_name", "image_id", "dataset", "views"]).issubset(train_batch)
    assert set(["index", "label", "label_name", "image_id", "dataset", "image"]).issubset(predict_batch)
    assert len(train_batch["views"]) == 2


def test_parse_target_update_interval() -> None:
    assert parse_target_update_interval(None) == ("fixed", None)
    assert parse_target_update_interval("fixed") == ("fixed", None)
    assert parse_target_update_interval(0) == ("fixed", None)
    assert parse_target_update_interval("3") == ("interval", 3)


def test_dec_idec_trainer_runs_tiny_cached_training() -> None:
    torch.manual_seed(0)
    model = DINOCLSDECModel(
        n_clusters=2,
        input_dim=4,
        hidden_dims=(8,),
        latent_dim=2,
        alpha=1.0,
    )
    config = {
        "trainer": {"batch_size": 4, "lr": 1.0e-3},
        "baseline": {
            "kmeans_init": "kmeans++",
            "kmeans_n_init": 2,
            "kmeans_max_iter": 20,
            "kmeans_tol": 1.0e-4,
            "pretrain_epochs": 1,
            "refine_epochs": 1,
            "pretrain_lr": 1.0e-3,
            "refine_lr": 1.0e-3,
            "lambda_recon": 0.1,
            "target_update_interval": None,
        },
        "gloca": {"enabled": False},
    }
    trainer = DECIDECTrainer(
        model=model,
        x=torch.randn(8, 4),
        config=config,
        mode="dec",
        n_clusters=2,
        seed=0,
        device=torch.device("cpu"),
    )

    trainer.pretrain()
    trainer.initialize_cluster_centers()
    trainer.refine()
    latent, assignments = trainer.predict_all()

    assert len(trainer.pretrain_losses) == 1
    assert len(trainer.refine_losses) == 1
    assert latent.shape == (8, 2)
    assert assignments.shape == (8,)
