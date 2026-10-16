from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn

import src.training.cdc_trainer as cdc_trainer_module
from src.data import ClusteringDataModule
from src.data.folder_dataset import DatasetSpec
from src.experiments.config import load_experiment_config, validate_cdc_config
from src.models import ClusteringBaseModel
from src.models.clustering import CDCHead
from src.models.clustering.cdc import orthogonalize_prototype_rows
from src.training.cdc_utils import (
    calibration_target_loss,
    select_reliable_samples,
    split_sub_batch_indices,
)
from src.training.cdc_trainer import CDCTrainer
from src.training.checkpointing import resolve_resume_checkpoint


class _DummyBackbone(nn.Module):
    output_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.seen_means: list[float] = []
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, image: torch.Tensor) -> dict:
        self.seen_means.append(float(image.detach().mean().cpu()))
        cls = self.proj(image.float())
        return {"cls": cls, "patch_tokens": torch.empty(image.shape[0], 0, 4), "patch_grid": (0, 0)}


class _TinyCDCDataModule:
    def __init__(self) -> None:
        self.batch = {
            "weak": torch.zeros(4, 4),
            "strong": torch.ones(4, 4),
            "calibration": torch.full((4, 4), 2.0),
            "views": (torch.zeros(4, 4), torch.ones(4, 4)),
            "label": torch.tensor([0, 1, 0, 1]),
            "index": torch.tensor([0, 1, 2, 3]),
            "image_id": ["a", "b", "c", "d"],
            "dataset": ["tiny"] * 4,
        }

    def train_dataloader(self):
        return [self.batch]

    def predict_dataloader(self):
        return [
            {
                "image": torch.randn(4, 4),
                "label": torch.tensor([0, 1, 0, 1]),
                "index": torch.tensor([0, 1, 2, 3]),
                "image_id": ["a", "b", "c", "d"],
            }
        ]

    def train_eval_dataloader(self):
        return self.predict_dataloader()


def _cdc_config(overrides: dict | None = None) -> dict:
    cdc = {
        "embedding_dim": 4,
        "hidden_dim": 8,
        "training_views": "cdc_weak_strong_calibration",
        "meta_batch_size": 4,
        "sub_batch_size": 2,
        "per_class_selected_num": "auto",
        "calibration_k": 2,
        "meta_batch_drop_last": False,
        "w_en": 1.0,
        "prototype_init": True,
        "orthogonalize_init": False,
        "orthogonalize_epochs": 2,
        "orthogonalize_scale": 5.0,
        "kmeans_init": "kmeans++",
        "kmeans_n_init": 1,
        "kmeans_max_iter": 5,
        "kmeans_tol": 1.0e-4,
        "weight_decay": 0.0,
        "optimizer": "adamw",
        "gloca_lr_multiplier": 1.0,
        "gloca_alpha_lr_multiplier": 1.0,
        "freeze_gloca": False,
        "freeze_gloca_epochs": 0,
        "log_gloca_diagnostics": False,
        "progress": False,
        "augment": {},
        **(overrides or {}),
    }
    return {
        "experiment": {"seed": 3},
        "trainer": {"max_epochs": 1, "batch_size": 4, "lr": 1.0e-3},
        "cdc": cdc,
    }


def test_cdc_head_forward_shapes_and_finite_outputs() -> None:
    head = CDCHead(input_dim=16, hidden_dim=32, n_clusters=5)
    out = head(torch.randn(7, 16))

    assert out["clustering_logits"].shape == (7, 5)
    assert out["calibration_logits"].shape == (7, 5)
    assert out["predictions"].shape == (7,)
    assert out["confidences"].shape == (7,)
    assert all(torch.isfinite(value).all() for value in out.values() if torch.is_tensor(value))


def test_reliable_sample_selection_handles_no_selected_samples() -> None:
    probs = torch.full((4, 5), 0.2)
    selection = select_reliable_samples(probs, per_class_selected_num=1)

    assert selection["selected_count"] == 0
    assert not bool(selection["selected_mask"].any())


def test_cdc_chunking_handles_tail_sub_batch() -> None:
    chunks = split_sub_batch_indices(10, 4, device=torch.device("cpu"))

    assert [int(chunk.numel()) for chunk in chunks] == [4, 4, 2]


def test_reliable_sample_auto_uses_full_batch_size_not_sub_batch_size() -> None:
    probs = torch.zeros(10, 5)
    probs[:, 0] = 0.9
    probs[:, 1:] = 0.025
    selection = select_reliable_samples(probs, per_class_selected_num=None)

    assert selection["per_class_selected_num"] == 2


def test_reliable_sample_auto_uses_virtual_meta_batch_for_plantwild_scale() -> None:
    probs = torch.zeros(1024, 115)
    probs[:, 0] = 0.9
    probs[:, 1:] = 0.1 / 114
    selection = select_reliable_samples(probs, per_class_selected_num=None)

    assert selection["per_class_selected_num"] == 8


def test_calibration_loss_does_not_backprop_into_embeddings_or_clustering_head() -> None:
    head = CDCHead(input_dim=6, hidden_dim=8, n_clusters=3)
    embeddings = torch.randn(5, 6, requires_grad=True)
    with torch.no_grad():
        target = torch.softmax(head.clustering_head(embeddings.detach()), dim=1)
    logits = head.calibration_head(embeddings.detach())
    loss, _, _ = calibration_target_loss(logits, target, entropy_weight=1.0)
    loss.backward()

    assert embeddings.grad is None
    assert all(parameter.grad is None for parameter in head.clustering_head.parameters())
    assert any(parameter.grad is not None for parameter in head.calibration_head.parameters())


def test_cdc_initialization_logs_clean_fallback_when_too_few_samples() -> None:
    head = CDCHead(input_dim=4, hidden_dim=8, n_clusters=2)
    logs = head.initialize_from_embeddings(
        torch.randn(4, 4),
        kmeans_init="kmeans++",
        kmeans_n_init=1,
        kmeans_max_iter=5,
        kmeans_tol=1.0e-4,
        seed=0,
    )

    assert logs["cdc_init_mode"] == "random"
    assert not logs["prototype_init_used"]
    assert "n_samples" in logs["fallback_reason"]


def test_cdc_orthogonalize_prototype_rows_returns_finite_weights() -> None:
    torch.manual_seed(0)
    prototypes = torch.randn(4, 3)
    updated, logs = orthogonalize_prototype_rows(
        prototypes, epochs=2, scale=5.0, use_relu=True
    )

    assert updated.shape == prototypes.shape
    assert torch.isfinite(updated).all()
    assert logs["epochs"] == 2
    assert logs["use_relu"] is True
    assert logs["initial_loss"] is not None
    assert logs["final_loss"] is not None


def test_cdc_initialization_applies_orthogonalized_prototypes() -> None:
    torch.manual_seed(0)
    head = CDCHead(input_dim=4, hidden_dim=4, n_clusters=2)
    logs = head.initialize_from_embeddings(
        torch.randn(12, 4),
        kmeans_init="kmeans++",
        kmeans_n_init=1,
        kmeans_max_iter=5,
        kmeans_tol=1.0e-4,
        seed=0,
        orthogonalize=True,
        orthogonalize_epochs=2,
        orthogonalize_scale=5.0,
    )

    assert logs["cdc_init_mode"] == "prototype_kmeans_orthogonalized"
    assert logs["prototype_init_used"] is True
    assert logs["orthogonalization_used"] is True
    assert logs["first_orthogonalization_logs"]["epochs"] == 2
    assert logs["second_orthogonalization_logs"]["epochs"] == 2
    assert torch.allclose(
        head.clustering_head.linear1.weight, head.calibration_head.linear1.weight
    )
    assert torch.allclose(
        head.clustering_head.linear2.weight, head.calibration_head.linear2.weight
    )


def test_cdc_dataloader_returns_named_views_and_metadata(tmp_path: Path) -> None:
    root = tmp_path / "tiny"
    for class_name in ["a", "b"]:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for idx in range(2):
            Image.new("RGB", (24, 24), color=(idx * 50, 20, 100)).save(class_dir / f"{idx}.png")
    datamodule = ClusteringDataModule(
        spec=DatasetSpec(name="tiny", root=root),
        image_size=16,
        batch_size=2,
        num_workers=0,
        training_mode="cdc",
        training_views="cdc_weak_strong_calibration",
        cdc_augment_config={"strong": {"randaugment_n": 1, "cutout_enabled": False}},
    )
    batch = next(iter(datamodule.train_dataloader()))
    sample = datamodule.train_dataset[0]
    sample_again = datamodule.train_dataset[0]

    assert {"weak", "strong", "calibration", "views", "label", "label_name", "image_id", "index", "dataset"}.issubset(sample)
    assert batch["weak"].shape == batch["strong"].shape == batch["calibration"].shape
    assert torch.allclose(sample["calibration"], sample_again["calibration"])


def test_cdc_trainer_consumes_weak_strong_and_calibration_views() -> None:
    backbone = _DummyBackbone()
    model = ClusteringBaseModel(
        backbone=backbone,
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    trainer = CDCTrainer(
        model=model,
        datamodule=_TinyCDCDataModule(),
        config=_cdc_config({"prototype_init": False}),
        device=torch.device("cpu"),
    )
    loss, logs = trainer.train_step(_TinyCDCDataModule().batch, epoch=0)

    assert torch.isfinite(loss)
    assert logs["batch_size"] == 4
    assert logs["cdc_num_sub_batches"] == 2
    assert logs["actual_meta_batch_size"] == 4
    assert {0.0, 1.0, 2.0}.issubset({round(value, 1) for value in backbone.seen_means})


def test_cdc_trainer_keeps_backbone_out_of_optimizer_groups() -> None:
    backbone = _DummyBackbone()
    model = ClusteringBaseModel(
        backbone=backbone,
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    trainer = CDCTrainer(model=model, datamodule=_TinyCDCDataModule(), config=_cdc_config(), device=torch.device("cpu"))
    optimizer_param_ids = {
        id(parameter)
        for optimizer in [trainer.optimizer_cluster, trainer.optimizer_calibration]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    backbone_param_ids = {id(parameter) for parameter in backbone.parameters()}

    assert optimizer_param_ids.isdisjoint(backbone_param_ids)
    assert not any(parameter.requires_grad for parameter in backbone.parameters())


def test_cdc_optimizers_have_disjoint_ownership() -> None:
    model = ClusteringBaseModel(
        backbone=_DummyBackbone(),
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    trainer = CDCTrainer(model=model, datamodule=_TinyCDCDataModule(), config=_cdc_config(), device=torch.device("cpu"))
    cluster_ids = {id(parameter) for group in trainer.optimizer_cluster.param_groups for parameter in group["params"]}
    calibration_ids = {
        id(parameter)
        for group in trainer.optimizer_calibration.param_groups
        for parameter in group["params"]
    }
    clustering_head_ids = {id(parameter) for parameter in model.head.clustering_head.parameters()}
    calibration_head_ids = {id(parameter) for parameter in model.head.calibration_head.parameters()}

    assert cluster_ids.isdisjoint(calibration_ids)
    assert clustering_head_ids.issubset(cluster_ids)
    assert calibration_head_ids.issubset(calibration_ids)
    assert cluster_ids.isdisjoint(calibration_head_ids)
    assert calibration_ids.isdisjoint(clustering_head_ids)


def test_cdc_calibration_targets_are_built_once_per_full_batch() -> None:
    model = ClusteringBaseModel(
        backbone=_DummyBackbone(),
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    trainer = CDCTrainer(
        model=model,
        datamodule=_TinyCDCDataModule(),
        config=_cdc_config({"prototype_init": False, "sub_batch_size": 1}),
        device=torch.device("cpu"),
    )
    _loss, logs = trainer.train_step(_TinyCDCDataModule().batch, epoch=0)

    assert trainer.calibration_target_calls == 1
    assert logs["cdc_num_sub_batches"] == 4


def test_cdc_skips_empty_selected_chunk_but_still_updates_calibration() -> None:
    model = ClusteringBaseModel(
        backbone=_DummyBackbone(),
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=4),
    )
    trainer = CDCTrainer(
        model=model,
        datamodule=_TinyCDCDataModule(),
        config=_cdc_config({"prototype_init": False, "sub_batch_size": 3, "calibration_k": 2}),
        device=torch.device("cpu"),
    )
    _loss, logs = trainer.train_step(_TinyCDCDataModule().batch, epoch=0)

    assert logs["cdc_num_sub_batches"] == 2
    assert logs["skipped_clustering_sub_batches"] >= 1
    assert torch.isfinite(torch.tensor(logs["loss_cali_mean"]))


def test_cdc_tail_sub_batch_runs_training_step() -> None:
    model = ClusteringBaseModel(
        backbone=_DummyBackbone(),
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    datamodule = _TinyCDCDataModule()
    batch = dict(datamodule.batch)
    for key in ["weak", "strong", "calibration"]:
        batch[key] = torch.cat([batch[key], batch[key][:1]], dim=0)
    batch["views"] = (batch["weak"], batch["strong"])
    batch["label"] = torch.tensor([0, 1, 0, 1, 0])
    batch["index"] = torch.arange(5)
    trainer = CDCTrainer(
        model=model,
        datamodule=datamodule,
        config=_cdc_config({"prototype_init": False, "sub_batch_size": 2, "meta_batch_size": 5}),
        device=torch.device("cpu"),
    )
    _loss, logs = trainer.train_step(batch, epoch=0)

    assert logs["batch_size"] == 5
    assert logs["cdc_num_sub_batches"] == 3


def test_cdc_meta_batch_assembly_splits_physical_batches_sample_accurately() -> None:
    model = ClusteringBaseModel(
        backbone=_DummyBackbone(),
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    trainer = CDCTrainer(
        model=model,
        datamodule=_TinyCDCDataModule(),
        config=_cdc_config({"prototype_init": False, "meta_batch_size": 10}),
        device=torch.device("cpu"),
    )
    current = cdc_trainer_module.CDCMetaBatch()
    ready_all = []
    for offset in [0, 4, 8]:
        batch = dict(_TinyCDCDataModule().batch)
        batch["index"] = torch.arange(offset, offset + 4)
        ready, current = trainer._add_physical_batch_to_meta_batches(current, batch)
        ready_all.extend(ready)

    assert [meta.num_samples for meta in ready_all] == [10]
    assert current.num_samples == 2
    meta = ready_all[0].finalize()
    assert meta["strong"].shape[0] == meta["indices"].shape[0] == 10
    assert meta["strong"][0].shape == _TinyCDCDataModule().batch["strong"][0].shape


def test_cdc_train_epoch_processes_or_drops_partial_meta_batch() -> None:
    class _ThreeBatchDataModule(_TinyCDCDataModule):
        def __init__(self) -> None:
            super().__init__()
            self.batches = []
            for offset in [0, 4, 8]:
                batch = dict(self.batch)
                batch["index"] = torch.arange(offset, offset + 4)
                self.batches.append(batch)

        def train_dataloader(self):
            return self.batches

    model_process = ClusteringBaseModel(
        backbone=_DummyBackbone(),
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    trainer_process = CDCTrainer(
        model=model_process,
        datamodule=_ThreeBatchDataModule(),
        config=_cdc_config({"prototype_init": False, "meta_batch_size": 10}),
        device=torch.device("cpu"),
    )
    logs_process = trainer_process.train_epoch(epoch=0)

    model_drop = ClusteringBaseModel(
        backbone=_DummyBackbone(),
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    trainer_drop = CDCTrainer(
        model=model_drop,
        datamodule=_ThreeBatchDataModule(),
        config=_cdc_config({"prototype_init": False, "meta_batch_size": 10, "meta_batch_drop_last": True}),
        device=torch.device("cpu"),
    )
    logs_drop = trainer_drop.train_epoch(epoch=0)

    assert logs_process["cdc_num_meta_batches"] == 2
    assert logs_process["cdc_meta_batch_size_actual_min"] == 2
    assert logs_drop["cdc_num_meta_batches"] == 1
    assert logs_drop["skipped_meta_batches"] == 1


def test_cdc_calibration_kmeans_uses_meta_batch_not_physical_batch(monkeypatch) -> None:
    shapes = []

    def fake_fit_kmeans(features, n_clusters, **kwargs):
        shapes.append(tuple(features.shape))
        assignments = torch.arange(features.shape[0]) % int(n_clusters)
        return {"assignments": assignments.long(), "centers": torch.randn(n_clusters, features.shape[1]), "logs": {}}

    monkeypatch.setattr(cdc_trainer_module, "fit_kmeans", fake_fit_kmeans)
    model = ClusteringBaseModel(
        backbone=_DummyBackbone(),
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    trainer = CDCTrainer(
        model=model,
        datamodule=_TinyCDCDataModule(),
        config=_cdc_config({"prototype_init": False, "meta_batch_size": 10, "calibration_k": 2}),
        device=torch.device("cpu"),
    )
    current = cdc_trainer_module.CDCMetaBatch()
    ready_all = []
    for offset in [0, 4, 8]:
        batch = dict(_TinyCDCDataModule().batch)
        batch["index"] = torch.arange(offset, offset + 4)
        ready, current = trainer._add_physical_batch_to_meta_batches(current, batch)
        ready_all.extend(ready)
    trainer._process_meta_batch_or_skip(ready_all[0], epoch=0)

    assert shapes == [(10, 4)]


def test_cdc_process_uses_stored_strong_without_dataset_replay() -> None:
    class _ForbiddenDataset:
        calls = 0

        def __getitem__(self, index: int):
            self.calls += 1
            raise AssertionError(f"CDC must not replay dataset index {index}")

    class _NoReplayDataModule(_TinyCDCDataModule):
        def __init__(self) -> None:
            super().__init__()
            self.train_dataset = _ForbiddenDataset()

        def setup(self, stage=None):
            raise AssertionError("CDC optimization must not re-enter datamodule setup")

    backbone = _DummyBackbone()
    model = ClusteringBaseModel(
        backbone=backbone,
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    datamodule = _NoReplayDataModule()
    trainer = CDCTrainer(
        model=model,
        datamodule=datamodule,
        config=_cdc_config({"prototype_init": False}),
        device=torch.device("cpu"),
    )
    meta_batch = cdc_trainer_module.CDCMetaBatch()
    ready, meta_batch = trainer._add_physical_batch_to_meta_batches(
        meta_batch, datamodule.batch
    )
    assert ready
    logs = trainer._process_meta_batch_or_skip(ready[0], epoch=0)

    assert logs is not None
    assert datamodule.train_dataset.calls == 0
    assert 1.0 in {round(value, 1) for value in backbone.seen_means}


def test_cdc_meta_batch_releases_stored_tensors_after_processing() -> None:
    model = ClusteringBaseModel(
        backbone=_DummyBackbone(),
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    trainer = CDCTrainer(
        model=model,
        datamodule=_TinyCDCDataModule(),
        config=_cdc_config({"prototype_init": False}),
        device=torch.device("cpu"),
    )
    meta_batch = cdc_trainer_module.CDCMetaBatch()
    ready, _current = trainer._add_physical_batch_to_meta_batches(
        meta_batch, _TinyCDCDataModule().batch
    )
    assert ready[0].strong_parts

    logs = trainer._process_meta_batch_or_skip(ready[0], epoch=0)

    assert logs is not None
    assert ready[0].num_samples == 0
    assert not ready[0].strong_parts


def test_cdc_epoch_metrics_are_reported_from_deterministic_predictions() -> None:
    backbone = _DummyBackbone()
    model = ClusteringBaseModel(
        backbone=backbone,
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    trainer = CDCTrainer(
        model=model,
        datamodule=_TinyCDCDataModule(),
        config=_cdc_config({"prototype_init": False}),
        device=torch.device("cpu"),
    )
    metrics = trainer.evaluate_epoch_metrics()

    assert set(metrics) == {"ari", "nmi", "acc"}
    assert all(isinstance(value, float) for value in metrics.values())


def _make_cdc_trainer(
    config_overrides: dict | None = None,
    *,
    trainer_overrides: dict | None = None,
    checkpoint_dir: Path | None = None,
    resume_from_checkpoint: Path | None = None,
) -> CDCTrainer:
    model = ClusteringBaseModel(
        backbone=_DummyBackbone(),
        adapter=None,
        head=CDCHead(input_dim=4, hidden_dim=8, n_clusters=2),
    )
    config = _cdc_config({"prototype_init": False, **(config_overrides or {})})
    config["trainer"]["deterministic"] = False
    config["trainer"].update(trainer_overrides or {})
    return CDCTrainer(
        model=model,
        datamodule=_TinyCDCDataModule(),
        config=config,
        device=torch.device("cpu"),
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def _populate_cdc_optimizer_state(trainer: CDCTrainer) -> None:
    trainer.optimizer_cluster.zero_grad(set_to_none=True)
    cluster_loss = sum(parameter.sum() for parameter in trainer.model.head.clustering_head.parameters())
    cluster_loss.backward()
    trainer.optimizer_cluster.step()
    trainer.optimizer_calibration.zero_grad(set_to_none=True)
    calibration_loss = sum(parameter.sum() for parameter in trainer.model.head.calibration_head.parameters())
    calibration_loss.backward()
    trainer.optimizer_calibration.step()


def test_cdc_auto_resume_loads_latest_checkpoint(tmp_path: Path) -> None:
    source = _make_cdc_trainer(checkpoint_dir=tmp_path)
    source.global_step = 11
    _populate_cdc_optimizer_state(source)
    source.save_epoch_checkpoint(epoch=1)

    resume_path = resolve_resume_checkpoint("auto", tmp_path)
    resumed = _make_cdc_trainer(
        checkpoint_dir=tmp_path,
        resume_from_checkpoint=resume_path,
    )

    assert resumed.start_epoch == 2
    assert resumed.global_step == 11
    assert resumed.optimizer_cluster.state_dict()["state"]
    assert resumed.optimizer_calibration.state_dict()["state"]


def test_cdc_null_resume_ignores_latest_checkpoint(tmp_path: Path) -> None:
    source = _make_cdc_trainer(checkpoint_dir=tmp_path)
    source.global_step = 7
    source.save_epoch_checkpoint(epoch=0)

    resumed = _make_cdc_trainer(
        checkpoint_dir=tmp_path,
        resume_from_checkpoint=resolve_resume_checkpoint(None, tmp_path),
    )

    assert resumed.start_epoch == 0
    assert resumed.global_step == 0


def test_cdc_explicit_checkpoint_path_loads_exact_file(tmp_path: Path) -> None:
    source = _make_cdc_trainer(checkpoint_dir=tmp_path)
    source.global_step = 3
    source.save_epoch_checkpoint(epoch=0)
    explicit = tmp_path / "manual.ckpt"
    source.global_step = 13
    torch.save(source.resumable_checkpoint_payload(epoch=4), explicit)

    resumed = _make_cdc_trainer(checkpoint_dir=tmp_path, resume_from_checkpoint=explicit)

    assert resumed.start_epoch == 5
    assert resumed.global_step == 13


def test_resumed_cdc_skips_prototype_initialization(tmp_path: Path) -> None:
    source = _make_cdc_trainer(checkpoint_dir=tmp_path)
    source.save_epoch_checkpoint(epoch=0)
    resumed = _make_cdc_trainer(
        {"prototype_init": True},
        trainer_overrides={"eval_interval": "final_only"},
        checkpoint_dir=tmp_path,
        resume_from_checkpoint=tmp_path / "latest.ckpt",
    )

    def fail_initialize() -> None:
        raise AssertionError("prototype initialization should be skipped on resume")

    resumed.initialize_head = fail_initialize
    resumed.fit()

    assert resumed.start_epoch == 1


def test_cdc_checkpoint_eval_schedule_skips_non_eval_epochs(tmp_path: Path) -> None:
    trainer = _make_cdc_trainer(
        trainer_overrides={"eval_interval": "checkpoint", "checkpoint_interval": 5},
        checkpoint_dir=tmp_path,
    )
    trainer.config["trainer"]["max_epochs"] = 2
    calls = 0

    def fake_eval() -> dict[str, float]:
        nonlocal calls
        calls += 1
        return {"ari": 1.0, "nmi": 1.0, "acc": 1.0}

    trainer.evaluate_epoch_metrics = fake_eval
    trainer.fit()

    assert calls == 1
    assert trainer.epoch_history[0]["evaluated"] is False
    assert torch.isnan(torch.tensor(trainer.epoch_history[0]["ari"]))
    assert trainer.epoch_history[1]["evaluated"] is True
    assert {"train_epoch_time_s", "checkpoint_save_time_s", "eval_time_s", "epoch_total_wall_time_s"}.issubset(
        trainer.epoch_history[0]
    )
    assert trainer.training_logs()["resource_totals"]["total_train_time_s"] > 0.0


def test_cdc_interval_checkpoints_are_not_pruned(tmp_path: Path) -> None:
    trainer = _make_cdc_trainer(checkpoint_dir=tmp_path)

    trainer.save_epoch_checkpoint(epoch=0)
    trainer.save_epoch_checkpoint(epoch=1)

    assert (tmp_path / "epoch_0001.ckpt").exists()
    assert (tmp_path / "epoch_0002.ckpt").exists()
    assert (tmp_path / "latest.ckpt").exists()


def test_cdc_checkpoint_metric_logger_runs_on_checkpoint_eval(tmp_path: Path) -> None:
    rows: list[dict] = []
    trainer = _make_cdc_trainer(
        trainer_overrides={"checkpoint_interval": 1, "eval_interval": "checkpoint"},
        checkpoint_dir=tmp_path,
    )
    trainer.checkpoint_metric_logger = rows.append
    trainer.config["trainer"]["max_epochs"] = 1
    trainer.evaluate_epoch_metrics = lambda: {"ari": 0.1, "nmi": 0.2, "acc": 0.3}

    trainer.fit()

    assert len(rows) == 1
    assert rows[0]["checkpoint_saved"] is True
    assert rows[0]["evaluated"] is True
    assert rows[0]["nmi"] == 0.2


def test_run_cdc_script_parses_one_config_path() -> None:
    source = Path("scripts/run_cdc.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("config_path"' in source
    forbidden = ["--dataset", "--seed", "--epochs", "--batch-size", "--num-workers", "--output-root", "--gloca"]
    assert not any(token in source for token in forbidden)


def test_run_cdc_script_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_cdc.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "config_path" in result.stdout


def test_plantseg_cdc_ablation_configs_validate() -> None:
    config_paths = [
        Path("configs/cdc/cdc_plantseg_gloca_mean_pool_seed42.yaml"),
        Path("configs/cdc/cdc_plantseg_gloca_no_patches_seed42.yaml"),
        Path("configs/cdc/cdc_plantseg_gloca_no_cls_residual_seed42.yaml"),
    ]

    for config_path in config_paths:
        config = load_experiment_config(config_path)
        validate_cdc_config(config)
        assert config["dataset"]["name"] == "plantseg"
        assert config["experiment"]["seed"] == 42
        assert config["experiment"]["output_dir"] == "outputs/ablation_studies"
