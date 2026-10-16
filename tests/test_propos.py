from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.evaluation.assignment_schema import validate_assignment_payload
from src.models import ClusteringBaseModel
from src.models.clustering import ProPosHead
from src.models.clustering.kmeans import TorchKMeans, TorchSphericalKMeans, fit_kmeans, torch_spherical_kmeans
from src.models.gloca import GLoCAGatedAdapter
from src.training.propos_trainer import ProPosTrainer


def test_propos_projection_and_predictor_shapes() -> None:
    head = ProPosHead(
        n_clusters=3,
        embedding_dim=384,
        projection_dim=64,
        projection_hidden_dim=128,
        predictor_hidden_dim=128,
    )
    out = head.forward_online(torch.randn(8, 384))
    assert out["projected"].shape == (8, 64)
    assert out["predicted"].shape == (8, 64)


def test_propos_psa_loss_is_finite() -> None:
    head = ProPosHead(
        n_clusters=3,
        embedding_dim=16,
        projection_dim=8,
        projection_hidden_dim=32,
        predictor_hidden_dim=32,
    )
    online = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    target = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    loss = head.positive_sampling_alignment(online, target)
    assert torch.isfinite(loss)


def test_propos_psa_uses_official_negative_cosine() -> None:
    head = ProPosHead(
        n_clusters=3,
        embedding_dim=16,
        projection_dim=8,
        projection_hidden_dim=32,
        predictor_hidden_dim=32,
    )
    head.eval()
    online = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    target = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    loss = head.positive_sampling_alignment(online, target, sigma=0.0)
    manual = -2.0 * torch.nn.functional.cosine_similarity(
        torch.nn.functional.normalize(head.predictor(online), dim=-1),
        torch.nn.functional.normalize(target.detach(), dim=-1),
        dim=-1,
    ).mean()
    assert torch.allclose(loss, manual)


def test_propos_psl_loss_is_finite_with_nonempty_clusters() -> None:
    head = ProPosHead(
        n_clusters=3,
        embedding_dim=16,
        projection_dim=8,
        projection_hidden_dim=32,
        predictor_hidden_dim=32,
    )
    online = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    target = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    loss, valid = head.prototype_scattering_loss(online, target, labels)
    assert torch.isfinite(loss)
    assert valid


def test_propos_psl_returns_zero_for_less_than_two_nonempty_clusters() -> None:
    head = ProPosHead(
        n_clusters=3,
        embedding_dim=16,
        projection_dim=8,
        projection_hidden_dim=32,
        predictor_hidden_dim=32,
    )
    online = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    target = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    labels = torch.zeros(4, dtype=torch.long)
    loss, valid = head.prototype_scattering_loss(online, target, labels)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert not valid


def test_propos_ema_update_changes_target_parameters() -> None:
    head = ProPosHead(
        n_clusters=3,
        embedding_dim=16,
        projection_dim=8,
        projection_hidden_dim=32,
        predictor_hidden_dim=32,
    )
    target_before = [parameter.detach().clone() for parameter in head.target_projector.parameters()]
    with torch.no_grad():
        for parameter in head.projector.parameters():
            parameter.add_(0.25)
    head.update_target_projector(momentum=0.5)
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(target_before, head.target_projector.parameters())
    )


class _DummyBackbone(nn.Module):
    output_dim = 16

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 16)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, image: torch.Tensor) -> dict:
        cls = self.proj(image)
        return {"cls": cls, "patch_tokens": torch.empty(image.shape[0], 0, 16), "patch_grid": (0, 0)}


class _DummyAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 16)

    def forward(self, cls: torch.Tensor, patch_tokens: torch.Tensor, patch_grid=None) -> dict:
        return {"embedding": torch.nn.functional.normalize(self.proj(cls), dim=-1), "attention": None, "patch_grid": patch_grid}


class _PatchBackbone(nn.Module):
    output_dim = 16

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 16)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, image: torch.Tensor) -> dict:
        cls = self.proj(image)
        patch_tokens = cls.unsqueeze(1).repeat(1, 4, 1)
        return {"cls": cls, "patch_tokens": patch_tokens, "patch_grid": (2, 2)}


class _TinyEvalDataModule:
    def train_eval_dataloader(self):
        return [
            {
                "image": torch.randn(2, 4),
                "label": torch.tensor([0, 1]),
                "index": torch.tensor([0, 1]),
            }
        ]


class _TinyProPosDataModule(_TinyEvalDataModule):
    def train_dataloader(self):
        return [
            {
                "views": (torch.randn(2, 4), torch.randn(2, 4)),
                "label": torch.tensor([0, 1]),
                "index": torch.tensor([0, 1]),
            }
        ]


def _base_propos_config(overrides: dict | None = None) -> dict:
    return {
        "ema_momentum": 0.996,
        "ema_momentum_max": 1.0,
        "ema_momentum_increase": True,
        "kmeans_interval": 1,
        "kmeans_init": "kmeans++",
        "kmeans_n_init": 2,
        "kmeans_max_iter": 20,
        "kmeans_tol": 1.0e-4,
        "sigma": 0.001,
        "lambda_psl": 0.1,
        "symmetric_loss": True,
        "profile_batches": False,
        "predictor_lr_multiplier": 10.0,
        "weight_decay": 0.0,
        "optimizer": "adamw",
        "warmup_epochs": 0,
        "freeze_gloca": False,
        "freeze_gloca_epochs": 0,
        "gloca_lr_multiplier": 1.0,
        "gloca_alpha_lr_multiplier": 1.0,
        "log_gloca_diagnostics": True,
        **(overrides or {}),
    }


def _make_gloca_trainer(propos_config: dict | None = None) -> ProPosTrainer:
    config = {
        "trainer": {"max_epochs": 1, "batch_size": 2, "lr": 1.0e-3},
        "propos": _base_propos_config(propos_config),
    }
    model = ClusteringBaseModel(
        backbone=_PatchBackbone(),
        adapter=GLoCAGatedAdapter(input_dim=16, embedding_dim=16, alpha_init=0.0),
        head=ProPosHead(
            n_clusters=2,
            embedding_dim=16,
            projection_dim=8,
            projection_hidden_dim=16,
            predictor_hidden_dim=16,
        ),
    )
    return ProPosTrainer(model=model, datamodule=_TinyEvalDataModule(), config=config, device=torch.device("cpu"))


def _make_propos_resume_trainer(
    tmp_path: Path | None = None,
    resume_from_checkpoint: Path | None = None,
    *,
    max_epochs: int = 1,
) -> ProPosTrainer:
    config = {
        "trainer": {
            "max_epochs": max_epochs,
            "deterministic": False,
            "batch_size": 2,
            "lr": 1.0e-3,
            "checkpoint_interval": 1,
            "eval_interval": "checkpoint",
        },
        "experiment": {"seed": 3},
        "propos": _base_propos_config({"kmeans_interval": 100}),
    }
    model = ClusteringBaseModel(
        backbone=_DummyBackbone(),
        adapter=None,
        head=ProPosHead(
            n_clusters=2,
            embedding_dim=16,
            projection_dim=8,
            projection_hidden_dim=16,
            predictor_hidden_dim=16,
        ),
    )
    return ProPosTrainer(
        model=model,
        datamodule=_TinyProPosDataModule(),
        config=config,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def _populate_propos_optimizer_state(trainer: ProPosTrainer) -> None:
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.sum() for parameter in trainer.model.head.projector.parameters())
    loss.backward()
    trainer.optimizer.step()


def test_backbone_parameters_remain_frozen_in_clustering_model() -> None:
    backbone = _DummyBackbone()
    model = ClusteringBaseModel(
        backbone=backbone,
        adapter=None,
        head=ProPosHead(
            n_clusters=2,
            embedding_dim=16,
            projection_dim=8,
            projection_hidden_dim=32,
            predictor_hidden_dim=32,
        ),
    )
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())


def test_target_adapter_updates_by_ema_when_gloca_enabled() -> None:
    model = ClusteringBaseModel(
        backbone=_DummyBackbone(),
        adapter=_DummyAdapter(),
        head=ProPosHead(
            n_clusters=2,
            embedding_dim=16,
            projection_dim=8,
            projection_hidden_dim=32,
            predictor_hidden_dim=32,
        ),
    )
    trainer = ProPosTrainer(
        model=model,
        datamodule=object(),
        config={
            "trainer": {"max_epochs": 1, "batch_size": 2, "lr": 1.0e-3},
            "propos": _base_propos_config(),
        },
        device=torch.device("cpu"),
    )
    assert trainer.target_encoder.adapter is not None
    before = [parameter.detach().clone() for parameter in trainer.target_encoder.adapter.parameters()]
    with torch.no_grad():
        for parameter in model.adapter.parameters():
            parameter.add_(0.5)
    trainer.update_target_adapter(momentum=0.5)
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, trainer.target_encoder.adapter.parameters())
    )


def test_propos_optimizer_uses_separate_gloca_parameter_groups() -> None:
    trainer = _make_gloca_trainer({"gloca_lr_multiplier": 0.1, "gloca_alpha_lr_multiplier": 2.0})
    groups = {group["name"]: group for group in trainer.optimizer.param_groups}

    assert {"projector", "predictor", "gloca", "gloca_alpha"}.issubset(groups)
    assert groups["projector"]["lr"] == 1.0e-3
    assert groups["predictor"]["lr"] == 1.0e-2
    assert groups["gloca"]["lr"] == 1.0e-4
    assert groups["gloca_alpha"]["lr"] == 2.0e-3


def test_propos_optimizer_does_not_duplicate_alpha_parameter() -> None:
    trainer = _make_gloca_trainer()
    alpha_id = id(trainer.model.adapter.alpha)
    occurrences = 0
    for group in trainer.optimizer.param_groups:
        occurrences += sum(id(parameter) == alpha_id for parameter in group["params"])

    assert occurrences == 1


def test_freeze_gloca_excludes_gloca_from_optimizer_and_requires_grad() -> None:
    trainer = _make_gloca_trainer({"freeze_gloca": True})
    group_names = {group["name"] for group in trainer.optimizer.param_groups}

    assert not any(parameter.requires_grad for parameter in trainer.model.adapter.parameters())
    assert "gloca" not in group_names
    assert "gloca_alpha" not in group_names


def test_freeze_gloca_epochs_toggles_online_adapter_trainability() -> None:
    trainer = _make_gloca_trainer({"freeze_gloca_epochs": 2})
    group_names = {group["name"] for group in trainer.optimizer.param_groups}

    assert not any(parameter.requires_grad for parameter in trainer.model.adapter.parameters())
    assert {"gloca", "gloca_alpha"}.issubset(group_names)
    trainer.maybe_update_gloca_freeze_state(epoch=1)
    assert not any(parameter.requires_grad for parameter in trainer.model.adapter.parameters())
    trainer.maybe_update_gloca_freeze_state(epoch=2)
    assert all(parameter.requires_grad for parameter in trainer.model.adapter.parameters())


def test_gloca_diagnostics_run_without_training_output_changes() -> None:
    trainer = _make_gloca_trainer()
    diagnostics = trainer.compute_gloca_diagnostics(epoch=0)

    assert diagnostics is not None
    assert set(diagnostics) == {
        "epoch",
        "gloca_alpha_value",
        "gloca_alpha_grad_norm",
        "gloca_param_grad_norm",
        "gloca_delta_norm_mean",
        "gloca_delta_norm_std",
        "gloca_embedding_cls_cosine_mean",
        "gloca_embedding_cls_cosine_std",
        "gloca_attention_entropy_mean",
        "gloca_attention_max_mean",
        "gloca_trainable",
        "gloca_frozen",
    }
    assert diagnostics["gloca_alpha_value"] == 0.0
    assert diagnostics["gloca_embedding_cls_cosine_mean"] > 0.999
    assert diagnostics["gloca_delta_norm_mean"] is not None
    assert diagnostics["gloca_attention_entropy_mean"] is not None


def test_pseudo_label_indexing_uses_batch_indices() -> None:
    head = ProPosHead(
        n_clusters=3,
        embedding_dim=16,
        projection_dim=8,
        projection_hidden_dim=32,
        predictor_hidden_dim=32,
    )
    head.set_pseudo_labels(torch.tensor([0, 2, 1, 2, 0, 1]))
    assert head.batch_pseudo_labels(torch.tensor([5, 2, 0])).tolist() == [1, 1, 0]


def test_torch_spherical_kmeans_supports_kmeans_plus_plus() -> None:
    features = torch.randn(24, 8)
    labels, centers, logs = torch_spherical_kmeans(
        features,
        n_clusters=4,
        init="kmeans++",
        n_init=2,
        max_iter=20,
        seed=7,
    )
    assert labels.shape == (24,)
    assert labels.dtype == torch.long
    assert centers.shape == (4, 8)
    assert torch.allclose(centers.norm(dim=-1), torch.ones(4), atol=1.0e-5)
    assert logs["backend"] == "torch"
    assert logs["init"] == "kmeans++"


def test_torch_kmeans_module_fit_predict_shapes() -> None:
    features = torch.randn(24, 8)
    model = TorchKMeans(n_clusters=4, init="kmeans++", n_init=2, max_iter=20, seed=5)
    result = model.fit_predict(features)

    assert isinstance(model, torch.nn.Module)
    assert result["assignments"].shape == (24,)
    assert result["centers"].shape == (4, 8)
    assert torch.isfinite(result["centers"]).all()
    assert result["logs"]["algorithm"] == "kmeans"


def test_torch_spherical_kmeans_module_fit_predict_shapes() -> None:
    features = torch.randn(24, 8)
    model = TorchSphericalKMeans(n_clusters=4, init="random", n_init=2, max_iter=20, seed=5)
    result = model.fit_predict(features)

    assert isinstance(model, torch.nn.Module)
    assert result["assignments"].shape == (24,)
    assert result["centers"].shape == (4, 8)
    assert torch.allclose(result["centers"].norm(dim=-1), torch.ones(4), atol=1.0e-5)
    assert result["logs"]["algorithm"] == "spherical_kmeans"


def test_fit_kmeans_dispatches_euclidean_and_spherical() -> None:
    features = torch.randn(18, 5)
    euclidean = fit_kmeans(
        features,
        3,
        spherical=False,
        init="random",
        n_init=1,
        max_iter=10,
        tol=1.0e-4,
        seed=1,
    )
    spherical = fit_kmeans(
        features,
        3,
        spherical=True,
        init="kmeans++",
        n_init=1,
        max_iter=10,
        tol=1.0e-4,
        seed=1,
    )

    assert euclidean["logs"]["algorithm"] == "kmeans"
    assert spherical["logs"]["algorithm"] == "spherical_kmeans"
    assert torch.isfinite(euclidean["centers"]).all()
    assert torch.allclose(spherical["centers"].norm(dim=-1), torch.ones(3), atol=1.0e-5)


def test_torch_spherical_kmeans_supports_random_init() -> None:
    features = torch.randn(16, 5)
    labels, centers, logs = torch_spherical_kmeans(
        features,
        n_clusters=3,
        init="random",
        n_init=1,
        max_iter=10,
        seed=3,
    )
    assert labels.min().item() >= 0
    assert labels.max().item() < 3
    assert torch.isfinite(centers).all()
    assert logs["init"] == "random"


def test_torch_spherical_kmeans_plus_plus_falls_back_for_identical_features() -> None:
    features = torch.ones(6, 4)
    labels, centers, logs = torch_spherical_kmeans(
        features,
        n_clusters=3,
        init="kmeans++",
        n_init=1,
        max_iter=5,
        seed=11,
    )
    assert labels.shape == (6,)
    assert torch.isfinite(centers).all()
    assert logs["empty_reinitializations"] >= 0


def test_propos_runner_imports_no_lightning() -> None:
    source = open("src/runners/propos.py", encoding="utf-8").read()
    trainer_source = open("src/training/propos_trainer.py", encoding="utf-8").read()
    forbidden = ["pytorch_lightning", "pl.Trainer", "src.models.lightning", "LightningModule"]
    assert not any(token in source for token in forbidden)
    assert not any(token in trainer_source for token in forbidden)


def test_assignment_arrays_have_equal_length() -> None:
    validate_assignment_payload(
        {
            "head": "propos",
            "backbone": "facebook/dinov2-small",
            "gloca": "disabled",
            "dataset": "plantwild",
            "seed": 42,
            "n_clusters": 2,
            "image_ids": ["a.jpg", "b.jpg"],
            "labels": [0, 1],
            "assignments": [1, 0],
            "patch_grid": [16, 16],
        }
    )


def test_propos_resume_restores_optimizer_target_and_history(tmp_path: Path) -> None:
    source = _make_propos_resume_trainer(tmp_path)
    source.global_step = 9
    source.model.head.set_pseudo_labels(torch.tensor([0, 1, 0, 1, 0, 1]))
    source.estep_history.append({"epoch": 0, "nmi": 0.5, "ari": 0.25, "acc": 0.75})
    _populate_propos_optimizer_state(source)
    with torch.no_grad():
        for parameter in source.target_encoder.parameters():
            parameter.fill_(0.123)
    source.save_epoch_checkpoint(epoch=0)

    resumed = _make_propos_resume_trainer(
        tmp_path, resume_from_checkpoint=tmp_path / "latest.ckpt"
    )

    assert resumed.start_epoch == 1
    assert resumed.global_step == 9
    assert resumed.model.head.pseudo_labels.tolist() == [0, 1, 0, 1, 0, 1]
    assert resumed.estep_history == source.estep_history
    assert resumed.optimizer.state_dict()["state"]
    for parameter in resumed.target_encoder.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 0.123))


def test_resumed_propos_skips_target_reinitialization(tmp_path: Path) -> None:
    source = _make_propos_resume_trainer(tmp_path)
    source.save_epoch_checkpoint(epoch=0)
    resumed = _make_propos_resume_trainer(
        tmp_path, resume_from_checkpoint=tmp_path / "latest.ckpt", max_epochs=1
    )

    def fail_update(momentum: float) -> None:
        raise AssertionError("target projector should not be reinitialized on resume")

    resumed.model.head.update_target_projector = fail_update
    resumed.update_target_adapter = fail_update
    resumed.fit()

    assert resumed.start_epoch == 1


def test_propos_training_continues_from_next_epoch(tmp_path: Path) -> None:
    source = _make_propos_resume_trainer(tmp_path)
    source.save_epoch_checkpoint(epoch=0)
    resumed = _make_propos_resume_trainer(
        tmp_path, resume_from_checkpoint=tmp_path / "latest.ckpt", max_epochs=2
    )
    seen_epochs: list[int] = []

    def fake_train_epoch(epoch: int, train_loader=None, max_epochs=None) -> dict:
        seen_epochs.append(epoch)
        return {
            "epoch": epoch,
            "n_batches": 1,
            "loss_total_mean": 0.1,
            "loss_psa_mean": 0.1,
            "loss_psl_mean": 0.0,
            "ari": 0.0,
            "nmi": 0.0,
            "acc": 0.0,
        }

    resumed.train_epoch = fake_train_epoch
    resumed.compute_gloca_diagnostics = lambda epoch: None
    resumed.fit()

    assert seen_epochs == [1]
    assert resumed.epoch_history[-1]["checkpoint_saved"] is True
    assert "train_epoch_time_s" in resumed.epoch_history[-1]
