from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import yaml

from src.data import ClusteringDataModule
from src.data.folder_dataset import DatasetSpec
from src.experiments import baselines
from src.experiments.cli import add_seed_arguments, override_seeds, seed_value
from src.experiments.reproducibility import (
    atomic_json, dataset_identity, dataset_records, digest, feature_cache_key,
    is_complete, record_run, validate_checkpoint,
)
from src.experiments.reproducibility import backbone_identity as actual_backbone_identity
from src.runners.common import apply_runtime_settings
from src.utils import resolve_output_dir, seed_everything

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def synthetic_backbone_identity(monkeypatch):
    # Protocol tests use tiny synthetic features; real pinned weights are tested in CUDA smoke runs.
    import src.experiments.reproducibility as reproduction
    monkeypatch.setattr(reproduction, "backbone_identity", lambda config: {"variant": config["variant"], "revision": config["revision"], "files": {"synthetic": "test"}})


@pytest.fixture
def images(tmp_path):
    root = tmp_path / "images"
    generator = np.random.default_rng(123)
    for label in range(2):
        directory = root / str(label)
        directory.mkdir(parents=True)
        for index in range(4):
            pixels = generator.integers(0, 70, (40, 40, 3), dtype=np.uint8)
            pixels[:, :, label] += 170
            Image.fromarray(pixels).save(directory / f"{index}.png")
    return root


def tiny_config(images, output):
    sweep = yaml.safe_load((ROOT / "configs/baselines/baselines_smoke.yaml").read_text())
    sweep["sweep"]["datasets"] = ["plantseg"]
    sweep["base_config"]["dataset"]["root"] = str(images)
    sweep["experiment"]["output_dir"] = str(output)
    config = baselines._build_plans(sweep)[0]["config"]
    config["dataset"]["limit_per_class"] = None
    config["head"]["n_clusters"] = 2
    config["trainer"].update(num_workers=0, batch_size=4, max_epochs=3, checkpoint_interval=1, eval_interval="every_epoch")
    return config


@pytest.mark.parametrize("value", ["-1", "4294967296", "x", "1.5", True])
def test_bad_seed_is_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        seed_value(value)


def test_seed_precedence_and_source_config_unchanged():
    config = {"base_config": {"experiment": {"seed": 42}}, "diagnostics": {"runs": [{"config": {"experiment": {"seed": 123}}}]}}
    resolved = override_seeds(config, seed=67)
    assert resolved["base_config"]["experiment"]["seed"] == 67
    assert resolved["diagnostics"]["runs"][0]["config"]["experiment"]["seed"] == 67
    assert config["diagnostics"]["runs"][0]["config"]["experiment"]["seed"] == 123
    sweep = {"sweep": {"seeds": [42, 99]}}
    assert override_seeds(sweep, seed=0)["sweep"]["seeds"] == [0]
    assert override_seeds(sweep, seeds=[42, 67, 69])["sweep"]["seeds"] == [42, 67, 69]
    with pytest.raises(ValueError, match="unique"):
        override_seeds(sweep, seeds=[42, 42])
    parser = argparse.ArgumentParser()
    add_seed_arguments(parser, sweep=True)
    with pytest.raises(SystemExit):
        parser.parse_args(["--seed", "67", "--seeds", "42", "69"])


def test_all_baseline_configs_expand_and_override():
    for path in (ROOT / "configs/baselines").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        assert config["sweep"]["seeds"] == [42]
        plans = baselines._build_plans(override_seeds(config, seeds=[42, 67, 69]))
        assert {p["seed"] for p in plans} == {42, 67, 69}
        assert all(f"seed_{p['seed']}" in str(resolve_output_dir(p["config"])) for p in plans)


def test_no_non42_seed_variants_and_configs_load():
    import re
    for path in (ROOT / "configs").rglob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        assert not re.search(r"seed(?:67|69|99|123)", path.name)
        if "experiment" in config and "seed" in config["experiment"] and "templates" not in path.parts:
            assert config["experiment"]["seed"] == 42


def test_duplicate_destinations_rejected():
    config = yaml.safe_load((ROOT / "configs/baselines/baselines_smoke.yaml").read_text())
    config["sweep"]["runs"].append(deepcopy(config["sweep"]["runs"][0]))
    with pytest.raises(ValueError, match="duplicate|unique"):
        baselines._build_plans(config)


def test_manifest_detects_bytes_and_membership(images, tmp_path):
    spec = DatasetSpec("plantseg", images)
    records = dataset_records(spec)
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, {"version": 1, "dataset": "plantseg", "fingerprint": digest(records), "class_names": ["0", "1"], "records": records})
    config = {"name": "plantseg", "root": str(images), "manifest": str(manifest)}
    first = dataset_identity(config)
    Image.new("RGB", (40, 40), "red").save(images / "0/0.png")
    with pytest.raises(ValueError, match="content"):
        dataset_identity(config)
    config.pop("manifest")
    assert dataset_identity(config)["fingerprint"] != first["fingerprint"]


def test_cache_key_separates_inputs_but_not_seed(images, tmp_path):
    config = tiny_config(images, tmp_path)
    key = feature_cache_key(config, "cls")
    config["experiment"]["seed"] = 67
    assert feature_cache_key(config, "cls") == key
    assert feature_cache_key(config, "cls_mean_patch_concat") != key
    config["backbone"]["image_size"] = 28
    assert feature_cache_key(config, "cls") != key


def test_completion_requires_all_files_and_matching_identity(images, tmp_path):
    from src.experiments.outputs import write_outputs
    config = tiny_config(images, tmp_path)

    @record_run
    def run(config):
        directory = resolve_output_dir(config)
        payload = dict(head="kmeans", backbone=config["backbone"]["variant"], gloca="disabled", dataset="plantseg", seed=config["experiment"]["seed"], n_clusters=2,
                       image_ids=["0/0.png", "1/0.png"], labels=[0, 1], assignments=[0, 1], patch_grid=[1, 1])
        write_outputs(directory, config, payload, {"head": "kmeans"}, torch.eye(2), None, {})
        return {"output_dir": str(directory)}

    directory = resolve_output_dir(config)
    directory.mkdir(parents=True)
    (directory / "metrics.csv").write_text("partial")
    assert not is_complete(config)
    with pytest.raises(ValueError, match="Unverified"):
        run(config)
    (directory / "metrics.csv").unlink()
    result = run(config)
    assert is_complete(config)
    saved = yaml.safe_load((directory / "config.yaml").read_text())
    assert saved["experiment"]["seed"] == 42
    metadata = json.loads((directory / "reproducibility.json").read_text())
    assert metadata["config"]["trainer"]["deterministic"] is True
    moved = deepcopy(config)
    moved["experiment"]["output_dir"] = str(tmp_path / "another")
    from src.experiments.reproducibility import describe_run
    assert describe_run(moved)["fingerprint"] == metadata["fingerprint"]
    changed = deepcopy(config)
    changed["baseline"]["kmeans_n_init"] += 1
    with pytest.raises(ValueError, match="identity differs"):
        is_complete(changed)
    (directory / "embeddings.pt").unlink()
    assert not is_complete(config)


@pytest.mark.parametrize("workers", [0, 2])
def test_epoch_loaders_reproduce_shuffle_and_augmentation(images, workers):
    spec = DatasetSpec("plantseg", images)
    def module():
        return ClusteringDataModule(spec, image_size=28, batch_size=4, num_workers=workers, training_mode="contrastive_two_view", seed=67)
    first = module().train_dataloader()
    first.set_epoch(2)
    batch = next(iter(first))
    # Neither model initialization nor evaluation may advance the training stream.
    random.random()
    np.random.rand(10)
    torch.rand(10)
    other_module = module()
    list(other_module.train_eval_dataloader())
    resumed = other_module.train_dataloader()
    resumed.set_epoch(2)
    actual = next(iter(resumed))
    assert torch.equal(batch["index"], actual["index"])
    assert all(torch.equal(a, b) for a, b in zip(batch["views"], actual["views"]))
    resumed.set_epoch(3)
    assert not torch.equal(actual["views"][0], next(iter(resumed))["views"][0])


def test_summary_keeps_embedding_sources_and_supervised_metrics_separate():
    rows = [{"status": "success", "baseline_name": source, "embedding_source": source, "head": "kmeans", "dataset": "plantseg", "seed": 42, "ari": .5} for source in ["cls", "cls_mean_patch_concat"]]
    rows += [{"status": "success", "baseline_name": "mlp", "head": "supervised_mlp", "dataset": "plantseg", "seed": seed, "evaluation_split": "test", "classification_accuracy": score} for seed, score in [(42, .8), (67, 1.)]]
    aggregate = baselines._aggregate_rows(rows)
    assert len(aggregate) == 3
    mlp = next(row for row in aggregate if row["head"] == "supervised_mlp")
    assert float(mlp["classification_accuracy_mean"]) == .9


class ImageBackbone(nn.Module):
    output_dim = 4
    def __init__(self, **kwargs):
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False)
        self.requires_grad_(False)
    def forward(self, image):
        cls = self.projection(image.mean(dim=(2, 3)))
        return {"cls": cls, "patch_tokens": cls.unsqueeze(1).expand(-1, 4, -1), "patch_grid": (2, 2)}


class Interrupted(Exception):
    pass


@pytest.mark.parametrize("method,gloca,workers", [("cdc", False, 0), ("cdc", True, 0), ("propos", False, 0), ("propos", True, 0), ("cdc", True, 2)])
def test_live_training_exact_resume(images, tmp_path, method, gloca, workers):
    from src.models import ClusteringBaseModel, build_adapter
    from src.models.clustering import CDCHead, ProPosHead
    from src.training.cdc_trainer import CDCTrainer
    from src.training.propos_trainer import ProPosTrainer
    config = yaml.safe_load((ROOT / f"configs/{method}/{method}_plantseg_{'gloca_gated' if gloca else 'cls'}{'_full' if method == 'propos' else ''}_seed42.yaml").read_text())
    config["experiment"]["seed"] = 67
    config["head"]["n_clusters"] = 2
    config["gloca"].update(embedding_dim=4, attention_hidden_dim=4, mlp_hidden_dim=8)
    config["trainer"].update(max_epochs=2 if workers else 3, batch_size=4, num_workers=workers, checkpoint_interval=1, eval_interval="every_epoch")
    config[method].update(embedding_dim=4, hidden_dim=8, projection_dim=4, projection_hidden_dim=8, predictor_hidden_dim=8,
                          prototype_init=False, orthogonalize_init=False, calibration_k=2, meta_batch_size=4, sub_batch_size=2,
                          kmeans_n_init=1, kmeans_max_iter=4, warmup_epochs=0, log_gloca_diagnostics=gloca, progress=False)
    apply_runtime_settings(config)
    def trainer(directory, resume=None):
        seed_everything(67)
        dm = ClusteringDataModule(DatasetSpec("plantseg", images), 28, 4, workers, "cdc" if method == "cdc" else "contrastive_two_view", seed=67)
        head = CDCHead(input_dim=4, hidden_dim=8, n_clusters=2) if method == "cdc" else ProPosHead(embedding_dim=4, n_clusters=2, projection_dim=4, projection_hidden_dim=8, predictor_hidden_dim=8)
        model = ClusteringBaseModel(ImageBackbone(), build_adapter(config, 4), head)
        cls = CDCTrainer if method == "cdc" else ProPosTrainer
        return cls(model=model, datamodule=dm, config=deepcopy(config), device=torch.device("cpu"), checkpoint_dir=directory, resume_from_checkpoint=resume)
    full = trainer(tmp_path / "full")
    full.fit()
    repeat = trainer(tmp_path / "repeat")
    repeat.fit()
    interrupted = trainer(tmp_path / "interrupted")
    original = interrupted.save_epoch_checkpoint
    def stop(epoch):
        original(epoch)
        raise Interrupted()
    interrupted.save_epoch_checkpoint = stop
    with pytest.raises(Interrupted):
        interrupted.fit()
    resumed = trainer(tmp_path / "interrupted", tmp_path / "interrupted/latest.ckpt")
    resumed.fit()
    for other in [repeat, resumed]:
        for key, tensor in full.model.state_dict().items():
            assert torch.equal(tensor, other.model.state_dict()[key]), key
        assert full.global_step == other.global_step
        assert full.loss_total_final == other.loss_total_final
        if method == "propos":
            for key, tensor in full.target_encoder.state_dict().items():
                assert torch.equal(tensor, other.target_encoder.state_dict()[key]), key


@pytest.mark.parametrize("mode", ["dec", "idec"])
def test_cached_autoencoder_exact_resume(tmp_path, mode):
    from src.models.baselines.dec_idec import DINOCLSDECModel
    from src.training.dec_idec_trainer import DECIDECTrainer
    config = {"trainer": {"batch_size": 4, "lr": .001, "checkpoint_interval": 1, "deterministic": True},
              "baseline": {"pretrain_epochs": 3, "refine_epochs": 3, "target_update_interval": 1, "kmeans_init": "kmeans++", "kmeans_n_init": 1, "kmeans_max_iter": 4, "kmeans_tol": .0001}}
    apply_runtime_settings(config)
    features = torch.randn(12, 4, generator=torch.Generator().manual_seed(42))
    def trainer(path, resume=None):
        seed_everything(67)
        return DECIDECTrainer(DINOCLSDECModel(n_clusters=2, input_dim=4, hidden_dims=(8,), latent_dim=2), features,
                              deepcopy(config), mode, 2, 67, torch.device("cpu"), checkpoint_dir=path, resume_from_checkpoint=resume)
    full = trainer(tmp_path / "full")
    full.fit()
    interrupted = trainer(tmp_path / "interrupted")
    original = interrupted.save_phase_checkpoint
    def stop(phase, epoch):
        original(phase, epoch)
        if phase == "pretrain":
            raise Interrupted()
    interrupted.save_phase_checkpoint = stop
    with pytest.raises(Interrupted):
        interrupted.fit()
    resumed = trainer(tmp_path / "interrupted", tmp_path / "interrupted/latest.ckpt")
    resumed.fit()
    for key, tensor in full.model.state_dict().items():
        assert torch.equal(tensor, resumed.model.state_dict()[key]), key
    assert full.pretrain_losses == resumed.pretrain_losses
    assert full.refine_losses == resumed.refine_losses


def test_mlp_exact_resume(tmp_path, monkeypatch):
    import src.training.cached_cls_trainer as training
    features = torch.randn(24, 4, generator=torch.Generator().manual_seed(42))
    labels = torch.arange(2).repeat_interleave(12)
    def run(path, resume=None):
        seed_everything(67)
        return training.train_cached_cls_mlp(features, labels, n_classes=2, hidden_dims=[8, 4], seed=67, max_epochs=3,
                                             batch_size=4, learning_rate=.001, weight_decay=0, output_dir=path,
                                             checkpoint_interval=1, resume_from_checkpoint=resume)
    full = run(tmp_path / "full")
    original = training.copy_as_latest
    def stop(source, target):
        original(source, target)
        raise Interrupted()
    with monkeypatch.context() as patch:
        patch.setattr(training, "copy_as_latest", stop)
        with pytest.raises(Interrupted):
            run(tmp_path / "interrupted")
    resumed = run(tmp_path / "interrupted", "auto")
    assert full["history"] == resumed["history"]
    assert full["splits"].as_dict() == resumed["splits"].as_dict()
    for key, tensor in full["model"].state_dict().items():
        assert torch.equal(tensor, resumed["model"].state_dict()[key])


def test_legacy_and_mismatched_checkpoints_rejected():
    config = {"trainer": {"deterministic": True}, "experiment": {"seed": 67}}
    with pytest.raises(ValueError, match="Checkpoint"):
        validate_checkpoint({"rng_state": {}}, config)
    config["trainer"]["deterministic"] = False
    validate_checkpoint({}, config)


@pytest.mark.parametrize("method", ["cdc", "propos"])
def test_cli_seed_overrides_without_changing_yaml(tmp_path, monkeypatch, method):
    from importlib import import_module
    module = import_module("scripts.run_" + method)
    path = ROOT / f"configs/{method}/{method}_plantseg_cls{'_full' if method == 'propos' else ''}_seed42.yaml"
    before = path.read_bytes()
    captured = []
    monkeypatch.setattr(module, "run_" + method, lambda config: captured.append(config) or type("Result", (), {"output_dir": tmp_path})())
    monkeypatch.setattr(sys, "argv", [str(module.__file__), str(path), "--seed", "67"])
    module.main()
    assert captured[0]["experiment"]["seed"] == 67
    assert path.read_bytes() == before


@pytest.mark.parametrize("fail_fast", [False, True])
def test_failed_sweep_writes_summary_and_exits_nonzero(tmp_path, monkeypatch, fail_fast):
    config = yaml.safe_load((ROOT / "configs/baselines/baselines_smoke.yaml").read_text())
    config["experiment"]["output_dir"] = str(tmp_path / "output")
    config["sweep"]["fail_fast"] = fail_fast
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["run_baselines.py", str(path), "--seeds", "42", "67"])
    monkeypatch.setattr(baselines, "is_complete", lambda config: False)
    def fail(plan, caches):
        raise RuntimeError("failed extraction")
    monkeypatch.setattr(baselines, "_run_plan", fail)
    with pytest.raises(SystemExit) as error:
        baselines.main()
    assert error.value.code == 1
    import csv
    with (tmp_path / "output/baseline_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == (1 if fail_fast else 4)
    assert all(row["status"] == "failed" for row in rows)
    assert yaml.safe_load(path.read_text())["sweep"]["seeds"] == [42]


@pytest.mark.parametrize("method", ["cdc", "propos"])
def test_cls_and_gloca_start_with_identical_clustering_heads(images, tmp_path, monkeypatch, method):
    from importlib import import_module
    module = import_module("src.runners." + method)
    config = yaml.safe_load((ROOT / f"configs/{method}/{method}_plantseg_cls{'_full' if method == 'propos' else ''}_seed42.yaml").read_text())
    config["experiment"]["output_dir"] = str(tmp_path)
    config["dataset"].update(root=str(images), manifest=None)
    config["gloca"].update(embedding_dim=4, attention_hidden_dim=4, mlp_hidden_dim=8)
    captured = []
    def capture(model, **kwargs):
        captured.append(deepcopy(model.head.state_dict()))
        raise Interrupted()
    monkeypatch.setattr(module, "DINOv2Backbone", ImageBackbone)
    monkeypatch.setattr(module, "CDCTrainer" if method == "cdc" else "ProPosTrainer", capture)
    for enabled in [False, True]:
        config["gloca"].update(enabled=enabled, name="gloca_gated" if enabled else "disabled")
        with pytest.raises(Interrupted):
            getattr(module, "run_" + method).__wrapped__(config)
    assert captured[0].keys() == captured[1].keys()
    assert all(torch.equal(value, captured[1][key]) for key, value in captured[0].items())


def test_superseded_scripts_are_absent():
    assert not (ROOT / "scripts/run_cached_cls_baselines.py").exists()
    assert not (ROOT / "scripts/run_global_local_baseline.py").exists()


def test_verification_is_shared_only_within_an_invocation(images, monkeypatch):
    import src.experiments.reproducibility as reproduction
    config = {"name": "plantseg", "root": str(images)}
    original = reproduction.dataset_records
    calls = []
    def counted(spec):
        calls.append(spec)
        return original(spec)
    monkeypatch.setattr(reproduction, "dataset_records", counted)
    with reproduction.verification_session():
        reproduction.dataset_identity(config)
        reproduction.dataset_identity(config)
    assert len(calls) == 1
    reproduction.dataset_identity(config)
    assert len(calls) == 2


def test_backbone_default_revision_is_immutable(monkeypatch):
    from types import SimpleNamespace
    from src.features import dinov2_backbone as module
    from src.features.dinov2 import DINOV2_SMALL_REVISION
    captured = []
    model = nn.Linear(1, 1)
    model.config = SimpleNamespace(hidden_size=4, patch_size=14)
    def load(variant, **kwargs):
        captured.append(kwargs)
        return model
    monkeypatch.setattr(module.AutoModel, "from_pretrained", load)
    backbone = module.DINOv2Backbone("facebook/dinov2-small")
    assert captured == [{"revision": DINOV2_SMALL_REVISION}]
    assert all(not p.requires_grad for p in backbone.parameters())


def test_cls_cache_cannot_bypass_enabled_gloca(images, tmp_path):
    from src.experiments.reproducibility import validate_feature_cache
    config = tiny_config(images, tmp_path)
    cache = {"feature_key": feature_cache_key(config, "cls")}
    config["gloca"].update(enabled=True, name="gloca_gated")
    with pytest.raises(ValueError, match="cannot replace GLoCA"):
        validate_feature_cache(config, cache, "cls")


@pytest.mark.parametrize("method", ["kmeans", "dec_idec"])
def test_cached_runner_total_retains_extraction_cost(images, tmp_path, method):
    from importlib import import_module
    config = tiny_config(images, tmp_path)
    config["trainer"]["accelerator"] = "cpu"
    if method == "dec_idec":
        sweep = yaml.safe_load((ROOT / "configs/baselines/baselines_full.yaml").read_text())
        recipe = next(run["config"] for run in sweep["sweep"]["runs"] if run["name"] == "dec_cls")
        config["head"] = recipe["head"]
        config["head"]["n_clusters"] = 2
        config["baseline"] = recipe["baseline"]
        config["baseline"].update(input_dim=4, hidden_dims=[8], latent_dim=2, pretrain_epochs=1, refine_epochs=1, kmeans_n_init=1, kmeans_max_iter=4)
    backbone = ImageBackbone()
    records = dataset_records(DatasetSpec("plantseg", images))
    cache = {"feature_key": feature_cache_key(config, "cls"), "backbone": backbone,
             "embeddings": torch.nn.functional.normalize(torch.randn(8, 4, generator=torch.Generator().manual_seed(42)), dim=-1),
             "labels": torch.arange(2).repeat_interleave(4), "image_ids": [r["image_id"] for r in records],
             "indices": torch.arange(8), "patch_grid": (1, 1), "attention": None, "cache_time_s": 25.}
    function = getattr(import_module("src.runners." + method), "run_" + method)
    result = function(config, cache)
    assert result["metrics"]["total_time_s"] >= 25. + result["metrics"]["head_train_time_s"]


def test_source_fingerprint_is_available_without_git(monkeypatch):
    import src.experiments.reproducibility as reproduction
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(reproduction.subprocess, "run", missing)
    identity = reproduction.source_identity()
    assert len(identity["fingerprint"]) == 64
    assert identity["git_revision"] is None


def test_backbone_verification_detects_changed_cache_bytes(tmp_path, monkeypatch):
    import huggingface_hub
    from src.features import dinov2
    from src.experiments.reproducibility import file_hash
    paths = {}
    for name in ["config.json", "model.safetensors"]:
        paths[name] = tmp_path / name
        paths[name].write_bytes(name.encode())
    expected = {name: file_hash(path) for name, path in paths.items()}
    monkeypatch.setattr(dinov2, "DINOV2_SMALL_FILES", expected)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda variant, name, **kwargs: str(paths[name]))
    config = {"variant": dinov2.DINOV2_SMALL, "revision": dinov2.DINOV2_SMALL_REVISION}
    assert actual_backbone_identity(config)["files"] == expected
    paths["model.safetensors"].write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="cache content differs"):
        actual_backbone_identity(config)
