"""Content identities and completion checks for local, single-device experiments."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "transformers", "huggingface-hub", "Pillow", "PyYAML", "tqdm", "threadpoolctl")
_verified_inputs = ContextVar("verified_inputs", default=None)


@contextmanager
def verification_session():
    """Verify immutable input bytes once per invocation, not once per seed/head."""
    if _verified_inputs.get() is not None:
        yield
        return
    token = _verified_inputs.set({})
    try:
        yield
    finally:
        _verified_inputs.reset(token)


def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def dataset_records(spec) -> list[dict]:
    from src.data.folder_dataset import FolderImageDataset

    return [{"image_id": r.image_id, "label": r.label, "label_name": r.label_name, "sha256": file_hash(r.path)}
            for r in FolderImageDataset(spec).records]


def dataset_identity(config: dict) -> dict:
    from src.data import get_dataset_spec

    spec = get_dataset_spec(config)
    key = digest({"name": spec.name, "root": str(spec.root.resolve()), "classes": spec.include_classes,
                  "limit": spec.limit_per_class, "manifest": config.get("manifest")})
    verified = _verified_inputs.get()
    if verified is not None and key in verified:
        return verified[key]
    records = dataset_records(spec)
    if config.get("manifest"):
        manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or manifest.get("dataset") != spec.name or manifest.get("fingerprint") != digest(manifest["records"]):
            raise ValueError("Dataset manifest identity/fingerprint is invalid")
        expected = {r["image_id"]: r for r in manifest["records"]}
        if any(expected.get(r["image_id"]) != r for r in records):
            # Labels can be reindexed when include_classes selects a subset.
            fields = ("image_id", "label_name", "sha256")
            if any({k: expected.get(r["image_id"], {}).get(k) for k in fields} != {k: r[k] for k in fields} for r in records):
                raise ValueError("Dataset content differs from dataset.manifest")
        selected = set(config.get("include_classes") or manifest["class_names"])
        names = [r["image_id"] for r in manifest["records"] if r["label_name"] in selected]
        limit = config.get("limit_per_class")
        if limit is not None:
            counts: dict[str, int] = {}
            kept = []
            for name in names:
                label = expected[name]["label_name"]
                counts[label] = counts.get(label, 0) + 1
                if counts[label] <= int(limit):
                    kept.append(name)
            names = kept
        if names != [r["image_id"] for r in records]:
            raise ValueError("Dataset membership/order differs from dataset.manifest")
    identity = {"name": spec.name, "n_images": len(records), "n_classes": len({r["label_name"] for r in records}), "fingerprint": digest(records)}
    if verified is not None:
        verified[key] = identity
    return identity


def scientific_config(config: dict) -> dict:
    result = deepcopy(config)
    result.pop("_reproducibility", None)
    experiment = result.get("experiment", {})
    experiment.pop("output_dir", None)
    result.get("dataset", {}).pop("root", None)
    result.get("dataset", {}).pop("manifest", None)
    result.get("trainer", {}).pop("resume_from_checkpoint", None)
    # Derived DEC metadata is recomputed by the runner, not an independent input.
    for name in ("target_update_mode", "head", "gloca"):
        result.get("baseline", {}).pop(name, None)
    return result


def source_identity() -> dict:
    files = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/**/*.py"), ROOT / "pyproject.toml",
                    *ROOT.glob("environment*.yml"), *ROOT.glob("requirements*.txt")])
    # Git's Windows line-ending conversion does not change the executed program.
    fingerprint = digest({p.relative_to(ROOT).as_posix(): sha256(p.read_text(encoding="utf-8").encode()).hexdigest() for p in files})
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        return {"fingerprint": fingerprint, "git_revision": result.stdout.strip() or None,
                "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}
    except FileNotFoundError:
        return {"fingerprint": fingerprint, "git_revision": None, "git_dirty": None}


def backbone_identity(config: dict) -> dict:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from huggingface_hub.utils import EntryNotFoundError
    from src.features.dinov2 import DINOV2_SMALL, DINOV2_SMALL_REVISION, DINOV2_SMALL_FILES

    variant, revision = config["variant"], config["revision"]
    key = "backbone:" + digest([variant, revision])
    verified = _verified_inputs.get()
    if verified is not None and key in verified:
        return verified[key]
    hashes = {}
    for name in ["config.json", "model.safetensors", "pytorch_model.bin"]:
        if name == "pytorch_model.bin" and "model.safetensors" in hashes:
            continue
        path = try_to_load_from_cache(variant, name, revision=revision)
        if not isinstance(path, str):
            try:
                path = hf_hub_download(variant, name, revision=revision)
            except EntryNotFoundError:
                if name == "model.safetensors":
                    continue
                raise
        hashes[name] = file_hash(Path(path))
    if variant == DINOV2_SMALL and revision == DINOV2_SMALL_REVISION and hashes != DINOV2_SMALL_FILES:
        raise ValueError("Pinned DINOv2 cache content differs from the reference hashes; restore the backbone cache")
    identity = {"variant": variant, "revision": revision, "files": hashes}
    if verified is not None:
        verified[key] = identity
    return identity


def environment_identity() -> dict:
    import torch

    return {"python": platform.python_version(), "python_build": sys.version, "platform": platform.platform(),
            "packages": {p: version(p) for p in PACKAGES}, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor(), "gpu_driver": gpu_driver(),
            "torch_threads": torch.get_num_threads(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


@lru_cache(maxsize=1)
def gpu_driver() -> str | None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def describe_run(config: dict) -> dict:
    from src.runners.common import apply_runtime_settings

    config = deepcopy(config)
    settings = apply_runtime_settings(config)
    dataset = dataset_identity(config["dataset"])
    if config["head"]["n_clusters"] == "auto":
        config["head"]["n_clusters"] = dataset["n_classes"]
    head = config["head"]["name"]
    if head == "hdbscan":
        config["head"]["n_clusters"] = "inferred"
    if config["dataset"]["training_views"] == "auto":
        config["dataset"]["training_views"] = 2 if head == "propos" else (config.get("cdc", {}).get("training_views", "cdc_weak_strong_calibration") if head == "cdc" else 1)
    source = source_identity()
    identity = {"config": scientific_config(config), "dataset": dataset,
                "backbone": backbone_identity(config["backbone"]),
                "environment": environment_identity(), "source": source["fingerprint"]}
    return {"version": 1, "fingerprint": digest(identity), **identity,
            "runtime": settings, "source_details": source}


def feature_cache_key(config: dict, source: str) -> str:
    from src.runners.common import apply_runtime_settings
    config = deepcopy(config)
    apply_runtime_settings(config)
    trainer = config["trainer"]
    return digest({"dataset": dataset_identity(config["dataset"]), "backbone": config["backbone"],
                   "backbone_files": backbone_identity(config["backbone"]),
                   "source": source, "normalized": config.get("gloca", {}).get("normalize_output", True),
                   "batch_size": trainer["batch_size"], "num_workers": trainer["num_workers"],
                   "runtime": environment_identity(), "accelerator": trainer.get("accelerator", "auto"),
                   "source_code": source_identity()["fingerprint"],
                   "deterministic": trainer["deterministic"]})


def validate_feature_cache(config: dict, cache: dict, source: str) -> None:
    if config.get("gloca", {}).get("enabled", False):
        raise ValueError("Frozen CLS/global-local caches cannot replace GLoCA embeddings")
    if "backbone" not in cache or any(p.requires_grad for p in cache["backbone"].parameters()):
        raise ValueError("Cached DINOv2 backbone must remain frozen")
    if cache.get("feature_key") != feature_cache_key(config, source):
        raise ValueError("Cached features do not match dataset/backbone/extraction settings")


@verification_session()
def is_complete(config: dict) -> bool:
    from src.utils import resolve_output_dir

    directory = resolve_output_dir(config)
    marker = directory / "completed.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    current = describe_run(config)
    if record.get("fingerprint") != current["fingerprint"]:
        raise ValueError(f"Run identity differs at {directory}; choose a fresh output directory")
    files = record.get("artifacts", {})
    if not {"config.yaml", "assignments.json", "metrics.csv", "embeddings.pt", "logs.json", "reproducibility.json"} <= files.keys():
        return False
    return all((directory / name).is_file() and file_hash(directory / name) == expected for name, expected in files.items())


def record_run(function):
    """Wrap runner orchestration; trainers remain responsible only for training."""
    @wraps(function)
    @verification_session()
    def wrapped(config: dict, *args, **kwargs):
        from src.runners.common import apply_runtime_settings
        from src.utils import resolve_output_dir

        config = deepcopy(config)
        apply_runtime_settings(config)
        directory = resolve_output_dir(config)
        metadata = describe_run(config)
        cache = args[0] if args else kwargs.get("cache")
        if cache is not None:
            source = "cls_mean_patch_concat" if function.__name__ == "run_cached_global_local_kmeans" else "cls"
            validate_feature_cache(config, cache, source)
        previous = directory / "reproducibility.json"
        strict = config["trainer"]["deterministic"]
        if strict and directory.exists() and any(directory.iterdir()):
            if not previous.exists() or json.loads(previous.read_text(encoding="utf-8")).get("fingerprint") != metadata["fingerprint"]:
                raise ValueError(f"Unverified or incompatible output directory {directory}; choose a fresh output directory")
        config["_reproducibility"] = {"fingerprint": metadata["fingerprint"]}
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "completed.json").unlink(missing_ok=True)
        metadata["invocation"] = sys.argv
        atomic_json(previous, metadata)
        result = function(config, *args, **kwargs)
        required = ["config.yaml", "assignments.json", "metrics.csv", "embeddings.pt", "logs.json"]
        head = str(config["head"]["name"])
        if head in {"cdc", "propos", "dec", "idec", "supervised_mlp"}:
            required.append("checkpoint.ckpt")
        if head == "cdc":
            required.extend(["confidence.pt", "calibrated_confidence.pt", "pseudo_labels.pt"])
        if not all((directory / name).is_file() for name in required):
            raise RuntimeError(f"Runner did not finish all required artifacts in {directory}")
        logs_path = directory / "logs.json"
        logs = json.loads(logs_path.read_text(encoding="utf-8"))
        if logs.get("attention_shape") is not None and not (directory / "attention.pt").is_file():
            raise RuntimeError("Runner did not finish attention.pt")
        logs["reproducibility"] = {"fingerprint": metadata["fingerprint"], **metadata["runtime"]}
        atomic_json(logs_path, logs)
        artifacts = {p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != "completed.json" and not p.name.endswith(".tmp")}
        atomic_json(directory / "completed.json", {"fingerprint": metadata["fingerprint"], "artifacts": artifacts})
        return result
    return wrapped


def checkpoint_identity(config: dict) -> str:
    return config.get("_reproducibility", {}).get("fingerprint") or digest(scientific_config(config))


def validate_checkpoint(payload: dict, config: dict) -> None:
    if not config.get("trainer", {}).get("deterministic", True):
        return
    if payload.get("reproducibility") != checkpoint_identity(config) or payload.get("rng_state") is None:
        raise ValueError("Checkpoint lacks matching reproducibility/RNG state; use a fresh run or trainer.deterministic=false for legacy resume")
