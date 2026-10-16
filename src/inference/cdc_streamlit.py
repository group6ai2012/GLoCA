from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F

from src.experiments.config import load_experiment_config
from src.utils import resolve_device


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_PIXELS = 50_000_000
MAX_TOP_CLUSTERS = 10


@dataclass(frozen=True)
class SupportedCDCRun:
    dataset: str
    display_name: str
    run_relative_path: Path
    dataset_relative_path: Path
    expected_n_clusters: int


SUPPORTED_CDC_DEMO_RUNS: dict[str, SupportedCDCRun] = {
    "plantseg": SupportedCDCRun(
        dataset="plantseg",
        display_name="PlantSeg",
        run_relative_path=Path(
            "outputs/main_experiments/gloca_gated/cdc/plantseg/seed_42"
        ),
        dataset_relative_path=Path("data/raw/plantseg_folder"),
        expected_n_clusters=114,
    ),
    "plantvillage": SupportedCDCRun(
        dataset="plantvillage",
        display_name="PlantVillage",
        run_relative_path=Path(
            "outputs/main_experiments/gloca_gated/cdc/plantvillage/seed_42"
        ),
        dataset_relative_path=Path("data/raw/plantvillage"),
        expected_n_clusters=38,
    ),
}


@dataclass
class CDCInferenceBundle:
    dataset: str
    display_name: str
    run_dir: Path
    dataset_root: Path
    config: dict[str, Any]
    model: Any
    device: torch.device
    transform: Callable[[Image.Image], torch.Tensor]
    n_clusters: int


@dataclass(frozen=True)
class ClusterPrediction:
    cluster_id: int
    probability: float


@dataclass(frozen=True)
class DatasetExemplar:
    cluster_id: int
    image_id: str
    path: Path
    probability: float


def get_supported_run(
    dataset: str, *, repository_root: Path = REPOSITORY_ROOT
) -> tuple[SupportedCDCRun, Path, Path]:
    key = str(dataset).strip().lower()
    if key not in SUPPORTED_CDC_DEMO_RUNS:
        accepted = ", ".join(sorted(SUPPORTED_CDC_DEMO_RUNS))
        raise ValueError(f"Unsupported demo dataset {dataset!r}. Accepted values: {accepted}")
    run = SUPPORTED_CDC_DEMO_RUNS[key]
    root = Path(repository_root).resolve()
    return run, (root / run.run_relative_path), (root / run.dataset_relative_path)


def artifact_fingerprint(
    dataset: str, *, repository_root: Path = REPOSITORY_ROOT
) -> tuple[int, ...]:
    _run, run_dir, _dataset_root = get_supported_run(
        dataset, repository_root=repository_root
    )
    paths = [
        run_dir / "config.yaml",
        run_dir / "checkpoints" / "latest.ckpt",
        run_dir / "assignments.json",
        run_dir / "embeddings.pt",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required CDC demo artifact(s) are missing: " + ", ".join(missing)
        )
    return tuple(value for path in paths for value in (path.stat().st_mtime_ns, path.stat().st_size))


def validate_demo_config(
    config: dict[str, Any], dataset: str, expected_n_clusters: int
) -> None:
    checks = {
        "dataset.name": str(config.get("dataset", {}).get("name", "")).lower()
        == dataset,
        "experiment.seed": int(config.get("experiment", {}).get("seed", -1)) == 42,
        "head.name": str(config.get("head", {}).get("name", "")).lower() == "cdc",
        "head.n_clusters": int(config.get("head", {}).get("n_clusters", -1))
        == int(expected_n_clusters),
        "gloca.enabled": bool(config.get("gloca", {}).get("enabled", False)),
        "gloca.name": str(config.get("gloca", {}).get("name", "")).lower()
        == "gloca_gated",
        "backbone.freeze": bool(config.get("backbone", {}).get("freeze", False)),
        "prediction.deterministic": bool(
            config.get("prediction", {}).get("deterministic", False)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "The selected run does not satisfy the CDC + gloca_gated seed-42 "
            f"demo contract: {', '.join(failed)}"
        )


def load_cdc_bundle(
    dataset: str,
    *,
    device: str | torch.device | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> CDCInferenceBundle:
    from src.data.transforms import build_eval_transform
    from src.features import DINOv2Backbone
    from src.models import ClusteringBaseModel, build_adapter
    from src.models.clustering import CDCHead

    run, run_dir, dataset_root = get_supported_run(
        dataset, repository_root=repository_root
    )
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "checkpoints" / "latest.ckpt"
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Original dataset root for {run.display_name} was not found: {dataset_root}"
        )
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Seed-42 CDC + gloca_gated artifacts were not found under {run_dir}"
        )

    config = load_experiment_config(config_path)
    validate_demo_config(config, run.dataset, run.expected_n_clusters)
    checkpoint = _torch_load(checkpoint_path, weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"Checkpoint has no config mapping: {checkpoint_path}")
    validate_demo_config(
        checkpoint_config, run.dataset, run.expected_n_clusters
    )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")

    backbone = DINOv2Backbone(
        variant=str(config["backbone"]["variant"]),
        image_size=int(config["backbone"]["image_size"]),
        freeze=True,
    )
    adapter = build_adapter(config, input_dim=backbone.output_dim)
    if adapter is None:
        raise ValueError("The selected demo checkpoint must contain gloca_gated.")
    embedding_dim = int(config["gloca"]["embedding_dim"])
    head = CDCHead(
        input_dim=embedding_dim,
        n_clusters=run.expected_n_clusters,
        hidden_dim=int(config["cdc"]["hidden_dim"]),
    )
    model = ClusteringBaseModel(
        backbone=backbone,
        adapter=adapter,
        head=head,
        normalize_cls=bool(config["gloca"].get("normalize_output", True)),
    )
    model.load_state_dict(state_dict, strict=True)
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("The DINOv2 backbone must remain frozen during inference.")

    selected_device = torch.device(device) if device is not None else resolve_device(config)
    model.to(selected_device)
    model.eval()
    return CDCInferenceBundle(
        dataset=run.dataset,
        display_name=run.display_name,
        run_dir=run_dir,
        dataset_root=dataset_root.resolve(),
        config=config,
        model=model,
        device=selected_device,
        transform=build_eval_transform(int(config["backbone"]["image_size"])),
        n_clusters=run.expected_n_clusters,
    )


def decode_uploaded_image(
    data: bytes, *, max_bytes: int = MAX_UPLOAD_BYTES
) -> Image.Image:
    if not data:
        raise ValueError("The uploaded image is empty.")
    if len(data) > int(max_bytes):
        raise ValueError(
            f"The uploaded image is larger than {int(max_bytes) // (1024 * 1024)} MB."
        )
    try:
        with Image.open(BytesIO(data)) as probe:
            probe.verify()
        with Image.open(BytesIO(data)) as source:
            if int(source.width) * int(source.height) > MAX_UPLOAD_PIXELS:
                raise ValueError("The uploaded image has too many pixels to process safely.")
            return source.convert("RGB").copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid supported image.") from exc


@torch.inference_mode()
def predict_top_clusters(
    bundle: CDCInferenceBundle,
    image: Image.Image,
    top_n: int,
) -> tuple[ClusterPrediction, ...]:
    top_n = int(top_n)
    allowed_max = min(MAX_TOP_CLUSTERS, int(bundle.n_clusters))
    if top_n < 1 or top_n > allowed_max:
        raise ValueError(f"top_n must be between 1 and {allowed_max}, got {top_n}")
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")

    tensor = bundle.transform(image.convert("RGB")).unsqueeze(0).to(bundle.device)
    output = bundle.model(tensor)
    probabilities = output.get("calibration_probabilities")
    if not torch.is_tensor(probabilities) or probabilities.shape != (
        1,
        bundle.n_clusters,
    ):
        shape = None if not torch.is_tensor(probabilities) else tuple(probabilities.shape)
        raise ValueError(
            "CDC calibration head returned an invalid probability shape: "
            f"expected {(1, bundle.n_clusters)}, got {shape}"
        )
    values = probabilities[0].detach().float().cpu()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("CDC calibration probabilities contain non-finite values.")
    ranked = sorted(
        range(bundle.n_clusters), key=lambda cluster_id: (-float(values[cluster_id]), cluster_id)
    )[:top_n]
    return tuple(
        ClusterPrediction(cluster_id=cluster_id, probability=float(values[cluster_id]))
        for cluster_id in ranked
    )


@torch.inference_mode()
def build_exemplar_index(
    bundle: CDCInferenceBundle,
    *,
    examples_per_cluster: int = 5,
    batch_size: int = 2048,
) -> dict[int, tuple[DatasetExemplar, ...]]:
    examples_per_cluster = int(examples_per_cluster)
    batch_size = int(batch_size)
    if examples_per_cluster <= 0:
        raise ValueError("examples_per_cluster must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    assignments_path = bundle.run_dir / "assignments.json"
    embeddings_path = bundle.run_dir / "embeddings.pt"
    if not assignments_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError(
            f"assignments.json and embeddings.pt are required under {bundle.run_dir}"
        )
    with assignments_path.open("r", encoding="utf-8") as handle:
        assignments = json.load(handle)
    _validate_assignment_metadata(assignments, bundle)
    image_ids = assignments.get("image_ids")
    if not isinstance(image_ids, list) or not all(
        isinstance(image_id, str) for image_id in image_ids
    ):
        raise ValueError("assignments.json image_ids must be a list of strings")

    embeddings = _torch_load(embeddings_path, weights_only=True)
    if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
        raise ValueError("embeddings.pt must contain a two-dimensional tensor")
    expected_dim = int(bundle.config["gloca"]["embedding_dim"])
    if embeddings.shape != (len(image_ids), expected_dim):
        raise ValueError(
            "Stored embeddings and assignment image IDs are not aligned: "
            f"embeddings={tuple(embeddings.shape)}, image_ids={len(image_ids)}, "
            f"expected_dim={expected_dim}"
        )
    if len(image_ids) < examples_per_cluster:
        raise ValueError(
            f"The dataset has fewer than {examples_per_cluster} images."
        )

    probability_parts: list[torch.Tensor] = []
    for start in range(0, len(image_ids), batch_size):
        batch = embeddings[start : start + batch_size].to(
            bundle.device, dtype=torch.float32
        )
        logits = bundle.model.head.calibration_head(batch)
        probability_parts.append(F.softmax(logits, dim=-1).detach().cpu())
    probabilities = torch.cat(probability_parts, dim=0)
    if probabilities.shape != (len(image_ids), bundle.n_clusters):
        raise ValueError(
            "CDC calibration head returned invalid exemplar probabilities: "
            f"{tuple(probabilities.shape)}"
        )
    return rank_cluster_exemplars(
        probabilities,
        image_ids,
        bundle.dataset_root,
        examples_per_cluster=examples_per_cluster,
    )


def rank_cluster_exemplars(
    probabilities: torch.Tensor,
    image_ids: list[str],
    dataset_root: Path,
    *,
    examples_per_cluster: int = 5,
) -> dict[int, tuple[DatasetExemplar, ...]]:
    if not torch.is_tensor(probabilities) or probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, K]")
    if probabilities.shape[0] != len(image_ids):
        raise ValueError("probability rows and image_ids must have equal lengths")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("exemplar probabilities contain non-finite values")
    examples_per_cluster = int(examples_per_cluster)
    if examples_per_cluster <= 0 or examples_per_cluster > len(image_ids):
        raise ValueError("examples_per_cluster must be positive and no larger than the dataset")

    root = Path(dataset_root).resolve()
    result: dict[int, tuple[DatasetExemplar, ...]] = {}
    for cluster_id in range(int(probabilities.shape[1])):
        ranked_indices = sorted(
            range(len(image_ids)),
            key=lambda index: (
                -float(probabilities[index, cluster_id]),
                image_ids[index],
            ),
        )[:examples_per_cluster]
        exemplars = []
        for index in ranked_indices:
            path = _resolve_dataset_image(root, image_ids[index])
            exemplars.append(
                DatasetExemplar(
                    cluster_id=cluster_id,
                    image_id=image_ids[index],
                    path=path,
                    probability=float(probabilities[index, cluster_id]),
                )
            )
        result[cluster_id] = tuple(exemplars)
    return result


def _validate_assignment_metadata(
    assignments: dict[str, Any], bundle: CDCInferenceBundle
) -> None:
    expected = {
        "dataset": bundle.dataset,
        "seed": 42,
        "head": "cdc",
        "gloca": "gloca_gated",
        "n_clusters": bundle.n_clusters,
    }
    mismatches = [
        key
        for key, value in expected.items()
        if assignments.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "assignments.json does not match the selected demo run: "
            + ", ".join(mismatches)
        )


def _resolve_dataset_image(dataset_root: Path, image_id: str) -> Path:
    relative = PurePosixPath(image_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe dataset image ID: {image_id!r}")
    candidate = (dataset_root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"Dataset image escapes its root: {image_id!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Dataset exemplar was not found: {candidate}")
    return candidate


def _torch_load(path: Path, *, weights_only: bool) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location="cpu")
