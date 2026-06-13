from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_IMAGE_SIZE = 224
DATASET_NAMES = ("plantseg", "plantvillage", "plantwild")


@dataclass(frozen=True)
class AttentionSample:
    index: int
    image_id: str
    label: int
    label_name: str
    assignment: int


@dataclass(frozen=True)
class EvalCropGeometry:
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize saved GLoCA patch-pooling attention on original dataset images."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory with attention.pt and assignments.json.")
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--n-per-class", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--cmap", type=str, default="jet")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_attention(path: Path) -> np.ndarray:
    import torch

    attention = torch.load(path, map_location="cpu")
    if not hasattr(attention, "detach"):
        raise TypeError(f"Expected {path} to contain a torch.Tensor, got {type(attention).__name__}")
    return attention.detach().cpu().float().numpy()


def dataset_root_for(name: str) -> Path:
    from src.data.registry import DATASET_ROOTS

    return DATASET_ROOTS[name]


def image_size_from_config(run_dir: Path) -> int:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return DEFAULT_IMAGE_SIZE
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return DEFAULT_IMAGE_SIZE
    return int(config.get("backbone", {}).get("image_size") or DEFAULT_IMAGE_SIZE)


def validate_run_inputs(
    *,
    run_dir: Path,
    dataset: str,
    assignments: dict[str, Any],
    attention: np.ndarray,
) -> tuple[list[str], list[int], list[int], tuple[int, int]]:
    attention_path = run_dir / "attention.pt"
    assignments_path = run_dir / "assignments.json"
    if not attention_path.exists():
        raise FileNotFoundError(f"Missing attention file: {attention_path}")
    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing assignments file: {assignments_path}")
    if attention.ndim != 2:
        raise ValueError(f"Expected attention to have shape [B, N], got {attention.shape}")

    run_dataset = str(assignments.get("dataset", "")).lower()
    if run_dataset != dataset:
        raise ValueError(f"Run dataset is '{run_dataset}', but --dataset is '{dataset}'.")

    image_ids = list(assignments.get("image_ids", []))
    labels = [int(label) for label in assignments.get("labels", [])]
    cluster_assignments = [int(value) for value in assignments.get("assignments", [])]
    if attention.shape[0] != len(image_ids):
        raise ValueError(
            f"attention rows ({attention.shape[0]}) must match image_ids ({len(image_ids)})."
        )
    if not (len(image_ids) == len(labels) == len(cluster_assignments)):
        raise ValueError(
            "assignments.json fields image_ids, labels, and assignments must have equal length."
        )

    patch_grid_raw = assignments.get("patch_grid")
    if not isinstance(patch_grid_raw, list | tuple) or len(patch_grid_raw) != 2:
        raise ValueError(f"Expected patch_grid to be a 2-item sequence, got {patch_grid_raw!r}")
    patch_grid = (int(patch_grid_raw[0]), int(patch_grid_raw[1]))
    expected_patches = patch_grid[0] * patch_grid[1]
    if attention.shape[1] != expected_patches:
        raise ValueError(
            f"attention columns ({attention.shape[1]}) must equal patch_grid product ({expected_patches})."
        )
    return image_ids, labels, cluster_assignments, patch_grid


def infer_label_names(image_ids: Iterable[str], labels: Iterable[int]) -> dict[int, str]:
    names: dict[int, str] = {}
    for image_id, label in zip(image_ids, labels):
        parts = Path(image_id).parts
        if not parts:
            continue
        names.setdefault(int(label), parts[0])
    return names


def build_samples(
    image_ids: list[str],
    labels: list[int],
    assignments: list[int],
) -> list[AttentionSample]:
    label_names = infer_label_names(image_ids, labels)
    return [
        AttentionSample(
            index=index,
            image_id=image_id,
            label=int(label),
            label_name=label_names.get(int(label), f"label_{int(label)}"),
            assignment=int(assignment),
        )
        for index, (image_id, label, assignment) in enumerate(zip(image_ids, labels, assignments))
    ]


def select_per_class(samples: list[AttentionSample], n_per_class: int, seed: int) -> list[AttentionSample]:
    if n_per_class <= 0:
        raise ValueError(f"--n-per-class must be positive, got {n_per_class}")
    grouped: dict[int, list[AttentionSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.label, []).append(sample)

    rng = random.Random(seed)
    selected: list[AttentionSample] = []
    for label in sorted(grouped):
        candidates = list(grouped[label])
        rng.shuffle(candidates)
        selected.extend(sorted(candidates[:n_per_class], key=lambda sample: sample.index))
    return selected


def attention_to_grid(attention_row: np.ndarray, patch_grid: tuple[int, int]) -> np.ndarray:
    row = np.asarray(attention_row, dtype=np.float32).reshape(-1)
    expected = int(patch_grid[0]) * int(patch_grid[1])
    if row.shape[0] != expected:
        raise ValueError(f"Attention row length {row.shape[0]} does not match patch_grid product {expected}.")
    return row.reshape(int(patch_grid[0]), int(patch_grid[1]))


def resize_like_eval_transform(original_size: tuple[int, int], image_size: int) -> EvalCropGeometry:
    width, height = original_size
    resize_size = int(round(image_size * 256 / 224))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {original_size}")
    if width < height:
        resized_width = resize_size
        resized_height = int(resize_size * height / width)
    else:
        resized_height = resize_size
        resized_width = int(resize_size * width / height)

    left = max(0, (resized_width - image_size) // 2)
    top = max(0, (resized_height - image_size) // 2)
    crop_box = (left, top, left + image_size, top + image_size)
    return EvalCropGeometry(
        original_size=(width, height),
        resized_size=(resized_width, resized_height),
        crop_box=crop_box,
    )


def project_attention_to_original_canvas(
    attention_grid: np.ndarray,
    original_size: tuple[int, int],
    image_size: int,
) -> np.ndarray:
    geometry = resize_like_eval_transform(original_size, image_size)
    crop_heatmap = Image.fromarray(attention_grid.astype(np.float32), mode="F").resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )
    resized_canvas = Image.new("F", geometry.resized_size, color=0.0)
    resized_canvas.paste(crop_heatmap, geometry.crop_box[:2])
    original_canvas = resized_canvas.resize(original_size, resample=Image.Resampling.BILINEAR)
    return np.asarray(original_canvas, dtype=np.float32)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    finite = np.isfinite(heatmap)
    if not bool(finite.any()):
        return np.zeros_like(heatmap, dtype=np.float32)
    valid = heatmap[finite]
    min_value = float(valid.min())
    max_value = float(valid.max())
    if max_value <= min_value:
        return np.zeros_like(heatmap, dtype=np.float32)
    normalized = (heatmap - min_value) / (max_value - min_value)
    return np.where(finite, normalized, 0.0).astype(np.float32)


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    *,
    alpha: float,
    cmap_name: str,
) -> Image.Image:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {alpha}")
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    normalized = normalize_heatmap(heatmap)
    colored = colorize_heatmap(normalized, cmap_name)
    blended = (1.0 - alpha) * base + alpha * colored
    return Image.fromarray(np.clip(blended * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def colorize_heatmap(normalized: np.ndarray, cmap_name: str) -> np.ndarray:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import colormaps

        return colormaps[cmap_name](normalized)[..., :3].astype(np.float32)
    except ModuleNotFoundError:
        lowered = cmap_name.lower()
        if lowered == "jet":
            return fallback_jet_colormap(normalized)
        if lowered in {"gray", "grey"}:
            return np.repeat(normalized[..., None], repeats=3, axis=-1).astype(np.float32)
        raise ModuleNotFoundError(
            f"Matplotlib is required for cmap '{cmap_name}'. Install matplotlib or use --cmap jet."
        )


def fallback_jet_colormap(normalized: np.ndarray) -> np.ndarray:
    x = np.clip(normalized.astype(np.float32), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1).astype(np.float32)


def safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "unnamed"


def output_path_for_sample(output_dir: Path, sample: AttentionSample) -> Path:
    image_stem = safe_path_part(Path(sample.image_id).stem)
    class_dir = output_dir / safe_path_part(sample.label_name)
    return class_dir / f"{image_stem}__idx_{sample.index}__cluster_{sample.assignment}.png"


def visualize_sample(
    *,
    sample: AttentionSample,
    attention_row: np.ndarray,
    dataset_root: Path,
    output_dir: Path,
    patch_grid: tuple[int, int],
    image_size: int,
    alpha: float,
    cmap_name: str,
) -> Path:
    image_path = dataset_root / sample.image_id
    if not image_path.exists():
        raise FileNotFoundError(f"Image for {sample.image_id!r} was not found at {image_path}")

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    attention_grid = attention_to_grid(attention_row, patch_grid)
    heatmap = project_attention_to_original_canvas(attention_grid, image.size, image_size)
    overlay = overlay_heatmap(image, heatmap, alpha=alpha, cmap_name=cmap_name)

    path = output_path_for_sample(output_dir, sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(path)
    return path


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "attention_visualizations"
    attention_path = run_dir / "attention.pt"
    assignments_path = run_dir / "assignments.json"

    if not attention_path.exists():
        raise FileNotFoundError(f"Missing attention file: {attention_path}")
    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing assignments file: {assignments_path}")

    assignments = load_json(assignments_path)
    attention = load_attention(attention_path)
    image_ids, labels, cluster_assignments, patch_grid = validate_run_inputs(
        run_dir=run_dir,
        dataset=args.dataset,
        assignments=assignments,
        attention=attention,
    )
    samples = build_samples(image_ids, labels, cluster_assignments)
    selected = select_per_class(samples, n_per_class=args.n_per_class, seed=args.seed)
    dataset_root = dataset_root_for(args.dataset)
    image_size = image_size_from_config(run_dir)

    written: list[Path] = []
    for sample in selected:
        written.append(
            visualize_sample(
                sample=sample,
                attention_row=attention[sample.index],
                dataset_root=dataset_root,
                output_dir=output_dir,
                patch_grid=patch_grid,
                image_size=image_size,
                alpha=float(args.alpha),
                cmap_name=args.cmap,
            )
        )

    print(f"Wrote {len(written)} GLoCA attention visualizations to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
