from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_HINTS = {"mask", "masks", "annotation", "annotations", "label", "labels", "seg", "segmentation"}
ILLEGAL_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class PlantSegMetadata:
    name: str
    index: str
    plant: str
    disease: str
    label_file: str
    split: str
    class_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw PlantSeg images into folder-per-class layout.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/plantseg_folder"))
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Path to PlantSeg Metadatav2.csv. Defaults to source_root/Metadatav2.csv, then source_root.parent/Metadatav2.csv.",
    )
    parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root
    output_root = args.output_root
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"PlantSeg source root does not exist or is not a directory: {source_root}")
    if output_root.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output root already exists: {output_root}. Pass --overwrite to add/replace files.")

    metadata_csv = _resolve_metadata_csv(source_root, args.metadata_csv)
    metadata_by_name = _load_metadata(metadata_csv)

    rows = []
    examples = []
    counts: Counter[str] = Counter()
    for source_path in _iter_images(source_root):
        metadata = metadata_by_name.get(source_path.name)
        if metadata is None:
            raise KeyError(f"Image is missing from PlantSeg metadata: {source_path.name} ({source_path})")
        inferred_class = metadata.class_name
        target_path = output_root / inferred_class / source_path.name
        split_or_source_folder = source_path.parent.relative_to(source_root).as_posix()
        rows.append(
            {
                "source_path": str(source_path),
                "target_path": str(target_path),
                "inferred_class": inferred_class,
                "metadata_index": metadata.index,
                "metadata_plant": metadata.plant,
                "metadata_disease": metadata.disease,
                "metadata_split": metadata.split,
                "metadata_label_file": metadata.label_file,
                "split_or_source_folder": split_or_source_folder,
                "copied": False,
            }
        )
        counts[inferred_class] += 1
        if len(examples) < 20:
            examples.append((source_path.name, inferred_class, metadata.disease))

    print(f"Loaded PlantSeg metadata from: {metadata_csv}")
    print("First filename -> metadata class examples:")
    for filename, inferred_class, disease in examples:
        print(f"  {filename} -> {inferred_class} ({disease})")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        for row in rows:
            source_path = Path(row["source_path"])
            target_path = Path(row["target_path"])
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                if not args.overwrite:
                    raise FileExistsError(f"Target already exists: {target_path}. Pass --overwrite to replace it.")
                target_path.unlink()
            _copy_one(source_path, target_path, args.copy_mode)
            row["copied"] = True
        _write_manifest(output_root / "_plantseg_conversion_manifest.csv", rows)
        _write_class_counts(output_root / "_class_counts.csv", counts)

    print("Class counts:")
    for class_name, count in sorted(counts.items()):
        print(f"  {class_name}: {count}")
    if counts:
        unknown_count = counts.get("unknown", 0)
        if unknown_count / sum(counts.values()) > 0.2:
            print(f"WARNING: {unknown_count} images were assigned to unknown.")
        if len(counts) == 1:
            print("WARNING: only one class was inferred.")
    print(f"{'Would process' if args.dry_run else 'Processed'} {sum(counts.values())} images.")


def _resolve_metadata_csv(source_root: Path, metadata_csv: Path | None) -> Path:
    candidates = [metadata_csv] if metadata_csv is not None else [source_root / "Metadatav2.csv", source_root.parent / "Metadatav2.csv"]
    for candidate in candidates:
        if candidate is not None and candidate.exists() and candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates if candidate is not None)
    raise FileNotFoundError(f"PlantSeg metadata CSV was not found. Checked: {searched}")


def _load_metadata(metadata_csv: Path) -> dict[str, PlantSegMetadata]:
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_metadata_columns(reader.fieldnames, metadata_csv)
        metadata_by_name: dict[str, PlantSegMetadata] = {}
        disease_by_index: dict[str, str] = {}
        class_by_index: dict[str, str] = {}
        index_by_class: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            name = row["Name"].strip()
            index = row["Index"].strip()
            plant = row["Plant"].strip()
            disease = row["Disease"].strip()
            label_file = row["Label file"].strip()
            split = row["Split"].strip()
            if not name:
                raise ValueError(f"Missing metadata Name at {metadata_csv}:{row_number}")
            if not index:
                raise ValueError(f"Missing metadata Index for {name} at {metadata_csv}:{row_number}")
            if not disease:
                raise ValueError(f"Missing metadata Disease for {name} at {metadata_csv}:{row_number}")
            if name in metadata_by_name:
                raise ValueError(f"Duplicate metadata Name '{name}' at {metadata_csv}:{row_number}")

            class_name = _sanitize_class_name(disease)
            previous_disease = disease_by_index.setdefault(index, disease)
            previous_class = class_by_index.setdefault(index, class_name)
            previous_index = index_by_class.setdefault(class_name, index)
            if previous_disease != disease:
                raise ValueError(
                    f"Conflicting Disease values for metadata Index {index}: '{previous_disease}' and '{disease}'"
                )
            if previous_class != class_name:
                raise ValueError(
                    f"Conflicting class folders for metadata Index {index}: '{previous_class}' and '{class_name}'"
                )
            if previous_index != index:
                raise ValueError(
                    f"Class folder '{class_name}' is shared by metadata indexes {previous_index} and {index}"
                )

            metadata_by_name[name] = PlantSegMetadata(
                name=name,
                index=index,
                plant=plant,
                disease=disease,
                label_file=label_file,
                split=split,
                class_name=class_name,
            )
    if not metadata_by_name:
        raise ValueError(f"PlantSeg metadata CSV has no rows: {metadata_csv}")
    return metadata_by_name


def _validate_metadata_columns(fieldnames: list[str] | None, metadata_csv: Path) -> None:
    required = {"Name", "Index", "Plant", "Disease", "Label file", "Split"}
    missing = required - set(fieldnames or [])
    if missing:
        raise ValueError(f"PlantSeg metadata CSV is missing required columns {sorted(missing)}: {metadata_csv}")


def _sanitize_class_name(value: str) -> str:
    value = value.strip()
    value = ILLEGAL_PATH_CHARS.sub("", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[-_]+", "_", value)
    return value.strip("._-") or "unknown"


def _iter_images(source_root: Path):
    for path in sorted(source_root.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = {part.lower() for part in path.parts}
        if parts & MASK_HINTS:
            continue
        lower_stem = path.stem.lower()
        if any(re.search(rf"(^|[_\-\s]){hint}([_\-\s]|$)", lower_stem) for hint in MASK_HINTS):
            continue
        yield path


def _copy_one(source_path: Path, target_path: Path, copy_mode: str) -> None:
    if copy_mode == "copy":
        shutil.copy2(source_path, target_path)
    elif copy_mode == "hardlink":
        target_path.hardlink_to(source_path)
    elif copy_mode == "symlink":
        target_path.symlink_to(source_path.resolve())
    else:
        raise ValueError(f"Unsupported copy mode: {copy_mode}")


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "target_path",
                "inferred_class",
                "metadata_index",
                "metadata_plant",
                "metadata_disease",
                "metadata_split",
                "metadata_label_file",
                "split_or_source_folder",
                "copied",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_class_counts(path: Path, counts: Counter[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "n_images"])
        writer.writeheader()
        for class_name, count in sorted(counts.items()):
            writer.writerow({"class_name": class_name, "n_images": count})


if __name__ == "__main__":
    main()
