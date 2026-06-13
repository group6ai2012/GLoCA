from __future__ import annotations

from pathlib import Path

from src.data.folder_dataset import DatasetSpec


DATASET_ROOTS = {
    "plantseg": Path("data/raw/plantseg_folder"),
    "plantvillage": Path("data/raw/plantvillage"),
    "plantwild": Path("data/raw/plantwild_v2"),
}


def get_dataset_spec(config: dict) -> DatasetSpec:
    name = str(config.get("name", "")).lower()
    if name not in DATASET_ROOTS:
        accepted = ", ".join(sorted(DATASET_ROOTS))
        raise ValueError(f"Unknown dataset '{name}'. Accepted values: {accepted}")
    root = Path(config.get("root") or DATASET_ROOTS[name])
    if name == "plantseg" and not root.exists():
        raise FileNotFoundError(
            "PlantSeg folder dataset was not found at "
            f"{root}. Run: python scripts/prepare_plantseg_folder_dataset.py --source-root <raw_plantseg_root>"
        )
    return DatasetSpec(
        name=name,
        root=root,
        split_file=config.get("split_file"),
        limit_per_class=config.get("limit_per_class"),
        include_classes=tuple(config["include_classes"]) if config.get("include_classes") else None,
    )
