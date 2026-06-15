from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    split_file: Path | None = None
    limit_per_class: int | None = None
    include_classes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Dataset root for '{self.name}' does not exist or is not a directory: {root}")
        if self.limit_per_class is not None and self.limit_per_class <= 0:
            raise ValueError(f"limit_per_class must be positive or null, got {self.limit_per_class}")
        object.__setattr__(self, "root", root)
        if self.split_file is not None:
            object.__setattr__(self, "split_file", Path(self.split_file))


@dataclass(frozen=True)
class SampleRecord:
    index: int
    path: Path
    label: int
    label_name: str
    image_id: str
    dataset: str


class FolderImageDataset(Dataset):
    def __init__(
        self,
        spec: DatasetSpec,
        transform: Callable | dict[str, Callable] | None = None,
        training_views: int | str = 1,
    ) -> None:
        if training_views not in {1, 2, "cdc_weak_strong_calibration"}:
            raise ValueError(
                "training_views must be 1, 2, or 'cdc_weak_strong_calibration', "
                f"got {training_views}"
            )
        self.spec = spec
        self.transform = transform
        self.training_views = training_views
        self.records = self._build_records()
        if not self.records:
            raise ValueError(f"Dataset '{spec.name}' has no images under {spec.root}")
        self.class_names = sorted({record.label_name for record in self.records})
        self.n_classes = len(self.class_names)

    def _build_records(self) -> list[SampleRecord]:
        class_dirs = [p for p in self.spec.root.iterdir() if p.is_dir()]
        class_dirs = sorted(class_dirs, key=lambda p: p.name.lower())
        if self.spec.include_classes is not None:
            allowed = set(self.spec.include_classes)
            class_dirs = [p for p in class_dirs if p.name in allowed]
        if not class_dirs:
            raise ValueError(f"Dataset '{self.spec.name}' has no class directories under {self.spec.root}")

        records: list[SampleRecord] = []
        label_by_name = {path.name: i for i, path in enumerate(class_dirs)}
        for class_dir in class_dirs:
            image_paths = [
                path
                for path in sorted(class_dir.rglob("*"), key=lambda p: str(p).lower())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if self.spec.limit_per_class is not None:
                image_paths = image_paths[: self.spec.limit_per_class]
            for path in image_paths:
                image_id = path.relative_to(self.spec.root).as_posix()
                records.append(
                    SampleRecord(
                        index=len(records),
                        path=path,
                        label=label_by_name[class_dir.name],
                        label_name=class_dir.name,
                        image_id=image_id,
                        dataset=self.spec.name,
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            if self.training_views == "cdc_weak_strong_calibration":
                if not isinstance(self.transform, dict):
                    raise TypeError("CDC training views require a transform dictionary.")
                weak = self.transform["weak"](image)
                calibration = self.transform["calibration"](image)
                strong = self.transform["strong"](image)
                payload = {
                    "weak": weak,
                    "strong": strong,
                    "calibration": calibration,
                    "views": (weak, strong),
                }
            elif self.training_views == 2:
                if self.transform is None:
                    view1 = image.copy()
                    view2 = image.copy()
                else:
                    if isinstance(self.transform, dict):
                        raise TypeError("Two-view training expects a single transform callable.")
                    view1 = self.transform(image)
                    view2 = self.transform(image)
                payload = {"views": (view1, view2)}
            else:
                if isinstance(self.transform, dict):
                    raise TypeError("Single-view training expects a single transform callable.")
                payload = {"image": self.transform(image) if self.transform is not None else image.copy()}

        payload.update(
            {
                "index": record.index,
                "label": record.label,
                "label_name": record.label_name,
                "image_id": record.image_id,
                "dataset": record.dataset,
            }
        )
        return payload
