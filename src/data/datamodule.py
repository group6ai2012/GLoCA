from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.folder_dataset import DatasetSpec, FolderImageDataset
from src.data.transforms import (
    build_cdc_transforms,
    build_eval_transform,
    build_train_transform,
)


class ClusteringDataModule:
    def __init__(
        self,
        spec: DatasetSpec,
        image_size: int,
        batch_size: int,
        num_workers: int,
        training_mode: str,
        training_views: int | str = "auto",
        cdc_augment_config: dict[str, Any] | None = None,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = None,
    ) -> None:
        if training_mode not in {"single_view", "contrastive_two_view", "cdc"}:
            raise ValueError(
                "Unsupported training_mode "
                f"'{training_mode}'. Accepted values: single_view, contrastive_two_view, cdc"
            )
        if training_views == "auto":
            training_views = (
                "cdc_weak_strong_calibration"
                if training_mode == "cdc"
                else (1 if training_mode == "single_view" else 2)
            )
        if training_views not in {1, 2, "cdc_weak_strong_calibration"}:
            raise ValueError(
                "training_views must resolve to 1, 2, or 'cdc_weak_strong_calibration', "
                f"got {training_views}"
            )
        self.spec = spec
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.training_mode = training_mode
        self.training_views = training_views
        self.cdc_augment_config = cdc_augment_config or {}
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.transform_backend = "pil"
        self.augmentation_name = (
            "cdc_weak_strong_calibration"
            if training_views == "cdc_weak_strong_calibration"
            else "pil_default"
        )
        self.train_dataset: FolderImageDataset | None = None
        self.predict_dataset: FolderImageDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is None:
            train_transform = (
                build_cdc_transforms(self.image_size, self.cdc_augment_config)
                if self.training_views == "cdc_weak_strong_calibration"
                else build_train_transform(self.image_size)
            )
            self.train_dataset = FolderImageDataset(
                self.spec,
                transform=train_transform,
                training_views=self.training_views,
            )
        if self.predict_dataset is None:
            self.predict_dataset = FolderImageDataset(
                self.spec,
                transform=build_eval_transform(self.image_size),
                training_views=1,
            )

    @property
    def n_clusters(self) -> int:
        self.setup()
        assert self.train_dataset is not None
        return self.train_dataset.n_classes

    @property
    def num_samples(self) -> int:
        self.setup()
        assert self.train_dataset is not None
        return len(self.train_dataset)

    def train_dataloader(self) -> DataLoader:
        self.setup("fit")
        assert self.train_dataset is not None
        kwargs = self._loader_kwargs(shuffle=True, pin_memory_default=torch.cuda.is_available())
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate,
            **kwargs,
        )

    def train_eval_dataloader(self) -> DataLoader:
        self.setup("predict")
        assert self.predict_dataset is not None
        kwargs = self._loader_kwargs(shuffle=False, pin_memory_default=False)
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate,
            **kwargs,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.train_eval_dataloader()

    def data_loading_info(self) -> dict[str, Any]:
        return {
            "transform_backend": self.transform_backend,
            "augmentation_name": self.augmentation_name,
            "pin_memory": self.pin_memory if self.pin_memory is not None else torch.cuda.is_available(),
            "persistent_workers": (
                self.persistent_workers
                if self.persistent_workers is not None
                else self.num_workers > 0
            ),
            "prefetch_factor": self.prefetch_factor if self.num_workers > 0 else None,
        }

    def _loader_kwargs(self, *, shuffle: bool, pin_memory_default: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": pin_memory_default if self.pin_memory is None else bool(self.pin_memory),
            "persistent_workers": (
                self.num_workers > 0
                and (
                    bool(self.persistent_workers)
                    if self.persistent_workers is not None
                    else True
                )
            ),
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return kwargs

    @staticmethod
    def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "index": torch.tensor(
                [sample["index"] for sample in samples], dtype=torch.long
            ),
            "label": torch.tensor(
                [sample["label"] for sample in samples], dtype=torch.long
            ),
            "label_name": [sample["label_name"] for sample in samples],
            "image_id": [sample["image_id"] for sample in samples],
            "dataset": [sample["dataset"] for sample in samples],
        }
        if "views" in samples[0]:
            view1 = torch.stack([sample["views"][0] for sample in samples])
            view2 = torch.stack([sample["views"][1] for sample in samples])
            batch["views"] = (view1, view2)
        if "weak" in samples[0]:
            batch["weak"] = torch.stack([sample["weak"] for sample in samples])
            batch["strong"] = torch.stack([sample["strong"] for sample in samples])
            batch["calibration"] = torch.stack(
                [sample["calibration"] for sample in samples]
            )
        elif "views" not in samples[0]:
            batch["image"] = torch.stack([sample["image"] for sample in samples])
        return batch
