from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.folder_dataset import DatasetSpec, FolderImageDataset
from src.data.transforms import build_eval_transform, build_train_transform


class ClusteringDataModule:
    def __init__(
        self,
        spec: DatasetSpec,
        image_size: int,
        batch_size: int,
        num_workers: int,
        training_mode: str,
        training_views: int | str = "auto",
    ) -> None:
        if training_mode not in {"single_view", "contrastive_two_view"}:
            raise ValueError(
                f"Unsupported training_mode '{training_mode}'. Accepted values: single_view, contrastive_two_view"
            )
        if training_views == "auto":
            training_views = 1 if training_mode == "single_view" else 2
        if training_views not in {1, 2}:
            raise ValueError(f"training_views must resolve to 1 or 2, got {training_views}")
        self.spec = spec
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.training_mode = training_mode
        self.training_views = int(training_views)
        self.transform_backend = "pil"
        self.augmentation_name = "pil_default"
        self.train_dataset: FolderImageDataset | None = None
        self.predict_dataset: FolderImageDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is None:
            self.train_dataset = FolderImageDataset(
                self.spec,
                transform=build_train_transform(self.image_size),
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate,
        )

    def train_eval_dataloader(self) -> DataLoader:
        self.setup("predict")
        assert self.predict_dataset is not None
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.train_eval_dataloader()

    def data_loading_info(self) -> dict[str, Any]:
        return {
            "transform_backend": self.transform_backend,
            "augmentation_name": self.augmentation_name,
        }

    @staticmethod
    def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "index": torch.tensor([sample["index"] for sample in samples], dtype=torch.long),
            "label": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
            "label_name": [sample["label_name"] for sample in samples],
            "image_id": [sample["image_id"] for sample in samples],
            "dataset": [sample["dataset"] for sample in samples],
        }
        if "views" in samples[0]:
            view1 = torch.stack([sample["views"][0] for sample in samples])
            view2 = torch.stack([sample["views"][1] for sample in samples])
            batch["views"] = (view1, view2)
        else:
            batch["image"] = torch.stack([sample["image"] for sample in samples])
        return batch
