__all__ = ["ClusteringDataModule", "DatasetSpec", "FolderImageDataset", "get_dataset_spec"]


def __getattr__(name: str):
    if name == "ClusteringDataModule":
        from src.data.datamodule import ClusteringDataModule

        return ClusteringDataModule
    if name in {"DatasetSpec", "FolderImageDataset"}:
        from src.data.folder_dataset import DatasetSpec, FolderImageDataset

        return {"DatasetSpec": DatasetSpec, "FolderImageDataset": FolderImageDataset}[name]
    if name == "get_dataset_spec":
        from src.data.registry import get_dataset_spec

        return get_dataset_spec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
