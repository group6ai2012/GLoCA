from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from src.features import DINOv2Backbone

if TYPE_CHECKING:
    from src.data import ClusteringDataModule


def extract_deterministic_embeddings(
    backbone: DINOv2Backbone,
    adapter: nn.Module | None,
    datamodule: ClusteringDataModule,
    device: torch.device,
    normalize_cls: bool = True,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    backbone.to(device)
    backbone.eval()
    if adapter is not None:
        adapter.to(device)
        adapter.eval()

    embedding_parts: list[torch.Tensor] = []
    attention_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []
    image_ids: list[str] = []
    patch_grid = None

    with torch.no_grad():
        for batch in datamodule.predict_dataloader():
            image = batch["image"].to(device)
            backbone_out = backbone(image)
            if adapter is None:
                embeddings = backbone_out["cls"]
                if normalize_cls:
                    embeddings = F.normalize(embeddings, dim=-1)
                attention = None
            else:
                adapter_out = adapter(
                    cls=backbone_out["cls"],
                    patch_tokens=backbone_out["patch_tokens"],
                    patch_grid=backbone_out["patch_grid"],
                )
                embeddings = adapter_out["embedding"]
                attention = adapter_out["attention"]

            embedding_parts.append(embeddings.detach().cpu())
            if attention is not None:
                attention_parts.append(attention.detach().cpu())
            label_parts.append(batch["label"].detach().cpu())
            index_parts.append(batch["index"].detach().cpu())
            image_ids.extend(batch["image_id"])
            patch_grid = backbone_out["patch_grid"]

    indices = torch.cat(index_parts, dim=0).long()
    order = torch.argsort(indices)
    ordered_ids = [image_ids[i] for i in order.tolist()]
    attention_tensor = torch.cat(attention_parts, dim=0)[order].contiguous() if attention_parts else None
    return {
        "embeddings": torch.cat(embedding_parts, dim=0)[order].contiguous(),
        "attention": attention_tensor,
        "labels": torch.cat(label_parts, dim=0)[order].long().contiguous(),
        "image_ids": ordered_ids,
        "indices": indices[order].contiguous(),
        "patch_grid": patch_grid,
        "cache_time_s": time.perf_counter() - start_time,
    }
