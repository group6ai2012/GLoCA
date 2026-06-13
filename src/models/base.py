from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.features.dinov2_backbone import DINOv2Backbone


class ClusteringBaseModel(nn.Module):
    def __init__(
        self,
        backbone: DINOv2Backbone,
        adapter: nn.Module | None,
        head: nn.Module,
        normalize_cls: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.gloca = adapter
        self.head = head
        self.normalize_cls = normalize_cls

    def encode_view(self, image: torch.Tensor) -> dict:
        backbone_out = self.backbone(image)
        if self.adapter is None:
            embedding = backbone_out["cls"]
            if self.normalize_cls:
                embedding = F.normalize(embedding, dim=-1)
            return {
                "embedding": embedding,
                "attention": None,
                "patch_grid": backbone_out["patch_grid"],
            }
        return self.adapter(
            cls=backbone_out["cls"],
            patch_tokens=backbone_out["patch_tokens"],
            patch_grid=backbone_out["patch_grid"],
        )

    def forward(self, image: torch.Tensor) -> dict:
        encoded = self.encode_view(image)
        return self.head(encoded["embedding"])

    def forward_views(self, views: tuple[torch.Tensor, torch.Tensor]) -> tuple[dict, dict]:
        first = self.head(self.encode_view(views[0])["embedding"])
        second = self.head(self.encode_view(views[1])["embedding"])
        return first, second
