from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from transformers import AutoModel

from src.features.dinov2 import DINOV2_VARIANTS


LOGGER = logging.getLogger(__name__)


class DINOv2Backbone(nn.Module):
    def __init__(self, variant: str, image_size: int = 224, freeze: bool = True) -> None:
        super().__init__()
        if variant not in DINOV2_VARIANTS:
            accepted = ", ".join(sorted(DINOV2_VARIANTS))
            raise ValueError(f"Unknown DINOv2 variant '{variant}'. Accepted values: {accepted}")
        self.variant = variant
        self.image_size = image_size
        self.model = AutoModel.from_pretrained(variant)
        self.output_dim = int(self.model.config.hidden_size)
        self.patch_size = int(getattr(self.model.config, "patch_size", 14))
        self.patch_grid = (image_size // self.patch_size, image_size // self.patch_size)
        self._logged_shapes = False
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, image: torch.Tensor) -> dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            output = self.model(pixel_values=image)
            hidden = output.last_hidden_state
        cls = hidden[:, 0, :]
        patch_tokens = hidden[:, 1:, :]
        if not self._logged_shapes:
            LOGGER.info(
                "DINOv2 output shapes cls=%s patch_tokens=%s patch_grid=%s",
                tuple(cls.shape),
                tuple(patch_tokens.shape),
                self.patch_grid,
            )
            self._logged_shapes = True
        return {"cls": cls, "patch_tokens": patch_tokens, "patch_grid": self.patch_grid}

