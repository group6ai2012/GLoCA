from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class FrozenCLSMeanPatchConcat(nn.Module):
    """Parameter-free global-local embedding built from frozen DINOv2 tokens."""

    def __init__(self, normalize_output: bool = True) -> None:
        super().__init__()
        self.normalize_output = bool(normalize_output)

    def forward(
        self,
        cls: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_grid: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        if cls.ndim != 2:
            raise ValueError(f"CLS tensor must have shape [B, D], got {tuple(cls.shape)}")
        if patch_tokens.ndim != 3:
            raise ValueError(
                "Patch-token tensor must have shape [B, N, D], "
                f"got {tuple(patch_tokens.shape)}"
            )
        if cls.shape[0] != patch_tokens.shape[0] or cls.shape[1] != patch_tokens.shape[2]:
            raise ValueError(
                "CLS and patch-token dimensions must agree, got "
                f"CLS={tuple(cls.shape)} patches={tuple(patch_tokens.shape)}"
            )
        if patch_tokens.shape[1] == 0:
            raise ValueError("Mean patch pooling requires at least one patch token.")

        mean_patch = patch_tokens.mean(dim=1)
        embedding = torch.cat((cls, mean_patch), dim=-1)
        if self.normalize_output:
            embedding = F.normalize(embedding, dim=-1)
        return {
            "embedding": embedding,
            "attention": None,
            "patch_grid": patch_grid,
        }
