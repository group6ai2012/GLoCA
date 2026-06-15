from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "ADAPTERS",
    "CLSAdapter",
    "GLOCA_VARIATIONS",
    "GLoCAGatedAdapter",
    "GLoCASumAdapter",
    "GatedAttentionPooling",
    "SimpleAttentionPooling",
    "build_adapter",
    "init_identity_linear_if_possible",
]


class SimpleAttentionPooling(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(patch_tokens).squeeze(-1)
        attention = torch.softmax(scores, dim=1)
        pooled = torch.sum(attention.unsqueeze(-1) * patch_tokens, dim=1)
        return pooled, attention


class GatedAttentionPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.value = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated = torch.tanh(self.value(patch_tokens)) * torch.sigmoid(self.gate(patch_tokens))
        scores = self.score(gated).squeeze(-1)
        attention = torch.softmax(scores, dim=1)
        pooled = torch.sum(attention.unsqueeze(-1) * patch_tokens, dim=1)
        return pooled, attention


def init_identity_linear_if_possible(layer: nn.Linear) -> bool:
    if layer.weight.shape[0] == layer.weight.shape[1]:
        nn.init.eye_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        return True
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return False


def zero_init_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # Start residual adapters as identity: forward(x) = x + 0.
        zero_init_linear(self.net[-1])
        self.final_layer_zero_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CLSAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        normalize_output: bool = True,
        mlp_hidden_dim: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.cls_proj = nn.Linear(input_dim, embedding_dim)
        self.cls_proj_identity_initialized = init_identity_linear_if_possible(self.cls_proj)
        self.residual = ResidualMLP(embedding_dim, mlp_hidden_dim)
        self.normalize_output = bool(normalize_output)

    def forward(
        self,
        cls: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_grid: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | tuple[int, int] | None]:
        embedding = self.residual(self.cls_proj(cls))
        if self.normalize_output:
            embedding = F.normalize(embedding, dim=-1)
        return {"embedding": embedding, "attention": None, "patch_grid": patch_grid}


class GLoCASumAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        normalize_output: bool = True,
        mlp_hidden_dim: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.cls_proj = nn.Linear(input_dim, embedding_dim)
        self.patch_pool = SimpleAttentionPooling(input_dim)
        self.patch_proj = nn.Linear(input_dim, embedding_dim)
        self.residual = ResidualMLP(embedding_dim, mlp_hidden_dim)
        self.normalize_output = bool(normalize_output)

    def forward(
        self,
        cls: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_grid: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | tuple[int, int] | None]:
        z_cls = self.cls_proj(cls)
        pooled, attention = self.patch_pool(patch_tokens)
        z_patch = self.patch_proj(pooled)
        embedding = self.residual(z_cls + z_patch)
        if self.normalize_output:
            embedding = F.normalize(embedding, dim=-1)
        return {"embedding": embedding, "attention": attention, "patch_grid": patch_grid}

class GLoCAGatedAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        normalize_output: bool = True,
        attention_hidden_dim: int | None = None,
        mlp_hidden_dim: int | None = None,
        alpha_init: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.cls_proj = nn.Linear(input_dim, embedding_dim)
        self.cls_proj_identity_initialized = init_identity_linear_if_possible(self.cls_proj)
        self.patch_pool = GatedAttentionPooling(input_dim, attention_hidden_dim or embedding_dim)
        self.patch_proj = nn.Linear(input_dim, embedding_dim)
        hidden_dim = mlp_hidden_dim or embedding_dim * 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.normalize_output = bool(normalize_output)

    def forward(
        self,
        cls: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_grid: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | tuple[int, int] | None]:
        computed = self._compute(cls=cls, patch_tokens=patch_tokens, patch_grid=patch_grid)
        return {
            "embedding": computed["embedding"],
            "attention": computed["attention"],
            "patch_grid": computed["patch_grid"],
        }

    def _compute(
        self,
        cls: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_grid: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | tuple[int, int] | None]:
        z_cls = self.cls_proj(cls)
        pooled, attention = self.patch_pool(patch_tokens)
        z_patch = self.patch_proj(pooled)
        delta = self.fusion_mlp(torch.cat([z_cls, z_patch], dim=-1))
        embedding = z_cls + self.alpha * delta
        if self.normalize_output:
            embedding = F.normalize(embedding, dim=-1)
        return {
            "embedding": embedding,
            "attention": attention,
            "patch_grid": patch_grid,
            "z_cls": z_cls,
            "z_patch": z_patch,
            "delta": delta,
        }

    def diagnostics(
        self,
        cls: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_grid: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | tuple[int, int] | None]:
        return self._compute(cls=cls, patch_tokens=patch_tokens, patch_grid=patch_grid)


ADAPTERS = {
    "cls": CLSAdapter,
    "gloca_sum": GLoCASumAdapter,
    "gloca_gated": GLoCAGatedAdapter,
}

GLOCA_VARIATIONS = set(ADAPTERS)


def build_adapter(config: dict, input_dim: int) -> nn.Module | None:
    gloca_config = config.get("gloca", {})
    if not bool(gloca_config.get("enabled", False)):
        return None

    name = gloca_config.get("name") or gloca_config.get("variation")
    if name in {None, "disabled"}:
        return None
    if name not in ADAPTERS:
        accepted = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown adapter '{name}'. Accepted values: {accepted}")

    embedding_dim = int(gloca_config.get("embedding_dim", input_dim))
    return ADAPTERS[name](
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        normalize_output=bool(gloca_config.get("normalize_output", True)),
        attention_hidden_dim=(
            None
            if gloca_config.get("attention_hidden_dim") is None
            else int(gloca_config["attention_hidden_dim"])
        ),
        mlp_hidden_dim=None if gloca_config.get("mlp_hidden_dim") is None else int(gloca_config["mlp_hidden_dim"]),
        alpha_init=float(gloca_config.get("alpha_init", 0.0)),
    )
