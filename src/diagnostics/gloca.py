from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

GLOCA_DIAGNOSTIC_KEYS = (
    "gloca_alpha_value",
    "gloca_alpha_grad_norm",
    "gloca_param_grad_norm",
    "gloca_delta_norm_mean",
    "gloca_delta_norm_std",
    "gloca_embedding_cls_cosine_mean",
    "gloca_embedding_cls_cosine_std",
    "gloca_attention_entropy_mean",
    "gloca_attention_max_mean",
)


@torch.no_grad()
def collect_gloca_diagnostics(
    model: nn.Module,
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, float | None]:
    adapter = _resolve_adapter(model)
    if adapter is None:
        return {}

    output: dict[str, float | None] = {
        "gloca_alpha_value": _current_gloca_alpha(adapter),
        "gloca_alpha_grad_norm": _parameter_grad_norm(_gloca_alpha_parameters(adapter)),
        "gloca_param_grad_norm": _parameter_grad_norm(_split_gloca_parameters(adapter)[0]),
        "gloca_delta_norm_mean": None,
        "gloca_delta_norm_std": None,
        "gloca_embedding_cls_cosine_mean": None,
        "gloca_embedding_cls_cosine_std": None,
        "gloca_attention_entropy_mean": None,
        "gloca_attention_max_mean": None,
    }
    if diagnostics is None:
        return output

    embedding = diagnostics.get("embedding")
    cls = _diagnostic_cls_tensor(diagnostics)
    if isinstance(embedding, torch.Tensor) and isinstance(cls, torch.Tensor) and embedding.shape == cls.shape:
        cosine = F.cosine_similarity(cls, embedding, dim=-1)
        output["gloca_embedding_cls_cosine_mean"] = float(cosine.mean().detach().cpu())
        output["gloca_embedding_cls_cosine_std"] = float(cosine.std(unbiased=False).detach().cpu())

    delta = diagnostics.get("delta")
    if isinstance(delta, torch.Tensor):
        delta_norm = delta.norm(dim=-1)
        output["gloca_delta_norm_mean"] = float(delta_norm.mean().detach().cpu())
        output["gloca_delta_norm_std"] = float(delta_norm.std(unbiased=False).detach().cpu())

    attention = diagnostics.get("attention")
    if isinstance(attention, torch.Tensor):
        attention_safe = attention.clamp_min(1.0e-12)
        entropy = -(attention_safe * attention_safe.log()).sum(dim=1)
        output["gloca_attention_entropy_mean"] = float(entropy.mean().detach().cpu())
        output["gloca_attention_max_mean"] = float(attention.max(dim=1).values.mean().detach().cpu())

    return output


def _resolve_adapter(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "adapter"):
        adapter = getattr(model, "adapter")
        return adapter if isinstance(adapter, nn.Module) else None
    return model if isinstance(model, nn.Module) else None


def _diagnostic_cls_tensor(diagnostics: Mapping[str, Any]) -> torch.Tensor | None:
    for key in ("cls", "normalized_cls", "z_cls"):
        value = diagnostics.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _split_gloca_parameters(adapter: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    gloca_params: list[nn.Parameter] = []
    alpha_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in adapter.named_parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        if "alpha" in name:
            alpha_params.append(parameter)
        else:
            gloca_params.append(parameter)
    return gloca_params, alpha_params


def _gloca_alpha_parameters(adapter: nn.Module) -> list[nn.Parameter]:
    return _split_gloca_parameters(adapter)[1]


def _current_gloca_alpha(adapter: nn.Module) -> float | None:
    alpha_params = _gloca_alpha_parameters(adapter)
    if not alpha_params:
        return None
    if len(alpha_params) == 1 and alpha_params[0].numel() == 1:
        return float(alpha_params[0].detach().cpu().item())
    alpha_values = torch.cat([parameter.detach().flatten().cpu() for parameter in alpha_params])
    return float(alpha_values.mean().item())


def _parameter_grad_norm(parameters: Iterable[nn.Parameter]) -> float | None:
    grad_norm_sq = 0.0
    found_grad = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found_grad = True
        grad_norm_sq += float(parameter.grad.detach().float().pow(2).sum().cpu())
    if not found_grad:
        return None
    return math.sqrt(grad_norm_sq)
