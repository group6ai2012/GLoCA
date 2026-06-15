from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.diagnostics import collect_gloca_diagnostics


def assert_backbone_frozen(backbone: nn.Module) -> None:
    trainable = [
        name
        for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError(
            f"DINOv2 backbone parameters require gradients: {trainable[:5]}"
        )


def set_gloca_trainable(adapter: nn.Module | None, trainable: bool) -> bool:
    if adapter is None:
        return False
    for parameter in adapter.parameters():
        parameter.requires_grad = bool(trainable)
    return bool(trainable)


def desired_gloca_trainable(
    adapter: nn.Module | None,
    *,
    freeze_gloca: bool,
    freeze_gloca_epochs: int,
    epoch: int,
) -> bool:
    if adapter is None:
        return False
    if freeze_gloca:
        return False
    if int(freeze_gloca_epochs) > 0 and int(epoch) < int(freeze_gloca_epochs):
        return False
    return True


def split_gloca_parameters(
    adapter: nn.Module | None,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    if adapter is None:
        return [], []
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


def gloca_alpha_parameters(adapter: nn.Module | None) -> list[nn.Parameter]:
    return split_gloca_parameters(adapter)[1]


def current_gloca_alpha(adapter: nn.Module | None) -> float | None:
    alpha_params = gloca_alpha_parameters(adapter)
    if not alpha_params:
        return None
    if len(alpha_params) == 1 and alpha_params[0].numel() == 1:
        return float(alpha_params[0].detach().cpu().item())
    alpha_values = torch.cat(
        [parameter.detach().flatten().cpu() for parameter in alpha_params]
    )
    return float(alpha_values.mean().item())


def gloca_optimizer_groups(
    adapter: nn.Module | None,
    *,
    base_lr: float,
    weight_decay: float,
    gloca_lr_multiplier: float,
    gloca_alpha_lr_multiplier: float,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    gloca_params, alpha_params = split_gloca_parameters(adapter)
    if gloca_params:
        groups.append(
            {
                "params": gloca_params,
                "lr": float(base_lr) * float(gloca_lr_multiplier),
                "weight_decay": float(weight_decay),
                "name": "gloca",
            }
        )
    if alpha_params:
        groups.append(
            {
                "params": alpha_params,
                "lr": float(base_lr) * float(gloca_alpha_lr_multiplier),
                "weight_decay": 0.0,
                "name": "gloca_alpha",
            }
        )
    return groups


def compute_gloca_diagnostics(
    *,
    model: nn.Module,
    datamodule: Any,
    device: torch.device,
    epoch: int,
    restore_training_fn: Callable[[], None] | None = None,
) -> dict[str, Any] | None:
    if model.adapter is None:
        return None

    diagnostics: dict[str, Any] = {
        "epoch": int(epoch),
        "gloca_trainable": bool(
            any(parameter.requires_grad for parameter in model.adapter.parameters())
        ),
        "gloca_frozen": bool(
            not any(parameter.requires_grad for parameter in model.adapter.parameters())
        ),
    }

    was_training = model.training
    model.eval()
    model.backbone.eval()
    model.adapter.eval()
    try:
        batch = next(iter(datamodule.train_eval_dataloader()))
        image = batch["image"].to(device)
        with torch.no_grad():
            backbone_out = model.backbone(image)
            normalized_cls = F.normalize(backbone_out["cls"], dim=-1)
            if hasattr(model.adapter, "diagnostics"):
                adapter_out = model.adapter.diagnostics(
                    cls=backbone_out["cls"],
                    patch_tokens=backbone_out["patch_tokens"],
                    patch_grid=backbone_out["patch_grid"],
                )
            else:
                adapter_out = model.adapter(
                    cls=backbone_out["cls"],
                    patch_tokens=backbone_out["patch_tokens"],
                    patch_grid=backbone_out["patch_grid"],
                )
            diagnostics.update(
                collect_gloca_diagnostics(
                    model, {**adapter_out, "normalized_cls": normalized_cls}
                )
            )
    finally:
        if was_training and restore_training_fn is not None:
            restore_training_fn()
        elif was_training:
            model.train()
            model.backbone.eval()
    return diagnostics
