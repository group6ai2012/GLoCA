from __future__ import annotations

import math

import torch
from torch import nn

from src.diagnostics import GLOCA_DIAGNOSTIC_KEYS, collect_gloca_diagnostics


class _Model(nn.Module):
    def __init__(self, adapter: nn.Module | None) -> None:
        super().__init__()
        self.adapter = adapter


class _Adapter(nn.Module):
    def __init__(self, with_alpha: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        if with_alpha:
            self.alpha = nn.Parameter(torch.tensor(0.25))


def _diagnostic_tensors(attention: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
    return {
        "cls": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "embedding": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "delta": torch.tensor([[3.0, 4.0], [0.0, 12.0]]),
        "attention": attention,
    }


def test_collect_gloca_diagnostics_disabled_path_returns_empty_dict() -> None:
    assert collect_gloca_diagnostics(_Model(adapter=None), _diagnostic_tensors()) == {}


def test_collect_gloca_diagnostics_attention_none_and_missing_grad_are_safe() -> None:
    diagnostics = collect_gloca_diagnostics(_Model(_Adapter()), _diagnostic_tensors(attention=None))

    assert set(diagnostics) == set(GLOCA_DIAGNOSTIC_KEYS)
    assert diagnostics["gloca_alpha_value"] == 0.25
    assert diagnostics["gloca_alpha_grad_norm"] is None
    assert diagnostics["gloca_attention_entropy_mean"] is None
    assert diagnostics["gloca_attention_max_mean"] is None


def test_collect_gloca_diagnostics_missing_alpha_is_safe() -> None:
    diagnostics = collect_gloca_diagnostics(_Model(_Adapter(with_alpha=False)), _diagnostic_tensors())

    assert diagnostics["gloca_alpha_value"] is None
    assert diagnostics["gloca_alpha_grad_norm"] is None


def test_collect_gloca_diagnostics_returns_python_floats_not_tensors() -> None:
    adapter = _Adapter()
    adapter.alpha.grad = torch.tensor(2.0)
    for parameter in adapter.proj.parameters():
        parameter.grad = torch.ones_like(parameter)

    diagnostics = collect_gloca_diagnostics(
        _Model(adapter),
        _diagnostic_tensors(attention=torch.tensor([[0.5, 0.5], [1.0, 0.0]])),
    )

    for value in diagnostics.values():
        assert value is None or isinstance(value, float)
        assert not isinstance(value, torch.Tensor)


def test_collect_gloca_diagnostics_attention_entropy_and_max_are_sane() -> None:
    diagnostics = collect_gloca_diagnostics(
        _Model(_Adapter()),
        _diagnostic_tensors(attention=torch.tensor([[0.5, 0.5], [1.0, 0.0]])),
    )

    assert math.isclose(diagnostics["gloca_attention_entropy_mean"], math.log(2.0) / 2.0, rel_tol=0.0, abs_tol=1.0e-6)
    assert diagnostics["gloca_attention_max_mean"] == 0.75


def test_collect_gloca_diagnostics_preserves_previous_propos_scalar_keys() -> None:
    assert GLOCA_DIAGNOSTIC_KEYS == (
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
