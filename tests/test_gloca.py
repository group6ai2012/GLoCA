from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.gloca import (
    CLSAdapter,
    GLoCAGatedAdapter,
    GLoCAMeanPoolAdapter,
    GLoCANoCLSResidualAdapter,
    GLoCANoPatchesAdapter,
    build_adapter,
    init_identity_linear_if_possible,
)


def test_gloca_gated_alpha_zero_preserves_normalized_cls_when_dims_match() -> None:
    batch_size, n_patches, dim = 4, 256, 384
    cls = torch.randn(batch_size, dim)
    patch_tokens = torch.randn(batch_size, n_patches, dim)
    adapter = GLoCAGatedAdapter(
        input_dim=dim,
        embedding_dim=dim,
        alpha_init=0.0,
        normalize_output=True,
    )

    out = adapter(cls, patch_tokens, patch_grid=(16, 16))
    expected = F.normalize(cls, dim=-1)
    cosine = F.cosine_similarity(out["embedding"], expected, dim=-1)

    assert float(cosine.mean().detach()) > 0.999
    assert out["embedding"].shape == (batch_size, dim)
    assert out["attention"].shape == (batch_size, n_patches)
    assert adapter.cls_proj_identity_initialized


def test_gloca_gated_alpha_participates_in_backward() -> None:
    batch_size, n_patches, dim = 2, 8, 16
    cls = torch.randn(batch_size, dim)
    patch_tokens = torch.randn(batch_size, n_patches, dim)
    target = F.normalize(torch.randn(batch_size, dim), dim=-1)
    adapter = GLoCAGatedAdapter(input_dim=dim, embedding_dim=dim, alpha_init=0.0)

    loss = -(adapter(cls, patch_tokens)["embedding"] * target).sum(dim=-1).mean()
    loss.backward()

    assert adapter.alpha.grad is not None
    assert torch.isfinite(adapter.alpha.grad)
    assert abs(float(adapter.alpha.grad.detach())) > 0.0


def test_cls_adapter_is_conservative_when_dims_match() -> None:
    batch_size, dim = 3, 32
    cls = torch.randn(batch_size, dim)
    adapter = CLSAdapter(input_dim=dim, embedding_dim=dim, normalize_output=True)
    out = adapter(cls, torch.empty(batch_size, 0, dim))

    expected = F.normalize(cls, dim=-1)
    cosine = F.cosine_similarity(out["embedding"], expected, dim=-1)
    assert float(cosine.mean().detach()) > 0.999
    assert adapter.cls_proj_identity_initialized
    assert adapter.residual.final_layer_zero_initialized


def test_identity_linear_helper_falls_back_for_non_square_projection() -> None:
    layer = nn.Linear(4, 6)
    was_identity = init_identity_linear_if_possible(layer)

    assert not was_identity
    assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))


def test_direct_cls_baseline_does_not_build_adapter() -> None:
    adapter = build_adapter(
        {
            "gloca": {
                "enabled": False,
                "name": "gloca_gated",
                "embedding_dim": 384,
            }
        },
        input_dim=384,
    )

    assert adapter is None


def test_gloca_ablation_variants_build_from_config() -> None:
    for name in ["gloca_mean_pool", "gloca_no_patches", "gloca_no_cls_residual"]:
        adapter = build_adapter(
            {
                "gloca": {
                    "enabled": True,
                    "name": name,
                    "embedding_dim": 16,
                    "alpha_init": 0.0,
                    "normalize_output": True,
                    "attention_hidden_dim": 8,
                    "mlp_hidden_dim": 32,
                }
            },
            input_dim=16,
        )

        assert adapter is not None


def test_gloca_mean_pool_returns_uniform_attention() -> None:
    batch_size, n_patches, dim = 3, 5, 16
    cls = torch.randn(batch_size, dim)
    patch_tokens = torch.randn(batch_size, n_patches, dim)
    adapter = GLoCAMeanPoolAdapter(input_dim=dim, embedding_dim=dim, alpha_init=0.0)

    out = adapter(cls, patch_tokens, patch_grid=(1, 5))

    assert out["embedding"].shape == (batch_size, dim)
    assert torch.isfinite(out["embedding"]).all()
    assert torch.allclose(out["embedding"].norm(dim=-1), torch.ones(batch_size), atol=1.0e-6)
    assert out["attention"].shape == (batch_size, n_patches)
    assert torch.allclose(out["attention"], torch.full((batch_size, n_patches), 1.0 / n_patches))


def test_gloca_no_patches_ignores_patch_content() -> None:
    batch_size, n_patches, dim = 3, 7, 16
    cls = torch.randn(batch_size, dim)
    patch_tokens = torch.randn(batch_size, n_patches, dim)
    changed_patch_tokens = torch.randn(batch_size, n_patches, dim) * 100.0
    adapter = GLoCANoPatchesAdapter(input_dim=dim, embedding_dim=dim, alpha_init=1.0)

    first = adapter(cls, patch_tokens, patch_grid=(1, 7))
    second = adapter(cls, changed_patch_tokens, patch_grid=(1, 7))

    assert first["attention"] is None
    assert torch.isfinite(first["embedding"]).all()
    assert torch.allclose(first["embedding"].norm(dim=-1), torch.ones(batch_size), atol=1.0e-6)
    assert torch.allclose(first["embedding"], second["embedding"])


def test_gloca_no_cls_residual_outputs_normalized_delta() -> None:
    batch_size, n_patches, dim = 3, 9, 16
    cls = torch.randn(batch_size, dim)
    patch_tokens = torch.randn(batch_size, n_patches, dim)
    adapter = GLoCANoCLSResidualAdapter(input_dim=dim, embedding_dim=dim, alpha_init=0.0)

    diagnostics = adapter.diagnostics(cls, patch_tokens, patch_grid=(3, 3))

    assert diagnostics["attention"].shape == (batch_size, n_patches)
    assert torch.isfinite(diagnostics["embedding"]).all()
    assert torch.allclose(diagnostics["embedding"].norm(dim=-1), torch.ones(batch_size), atol=1.0e-6)
    assert torch.allclose(diagnostics["embedding"], F.normalize(diagnostics["delta"], dim=-1))
