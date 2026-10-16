from __future__ import annotations

import numpy as np
import pytest

from scripts.visualize_gloca_attention import (
    AttentionSample,
    attention_to_grid,
    colorize_heatmap,
    project_attention_to_original_canvas,
    select_per_class,
)


def test_attention_to_grid_requires_patch_grid_product() -> None:
    grid = attention_to_grid(np.arange(6, dtype=np.float32), patch_grid=(2, 3))

    assert grid.shape == (2, 3)
    assert float(grid[1, 2]) == 5.0

    with pytest.raises(ValueError, match="patch_grid product"):
        attention_to_grid(np.arange(5, dtype=np.float32), patch_grid=(2, 3))


def test_select_per_class_is_deterministic_and_limited() -> None:
    samples = [
        AttentionSample(
            index=i,
            image_id=f"class_{i // 4}/img_{i}.jpg",
            label=i // 4,
            label_name=f"class_{i // 4}",
            assignment=i,
        )
        for i in range(12)
    ]

    selected_once = select_per_class(samples, n_per_class=2, seed=7)
    selected_twice = select_per_class(samples, n_per_class=2, seed=7)

    assert [sample.index for sample in selected_once] == [sample.index for sample in selected_twice]
    assert len(selected_once) == 6
    assert {label: sum(sample.label == label for sample in selected_once) for label in range(3)} == {
        0: 2,
        1: 2,
        2: 2,
    }


def test_project_attention_to_original_canvas_keeps_original_size() -> None:
    heatmap = np.ones((2, 2), dtype=np.float32)

    projected = project_attention_to_original_canvas(
        heatmap,
        original_size=(320, 240),
        image_size=224,
    )

    assert projected.shape == (240, 320)
    assert float(projected.max()) > 0.0
    assert float(projected[0, 0]) == 0.0


def test_colorize_heatmap_returns_rgb_values() -> None:
    colored = colorize_heatmap(np.array([[0.0, 0.5, 1.0]], dtype=np.float32), "jet")

    assert colored.shape == (1, 3, 3)
    assert float(colored.min()) >= 0.0
    assert float(colored.max()) <= 1.0
