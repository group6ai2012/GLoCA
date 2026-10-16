from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest
import torch
from torch import nn

from src.inference.cdc_streamlit import (
    CDCInferenceBundle,
    build_exemplar_index,
    decode_uploaded_image,
    predict_top_clusters,
    rank_cluster_exemplars,
    validate_demo_config,
)


class _PredictionModel(nn.Module):
    def __init__(self, probabilities: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("probabilities", probabilities)

    def forward(self, _image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"calibration_probabilities": self.probabilities.unsqueeze(0)}


class _CalibrationHead(nn.Module):
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings


class _ExemplarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Module()
        self.head.calibration_head = _CalibrationHead()


def _config(dataset: str = "plantseg", n_clusters: int = 114) -> dict:
    return {
        "experiment": {"seed": 42},
        "dataset": {"name": dataset},
        "backbone": {"freeze": True},
        "gloca": {"enabled": True, "name": "gloca_gated", "embedding_dim": n_clusters},
        "head": {"name": "cdc", "n_clusters": n_clusters},
        "prediction": {"deterministic": True},
    }


def _bundle(
    tmp_path: Path,
    *,
    model: nn.Module,
    n_clusters: int,
    config: dict | None = None,
) -> CDCInferenceBundle:
    return CDCInferenceBundle(
        dataset="plantseg",
        display_name="PlantSeg",
        run_dir=tmp_path / "run",
        dataset_root=tmp_path / "dataset",
        config=config or _config(n_clusters=n_clusters),
        model=model,
        device=torch.device("cpu"),
        transform=lambda _image: torch.zeros(3, 8, 8),
        n_clusters=n_clusters,
    )


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 12), color=(20, 120, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_validate_demo_config_enforces_seed_42_gloca_cdc_contract() -> None:
    config = _config()
    validate_demo_config(config, "plantseg", 114)

    config["gloca"]["name"] = "gloca_mean_pool"
    with pytest.raises(ValueError, match="gloca.name"):
        validate_demo_config(config, "plantseg", 114)


def test_decode_uploaded_image_validates_content_and_size() -> None:
    image = decode_uploaded_image(_image_bytes())
    assert image.mode == "RGB"
    assert image.size == (16, 12)

    with pytest.raises(ValueError, match="not a valid"):
        decode_uploaded_image(b"not an image")
    with pytest.raises(ValueError, match="larger than"):
        decode_uploaded_image(_image_bytes(), max_bytes=2)


def test_predict_top_clusters_uses_calibrated_probabilities_and_stable_order(
    tmp_path: Path,
) -> None:
    model = _PredictionModel(torch.tensor([0.10, 0.40, 0.40, 0.10]))
    bundle = _bundle(tmp_path, model=model, n_clusters=4)

    result = predict_top_clusters(bundle, Image.new("RGB", (8, 8)), 3)

    assert [item.cluster_id for item in result] == [1, 2, 0]
    assert [item.probability for item in result] == pytest.approx([0.4, 0.4, 0.1])
    with pytest.raises(ValueError, match="between 1 and 4"):
        predict_top_clusters(bundle, Image.new("RGB", (8, 8)), 5)


def test_rank_cluster_exemplars_selects_five_per_cluster_deterministically(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    image_ids = [f"class/image_{index}.png" for index in range(6)]
    (dataset_root / "class").mkdir()
    for image_id in image_ids:
        Image.new("RGB", (4, 4)).save(dataset_root / image_id)
    probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.4, 0.6],
        ]
    )

    result = rank_cluster_exemplars(
        probabilities, image_ids, dataset_root, examples_per_cluster=5
    )

    assert len(result[0]) == 5
    assert len(result[1]) == 5
    assert result[0][0].image_id == "class/image_0.png"
    assert result[1][0].image_id == "class/image_5.png"


def test_rank_cluster_exemplars_rejects_path_escape(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    with pytest.raises(ValueError, match="Unsafe dataset image ID"):
        rank_cluster_exemplars(
            torch.tensor([[1.0]]), ["../outside.png"], dataset_root, examples_per_cluster=1
        )


def test_build_exemplar_index_scores_every_cluster_from_stored_embeddings(
    tmp_path: Path,
) -> None:
    bundle = _bundle(
        tmp_path,
        model=_ExemplarModel(),
        n_clusters=2,
        config=_config(n_clusters=2),
    )
    bundle.run_dir.mkdir(parents=True)
    bundle.dataset_root.mkdir(parents=True)
    (bundle.dataset_root / "class").mkdir()
    image_ids = [f"class/image_{index}.png" for index in range(6)]
    for image_id in image_ids:
        Image.new("RGB", (4, 4)).save(bundle.dataset_root / image_id)
    assignments = {
        "dataset": "plantseg",
        "seed": 42,
        "head": "cdc",
        "gloca": "gloca_gated",
        "n_clusters": 2,
        "image_ids": image_ids,
    }
    (bundle.run_dir / "assignments.json").write_text(
        json.dumps(assignments), encoding="utf-8"
    )
    torch.save(
        torch.tensor(
            [
                [6.0, 0.0],
                [5.0, 1.0],
                [4.0, 2.0],
                [2.0, 4.0],
                [1.0, 5.0],
                [0.0, 6.0],
            ]
        ),
        bundle.run_dir / "embeddings.pt",
    )

    result = build_exemplar_index(bundle, examples_per_cluster=5, batch_size=2)

    assert result[0][0].image_id == "class/image_0.png"
    assert result[1][0].image_id == "class/image_5.png"
    assert all(len(exemplars) == 5 for exemplars in result.values())


def test_streamlit_page_renders_without_loading_a_model() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("streamlit_app.py").run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "CDC + GLoCA Plant Disease Cluster Explorer"
    assert app.selectbox[0].value == "plantseg"
    assert app.slider[0].value == 3
    assert app.button[0].disabled is True
