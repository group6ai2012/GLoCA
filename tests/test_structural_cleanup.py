from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_no_production_code_imports_sklearn_kmeans() -> None:
    offenders = []
    for path in [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/**/*.py")]:
        source = path.read_text(encoding="utf-8")
        if "from sklearn.cluster import KMeans" in source or "from sklearn.cluster import KMeans" in source:
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_kmeans_runner_uses_model_layer_fit_kmeans() -> None:
    source = _read("src/runners/kmeans.py")
    assert "from src.models.clustering.kmeans import fit_kmeans" in source
    assert "fit_kmeans(" in source


def test_runner_common_stays_minimal() -> None:
    source = _read("src/runners/common.py")
    forbidden_helpers = [
        "def build_single_view_datamodule",
        "def build_dinov2_backbone",
        "def build_common_metrics_row",
        "def build_common_logs",
    ]
    assert not any(helper in source for helper in forbidden_helpers)


def test_propos_uses_shared_fit_kmeans_for_estep_and_final_prediction() -> None:
    trainer_source = _read("src/training/propos_trainer.py")
    runner_source = _read("src/runners/propos.py")
    assert "from src.models.clustering.kmeans import fit_kmeans" in trainer_source
    assert "fit_kmeans(" in trainer_source
    assert "from src.models.clustering import ProPosHead, fit_kmeans" in runner_source
    assert "fit_kmeans(" in runner_source
    assert "spherical=True" in trainer_source
    assert "spherical=True" in runner_source


def test_experiment_scripts_parse_exactly_one_config_path() -> None:
    for path in [
        "scripts/run_cdc.py",
        "scripts/run_propos.py",
        "src/experiments/baselines.py",
        "scripts/run_propos_gloca_diagnostics.py",
    ]:
        source = _read(path)
        assert 'parser.add_argument("config_path"' in source
        assert "--dataset" not in source
        assert "--epochs" not in source
        assert "--batch-size" not in source
        assert "--num-workers" not in source
        assert "--output-root" not in source
        assert "--gloca" not in source
        assert "--kmeans-backend" not in source
        assert "--matmul-precision" not in source
        assert "--matmul_precision" not in source
        assert "--only" not in source


def test_removed_hidden_config_builders_are_absent() -> None:
    assert "_config_from_args" not in _read("scripts/run_propos.py")
    baselines = _read("scripts/run_baselines.py")
    assert "DEFAULT_DATASETS" not in baselines
    assert "DEFAULT_SEEDS" not in baselines
    assert "_baseline_definitions" not in baselines


def test_config_template_contains_major_sections() -> None:
    template = _read("configs/templates/full_config_template.yaml")
    for section in [
        "experiment:",
        "dataset:",
        "backbone:",
        "gloca:",
        "head:",
        "baseline:",
        "propos:",
        "cdc:",
        "prediction:",
        "trainer:",
        "sweep:",
        "diagnostics:",
    ]:
        assert section in template
    assert "matmul_precision:" in template


def test_propos_configs_do_not_duplicate_trainer_owned_fields() -> None:
    paths = [*ROOT.glob("configs/propos/*.yaml"), ROOT / "configs/diagnostics/propos_gloca_diagnostics_plantwild_seed42.yaml"]
    forbidden = {"max_epochs", "batch_size", "lr"}
    offenders = []
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        propos_sections = [config.get("propos", {})]
        if "base_config" in config:
            propos_sections.append(config["base_config"].get("propos", {}))
            propos_sections.extend(run.get("config", {}).get("propos", {}) for run in config.get("diagnostics", {}).get("runs", []))
        for section in propos_sections:
            duplicated = sorted(forbidden & set(section))
            if duplicated:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}: {duplicated}")
    assert offenders == []


def test_removed_centroid_diagnostic_head_is_absent() -> None:
    removed_class = "Student" + "T"
    removed_slug = "student" + "_t"
    removed_runner = "run" + "_" + removed_slug
    forbidden_tokens = [removed_class, removed_slug, removed_runner]
    roots = ["src", "scripts", "configs", "docs"]
    offenders = []
    for root in roots:
        for path in (ROOT / root).rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".py", ".yaml", ".yml"}:
                continue
            source = path.read_text(encoding="utf-8")
            if any(token in source for token in forbidden_tokens):
                offenders.append(path.relative_to(ROOT).as_posix())

    assert not (ROOT / "src/models/clustering" / f"{removed_slug}.py").exists()
    assert not (ROOT / "src/runners" / f"{removed_slug}.py").exists()
    assert offenders == []
