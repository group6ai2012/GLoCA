# GLoCA: plant disease clustering with frozen DINOv2

GLoCA is an embedding adapter for studying unsupervised plant disease discovery. It combines DINOv2's global CLS representation with pooled local patch features, then passes the resulting embedding to a clustering method. The DINOv2 backbone stays frozen.

The research comparison holds the clustering method fixed:

```text
Frozen DINOv2 → normalized CLS embedding → clustering method
Frozen DINOv2 → GLoCA embedding          → same clustering method
```

The main datasets are PlantSeg and PlantVillage. Saved experiments show improved mean clustering metrics with CDC, but the available ProPos comparison does not show the same benefit. Read the [experiment report](docs/full_result_report.md) for results, seed coverage, uncertainty, and limitations.

## Install

The pinned environment targets Windows x64 with Python 3.12 and CUDA-enabled PyTorch. Install Conda and a compatible NVIDIA driver, then run these commands from the repository root:

```powershell
conda env create -f environment.windows.yml
conda activate gloca-repro
python -m pip check
```

The lock file pins Windows packages; it is not a portable Linux or macOS environment. The first backbone load downloads the pinned `facebook/dinov2-small` model. Dataset images and trained experiment artifacts are not included in Git.

## Run your first experiment

Follow [dataset preparation](docs/USER_GUIDE.md#prepare-the-datasets) to obtain and arrange PlantSeg images. Then run a short CDC experiment using six images:

```powershell
python scripts/run_cdc.py configs/cdc/cdc_plantseg_cls_repro_smoke.yaml
python scripts/run_cdc.py configs/cdc/cdc_plantseg_gloca_gated_repro_smoke.yaml
```

These short runs check the workflow; their scores do not measure full-dataset performance. Results are written under `outputs/reproducibility_smoke/live/`.

For full experiments, use the [experiment commands](docs/USER_GUIDE.md#run-experiments). Choose a fresh output root before running against a directory containing older artifacts.

## Documentation

| Document | What you can find |
|---|---|
| [User guide](docs/USER_GUIDE.md) | Datasets, supported methods, configuration, experiment commands, saved outputs, attention maps, and the image-upload app |
| [Reproducibility guide](docs/REPRODUCIBILITY.md) | Environment and input verification, seed overrides, resume, completion checks, and troubleshooting |
| [Experiment report](docs/full_result_report.md) | Consolidated PlantSeg and PlantVillage results, per-seed evidence, statistical analysis, and limitations |

The optional Streamlit app explores anonymous learned clusters. Cluster affinity scores and attention maps are research diagnostics, not disease diagnoses.
