# GLoCA user guide

Use GLoCA to compare frozen DINOv2 CLS embeddings with a trainable global/local adapter under the same clustering method. This guide covers preparing inputs, choosing a recipe, running experiments, and interpreting the saved outputs. See the [README](../README.md) for installation and the [experiment report](full_result_report.md) for measured results.

## Prepare the datasets

Download the image collections separately:

| Dataset | Source | Expected folder | Reference images | Categories |
|---|---|---|---:|---:|
| PlantSeg | [PlantSeg dataset record](https://zenodo.org/records/14935094) | `data/raw/plantseg_folder` | 11,458 | 115 |
| PlantVillage | [PlantVillage dataset](https://huggingface.co/datasets/mohanty/PlantVillage) | `data/raw/plantvillage` | 54,305 | 38 |

The counts describe the image snapshots represented by the checked-in [PlantSeg manifest](../data/manifests/plantseg.json) and [PlantVillage manifest](../data/manifests/plantvillage.json). A differently exported download may not match these snapshots byte for byte. Follow [input verification](REPRODUCIBILITY.md#verify-your-inputs) when this happens.

Each dataset uses one directory per category:

```text
data/raw/plantvillage/
  Apple___healthy/
    image_001.jpg
  Apple___Black_rot/
    image_002.jpg
```

Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`. Images are converted to RGB. Folder names define the evaluation labels; the unsupervised methods do not train on these labels.

### Convert PlantSeg

Use the conversion utility with the raw image directory and `Metadatav2.csv`:

```powershell
python scripts/prepare_plantseg_folder_dataset.py --source-root data/raw/plantsegv3/images --metadata-csv data/raw/plantsegv3/Metadatav2.csv --dry-run
python scripts/prepare_plantseg_folder_dataset.py --source-root data/raw/plantsegv3/images --metadata-csv data/raw/plantsegv3/Metadatav2.csv
```

Adjust the source paths to your download. The destination defaults to `data/raw/plantseg_folder`; it must not already exist for a normal conversion. The dry run previews the conversion without copying images. Categories come from the metadata's `Disease` field, with sanitized folder names. Segmentation masks and annotation-like files are excluded; masks are not model inputs. Keep the generated conversion manifest and class counts with your dataset.

### Preprocessing

The default input size is 224 × 224, with ImageNet normalization. Evaluation uses `Resize(256) → CenterCrop(224)`; training uses stochastic crops and image augmentations. ProPos uses two augmented views. CDC uses weak, strong, and deterministic calibration views; its strong view includes RandAugment-like operations and Cutout.

Cropping trades field of view for image scale: it can remove off-center evidence, while whole-image resizing can make lesions smaller. Keep preprocessing identical within a CLS/GLoCA comparison. PlantWild remains available for diagnostic recipes but is excluded from the main report because of its reported overlap with PlantSeg.

## Choose an embedding and method

The backbone is frozen `facebook/dinov2-small` (ViT-S/14). At 224 × 224 it provides a 384-dimensional CLS vector and 256 patch tokens on a 16 × 16 grid.

Direct CLS is selected with `gloca.enabled: false`. The default trainable adapter, `gloca_gated`, produces a normalized 384-dimensional embedding:

```text
z_cls   = CLS projection
z_patch = projection of gated-attention pooled patch tokens
delta   = MLP(concat(z_cls, z_patch))
output  = normalize(z_cls + alpha * delta)
```

The square CLS projection starts as identity and `alpha` starts at zero, anchoring the initial embedding to normalized CLS. GLoCA is shared across clustering methods; it is not itself a clustering algorithm.

| Method or reference | Input and interpretation |
|---|---|
| K-Means / spherical K-Means | Cached deterministic embeddings, with Euclidean or spherical updates |
| Untrained GLoCA + spherical K-Means | Initialization reference; the adapter is not trained |
| Fixed CLS + mean-patch K-Means | Normalized concatenation of CLS and mean patch tokens; 768 dimensions, no learned adapter |
| GMM / HDBSCAN | Cached normalized CLS; HDBSCAN can discover a different cluster count and marks noise |
| DEC / IDEC | Standalone CLS autoencoder baselines; they are not GLoCA heads |
| CDC | Live-image training with separate clustering and calibration heads; exported predictions use the calibration head |
| ProPos | Live-image two-view training with an EMA target encoder and periodic clustering |
| Supervised MLP | Label-trained reference with stratified 70/15/15 train/validation/test splits; evaluate separately from unsupervised methods |

HDBSCAN noise is remapped to one explicit nonnegative assignment cluster for canonical metrics; noisy samples are not dropped. The supervised MLP selects weights by validation accuracy, breaking ties with validation loss, and exports test-split metrics. Unsupervised results evaluate assignments on the same image collection used to fit the model or clustering method (transductive evaluation).

## Run experiments

Activate `gloca-repro` and run commands from the repository root. Experiments take a YAML config path. Dataset selection, preprocessing, schedules, and output roots live in YAML; seed overrides are the supported command-line experiment overrides.

### Baselines

```powershell
python scripts/run_baselines.py configs/baselines/baselines_unified.yaml --seeds 42 67 69
```

The combined recipe runs nine variants on PlantSeg and PlantVillage: K-Means, spherical K-Means, untrained GLoCA + spherical K-Means, DEC, IDEC, GMM, HDBSCAN, supervised MLP, and fixed CLS + mean-patch K-Means. ProPos is run separately. Existing baseline recipes also use this entry point.

Compatible frozen features are shared within a sweep. Each invocation writes `baseline_summary.csv` and `baseline_summary_agg.csv`, including verified skipped runs. Summaries cover that invocation, so request all desired seeds together. Failed runs appear in the summary and produce a nonzero process exit status.

### Matched CDC experiments

```powershell
python scripts/run_cdc.py configs/cdc/cdc_plantseg_cls_seed42.yaml --seed 42
python scripts/run_cdc.py configs/cdc/cdc_plantseg_gloca_gated_seed42.yaml --seed 42
```

Repeat with the same seed on both paths. The reported CDC seed set is `42, 67, 69, 99, 123`. Corresponding PlantVillage recipes use `plantvillage` in place of `plantseg` in the filenames. Main CDC recipes use 200 epochs, batch size 256, virtual meta-batches of 2,048, and AdamW at `1e-4`. Calibration mini-cluster counts are 160 for PlantSeg and 128 for PlantVillage; these differ from the final head's output count.

### Matched ProPos experiments

```powershell
python scripts/run_propos.py configs/propos/propos_plantseg_cls_full_seed42.yaml --seed 42
python scripts/run_propos.py configs/propos/propos_plantseg_gloca_gated_full_seed42.yaml --seed 42
```

Main ProPos recipes use 100 epochs, batch size 256, AdamW at `5e-4`, a predictor learning-rate multiplier of 10, and a clustering E-step each epoch. The report has three CLS seeds (`42, 67, 69`) but only seed 42 with GLoCA. Do not describe that GLoCA comparison as a multi-seed result.

### Adapter ablations

Run adapter variants through the same CDC entry point:

```powershell
python scripts/run_cdc.py configs/ablation/cdc_plantseg_gloca_mean_pool_seed42.yaml --seed 42
python scripts/run_cdc.py configs/ablation/cdc_plantseg_gloca_no_patches_seed42.yaml --seed 42
python scripts/run_cdc.py configs/ablation/cdc_plantseg_gloca_no_cls_residual_seed42.yaml --seed 42
```

Compare against the CLS and gated GLoCA recipes with the same dataset, seed, schedule, output count, and evaluation. The report's historical ablation rows have capacity and output-count differences; new runs should keep those distinctions explicit.

### Configuration and output locations

Start from a recipe matching the method and dataset. The [configuration template](../configs/templates/full_config_template.yaml) lists the major sections. Important settings include:

| Setting | Purpose |
|---|---|
| `experiment.name`, `experiment.output_dir` | Run name and output root |
| `experiment.seed` | Default seed; replace with `--seed N` without editing the file |
| `dataset.name`, `dataset.root`, `dataset.manifest` | Registered dataset, optional location override, and reference content manifest |
| `dataset.include_classes`, `dataset.limit_per_class` | Optional subsets; change the experiment's meaning |
| `gloca.enabled`, `gloca.name` | Select direct CLS or an adapter variant |
| `head.name`, `head.n_clusters` | Clustering method and output count; `auto` derives the count from the dataset |
| `trainer.max_epochs`, `trainer.batch_size`, `trainer.lr` | Training schedule |
| `trainer.deterministic`, `trainer.resume_from_checkpoint` | Strict execution and checkpoint resume |

Set `experiment.output_dir` to a fresh directory before using historical output locations. For baseline sweeps, this is the top-level `experiment.output_dir`, outside `base_config`; `sweep` selects datasets, seeds, and runs. The [reproducibility guide](REPRODUCIBILITY.md) explains compatibility checks and resume.

Runners compose their destination as:

```text
<experiment.output_dir>/<experiment.name>/<head.name>/<dataset.name>/seed_<seed>/
```

Some baseline sweep families add a dataset/seed prefix to the effective output root before this composition. Use the output location printed by the script or recorded in the summary.

## Read the outputs

| File | Contents |
|---|---|
| `config.yaml` | Effective run configuration, including the selected seed |
| `assignments.json` | Image IDs, evaluation labels, predicted cluster IDs, and method metadata |
| `metrics.csv` | Canonical final metrics and method-specific scalar fields |
| `embeddings.pt` | Final embedding tensor in assignment image order |
| `logs.json` | Training history, diagnostics, and timing information |
| `reproducibility.json` | Input, source, configuration, and runtime identity |
| `completed.json` | Artifact hashes written when the run finishes |
| `attention.pt` | Patch attention, when available |
| `checkpoint.ckpt` | Final export for trainable methods |
| `checkpoints/latest.ckpt` | Periodic resumable checkpoint, when checkpointing is enabled |
| `checkpoint_metrics.csv` | Interval evaluation history, when enabled; separate from final metrics |

CDC additionally exports `confidence.pt`, `calibrated_confidence.pt`, and `pseudo_labels.pt`. Keep tensors and assignment metadata from the same run together.

Assignment metadata includes `head`, `backbone`, `gloca`, `dataset`, `seed`, `n_clusters`, `image_ids`, `labels`, `assignments`, and `patch_grid`. The image, label, and assignment arrays have matching lengths. Cluster IDs are anonymous and can be permuted between runs.

ARI and NMI measure agreement with evaluation labels. ACC is clustering accuracy after Hungarian cluster-to-label alignment, distinct from the supervised classifier's direct accuracy. Silhouette describes embedding geometry, using at most 2,000 sampled images with a fixed sampling seed; it can improve while label agreement worsens. ECE measures calibration error, so lower values are preferable. Reliable-sample coverage describes training selection and does not establish selected-pseudo-label correctness.

Metrics use native scales: NMI, ACC, ECE, and reliable ratios are in [0, 1]; ARI and silhouette can be negative. Multiply accuracy or ratio differences by 100 when expressing them in percentage points.

Inspect occupancy (`n_nonempty_clusters`, cluster sizes, and entropy) alongside quality metrics. For GLoCA, interpret alpha, CLS cosine, correction norms, and attention statistics jointly. A changing embedding demonstrates learning, not necessarily better disease discovery. The final exported scores are separate from peak scores in checkpoint histories; do not silently select checkpoints using evaluation labels.

## Visualize attention

For a run containing `attention.pt` and `assignments.json`:

```powershell
python scripts/visualize_gloca_attention.py --run-dir outputs/main_experiments/gloca_gated/cdc/plantseg/seed_42 --dataset plantseg --n-per-class 3
```

The command needs the original dataset at its registered location and writes `attention_visualizations/` beneath the run directory by default. It projects patch attention through the evaluation crop geometry onto the original image. Use `--output-dir` to choose another destination. Attention maps are diagnostics, not final proof of disease localization or cluster correctness.

## Explore an uploaded image

```powershell
python -m streamlit run streamlit_app.py
```

Select PlantSeg or PlantVillage, upload a JPG, JPEG, PNG, BMP, or WebP image (maximum 10 MB), and choose how many clusters to display. The app ranks clusters by calibration-head softmax affinity and shows five original dataset exemplars per cluster, ranked using saved embeddings. CUDA is used when available; otherwise inference uses CPU. The first load may take longer while the backbone and checkpoint load.

The app requires the original images at their registered locations and these matching seed-42 CDC + gated GLoCA artifacts:

```text
outputs/main_experiments/gloca_gated/cdc/<dataset>/seed_42/
  config.yaml
  assignments.json
  embeddings.pt
  checkpoints/latest.ckpt
```

These artifacts are not bundled with the repository. Keep the checkpoint, config, embeddings, and assignments from the same supported run together; an arbitrary new training output is not automatically supported.

If the app reports a config or head-size mismatch, the supplied bundle is incompatible. Changing only the YAML output count does not establish compatibility with the checkpoint and saved embeddings.

Cluster probabilities indicate affinity to anonymous learned groups. They do not establish a disease name, cluster correctness, or diagnosis confidence.
