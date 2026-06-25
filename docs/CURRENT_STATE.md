# GLoCA Repository Current State

This document describes the repository as it exists now. It is a present-tense reference for code structure, supported experiment paths, output schemas, and the current experimental interpretation.

## Research Frame

The repository studies plant disease clustering with a frozen DINOv2 backbone and an optional GLoCA embedding adapter. The controlled comparison pattern is:

```text
DINO CLS + clustering method
DINO + GLoCA + same clustering method
```

The intended controlled variables are dataset, seed, DINOv2 backbone, clustering method, evaluation code, and output schema. The intended difference in a matched comparison is the embedding source:

```text
normalized DINO CLS embedding
GLoCA embedding
```

GLoCA is an embedding adapter, not a clustering method. Clustering methods consume the final embedding tensor produced by the shared model path.

The main reported experiment scope is now:

```text
plantvillage
plantseg
```

`plantwild` / PlantWild v2 remains registered and useful for smoke tests, debugging, and historical artifact inspection, but it is not a main reported experiment dataset. PlantSeg has been identified as a near duplicate of PlantWild v2, so PlantWild results should not be treated as independent reported evidence.

## Active Layout

The active source tree is:

```text
src/
  data/
    datamodule.py
    folder_dataset.py
    registry.py
    transforms.py

  diagnostics/
    gloca.py

  evaluation/
    assignment_schema.py
    clustering_metrics.py

  experiments/
    config.py
    outputs.py
    registry.py

  features/
    dinov2.py
    dinov2_backbone.py

  models/
    base.py
    gloca.py

    baselines/
      dec_idec.py

    clustering/
      base.py
      cdc.py
      kmeans.py
      propos.py

  runners/
    common.py
    cdc.py
    dec_idec.py
    diagnostics.py
    embedding_export.py
    kmeans.py
    propos.py

  training/
    checkpointing.py
    cdc_trainer.py
    dec_idec_trainer.py
    propos_trainer.py

  utils.py

scripts/
  prepare_plantseg_folder_dataset.py
  run_baselines.py
  run_cdc.py
  run_propos.py
  run_propos_gloca_diagnostics.py
  visualize_gloca_attention.py
```

The supported script entry points are YAML-driven:

```text
python scripts/run_baselines.py <config_path>
python scripts/run_cdc.py <config_path>
python scripts/run_propos.py <config_path>
python scripts/run_propos_gloca_diagnostics.py <config_path>
python scripts/visualize_gloca_attention.py --run-dir <run_dir> --dataset <name> --n-per-class <n>
```

`scripts/run_baselines.py`, `scripts/run_cdc.py`, `scripts/run_propos.py`, and `scripts/run_propos_gloca_diagnostics.py` each parse one positional YAML config path.

## Configuration

Config loading and validation live in `src/experiments/config.py`. The template file is:

```text
configs/templates/full_config_template.yaml
```

The present config sections are:

```text
experiment
dataset
backbone
gloca
head
baseline
propos
cdc
prediction
trainer
base_config
sweep
diagnostics
```

The checked-in runnable configs are:

```text
configs/baselines/baselines_full.yaml
configs/baselines/baselines_smoke.yaml
configs/propos/propos_plantwild_cls_full_seed42.yaml
configs/propos/propos_plantwild_cls_smoke.yaml
configs/propos/propos_plantwild_gloca_gated_full_seed42.yaml
configs/propos/propos_plantwild_gloca_gated_smoke.yaml
configs/diagnostics/propos_gloca_diagnostics_plantwild_seed42.yaml
configs/cdc/cdc_plantwild_cls_smoke.yaml
configs/cdc/cdc_plantwild_gloca_gated_smoke.yaml
configs/cdc/cdc_plantwild_cls_seed42.yaml
configs/cdc/cdc_plantwild_gloca_gated_seed42.yaml
```

Runtime settings include `trainer.matmul_precision`, accepted values `highest`, `high`, `medium`, or null, and `trainer.num_workers` for dataloader throughput. Trainable runners also support resumable checkpointing fields:

```yaml
trainer:
  checkpoint_interval: 10
  eval_interval: checkpoint
  resume_from_checkpoint: auto
  profile_resources: true
```

`resume_from_checkpoint: auto` resumes from `<run_dir>/checkpoints/latest.ckpt` when it exists. `null` starts fresh, and an explicit path loads that checkpoint. `eval_interval: checkpoint` runs expensive trainer-level evaluation at checkpoint epochs and the final epoch; final runner export/evaluation still always runs.

## Dataset Layer

The dataset abstraction is `FolderImageDataset` in `src/data/folder_dataset.py`.

Expected folder layout:

```text
data/raw/<dataset_name>/
  <class_name>/
    image_001.jpg
    image_002.png
```

Supported image extensions are:

```text
.jpg .jpeg .png .bmp .webp
```

Single-view samples expose:

```python
{
    "index": int,
    "image": Tensor,
    "label": int,
    "label_name": str,
    "image_id": str,
    "dataset": str,
}
```

Two-view training samples expose:

```python
{
    "views": (Tensor, Tensor),
    "label": int,
    "label_name": str,
    "image_id": str,
    "index": int,
    "dataset": str,
}
```

`ClusteringDataModule` is a plain PyTorch dataloader helper. It supports `single_view` and `contrastive_two_view` training modes. When `training_views: auto`, it resolves to one view for `single_view` and two views for `contrastive_two_view`.

CDC uses an opt-in `cdc` training mode with:

```yaml
dataset:
  training_views: cdc_weak_strong_calibration
```

CDC training samples expose:

```python
{
    "weak": Tensor,
    "strong": Tensor,
    "calibration": Tensor,
    "views": (weak, strong),
    "label": int,
    "label_name": str,
    "image_id": str,
    "index": int,
    "dataset": str,
}
```

The weak view uses the current training crop/resize pipeline. The strong view uses the CDC reference RandAugment-like operator list plus optional Cutout. The calibration view uses deterministic prediction preprocessing. This view mode is only active when requested by CDC configs and does not change ProPos two-view behavior.

Training transforms use PIL/torchvision:

```text
RandomResizedCrop
RandomHorizontalFlip
ColorJitter
RandomGrayscale
ToTensor
ImageNet normalization
```

Prediction/export transforms are deterministic:

```text
Resize(round(image_size * 256 / 224))
CenterCrop(image_size)
ToTensor
ImageNet normalization
```

The dataset registry names and default roots are:

```text
plantseg      -> data/raw/plantseg_folder
plantvillage  -> data/raw/plantvillage
plantwild     -> data/raw/plantwild_v2
```

For reporting, use PlantVillage and PlantSeg. Treat PlantWild as smoke/debug-only even though it remains available through the registry and existing configs/artifacts.

`DatasetSpec` also supports `root`, `limit_per_class`, and `include_classes`.

## Backbone

The feature backbone is DINOv2. The supported variant set contains:

```text
facebook/dinov2-small
```

`DINOv2Backbone` in `src/features/dinov2_backbone.py`:

- loads with HuggingFace `AutoModel.from_pretrained`,
- freezes parameters when `freeze: true`,
- keeps the underlying model in eval mode,
- runs the model under `torch.no_grad()`,
- extracts CLS from `last_hidden_state[:, 0, :]`,
- extracts patch tokens from `last_hidden_state[:, 1:, :]`,
- returns `cls`, `patch_tokens`, and `patch_grid`.

At `image_size: 224` with DINOv2-S/14, the expected geometry is:

```text
CLS:          [B, 384]
Patch tokens: [B, 256, 384]
Patch grid:   (16, 16)
```

## GLoCA

All GLoCA adapter code lives in `src/models/gloca.py`.

Implemented adapter names:

```text
cls
gloca_sum
gloca_gated
```

Direct DINO CLS is represented in config by:

```yaml
gloca:
  enabled: false
  name: disabled
```

`build_adapter(config, input_dim)` returns `None` when GLoCA is disabled.

All adapter forwards return:

```python
{
    "embedding": Tensor,
    "attention": Tensor | None,
    "patch_grid": tuple[int, int] | None,
}
```

`CLSAdapter` projects CLS through a residual MLP and optionally normalizes the output.

`GLoCASumAdapter` computes simple attention over patch tokens, projects pooled patch information, adds it to the projected CLS vector, applies a residual MLP, and optionally normalizes the output.

`GLoCAGatedAdapter` is the main conservative residual adapter:

```text
z_cls = W_cls(CLS)
z_patch = W_patch(gated_attention_pool(patch_tokens))
delta = MLP(concat(z_cls, z_patch))
h = normalize(z_cls + alpha * delta)
```

`alpha` is a learnable scalar initialized from config, commonly `0.0`. Attention tensors use shape `[B, N]`. The adapter exposes a `diagnostics()` method that also returns `z_cls`, `z_patch`, and `delta`.

Tests assert that `gloca_gated` with `alpha_init: 0.0` and matching input/output dimensions preserves normalized CLS closely at initialization, while `alpha` still receives gradients.

## Shared Model Composition

`ClusteringBaseModel` in `src/models/base.py` owns the shared composition:

```text
DINOv2 backbone -> optional GLoCA adapter -> final embedding -> clustering head
```

`encode_view(image)` is the shared embedding path. With no adapter, it returns normalized DINO CLS when `normalize_cls` is true. With an adapter, it returns the adapter output.

`forward(image)` calls `encode_view()` and passes only `encoded["embedding"]` into the head.

`forward_views((view1, view2))` applies the same shared path to both views.

## Implemented Clustering and Baseline Methods

### K-Means and Spherical K-Means

Implementation files:

```text
src/models/clustering/kmeans.py
src/runners/kmeans.py
```

The model layer provides:

```text
TorchKMeans
TorchSphericalKMeans
fit_kmeans
torch_kmeans
torch_spherical_kmeans
```

Supported initialization modes are:

```text
random
kmeans++
```

`run_kmeans(config)` extracts deterministic embeddings with frozen DINOv2 and optional untrained GLoCA, then fits torch K-Means. When `baseline.spherical: true`, the effective output head is `spherical_kmeans`; otherwise it is `kmeans`.

K-Means runs report `uses_cached_backbone_features: true`.

### DEC and IDEC

Implementation files:

```text
src/models/baselines/dec_idec.py
src/training/dec_idec_trainer.py
src/runners/dec_idec.py
```

`DINOCLSDECModel` is an autoencoder over cached deterministic DINO CLS embeddings. It includes:

```text
encoder
decoder
cluster_centers
Student-t soft assignment
target distribution computation
```

`run_dec_idec(config)` supports `head.name: dec` and `head.name: idec`. These runs require GLoCA disabled. The runner:

- extracts deterministic DINO CLS embeddings,
- pretrains the autoencoder through the phase-aware trainer,
- initializes cluster centers with torch K-Means when needed,
- refines the DEC/IDEC objective through the phase-aware trainer,
- writes a checkpoint.

DEC/IDEC resumable checkpoints are phase-aware and can resume from `pretrain`, `cluster_init`, `refine`, or `complete` states without repeating completed phases.

DEC refinement trains the encoder and cluster centers. IDEC refinement trains the encoder, decoder, and cluster centers with reconstruction loss.

The target update parser treats null, empty, `"none"`, `"null"`, `"fixed"`, and `0` as fixed-target mode. A positive integer enables periodic target refresh.

### ProPos

Implementation files:

```text
src/models/clustering/propos.py
src/training/propos_trainer.py
src/runners/propos.py
scripts/run_propos.py
```

ProPos is the live-image, two-view trainable clustering method in this repository. It preserves the shared path:

```text
view -> frozen DINOv2 -> optional GLoCA -> embedding -> ProPos head
```

`ProPosHead` contains:

```text
projector
target_projector
predictor
pseudo_labels buffer
positive_sampling_alignment
prototype_scattering_loss
EMA target projector update
```

`ProPosTrainer`:

- uses stochastic two-view training,
- keeps DINOv2 frozen,
- keeps a target encoder with an EMA copy of the optional adapter,
- runs E-steps with spherical torch K-Means over deterministic target projections,
- stores pseudo-labels by dataset index,
- computes PSA and PSL losses,
- applies warmup with PSL and latent noise inactive while `epoch <= warmup_epochs`,
- supports symmetric loss,
- supports AdamW only,
- supports separate optimizer groups for projector, predictor, GLoCA parameters, and GLoCA alpha,
- supports `freeze_gloca`, `freeze_gloca_epochs`, `gloca_lr_multiplier`, and `gloca_alpha_lr_multiplier`,
- saves resumable checkpoints under `checkpoints/`,
- logs optional GLoCA diagnostics and separated resource timing totals.

After training, `run_propos(config)` exports deterministic single-view target projections, fits final spherical torch K-Means, computes metrics, writes `checkpoint.ckpt`, and writes standard output files.

ProPos rows append method-specific metrics:

```text
loss_psa_final
loss_psl_final
loss_total_final
kmeans_interval
warmup_epochs
lambda_psl
sigma
temperature
ema_momentum
ema_momentum_final
projection_dim
n_empty_cluster_batches
n_invalid_psl_batches
kmeans_backend
kmeans_init
```

ProPos logs include E-step history, epoch history, recent step logs, target EMA status, GLoCA alpha values, GLoCA trainability, resource timing totals, checkpoint/eval schedule fields, and the final K-Means logs. The E-step schedule is unchanged by `eval_interval`.

### CDC

Implementation files:

```text
src/models/clustering/cdc.py
src/training/cdc_trainer.py
src/runners/cdc.py
scripts/run_cdc.py
```

CDC is implemented as a live-image trainable clustering method over the shared embedding path:

```text
weak / strong / calibration view -> frozen DINOv2 -> optional GLoCA -> embedding -> CDC heads
```

`CDCHead` contains a clustering head and a calibration head. Both are:

```text
Linear(input_dim, hidden_dim) -> BatchNorm1d -> ReLU -> Linear(hidden_dim, n_clusters)
```

The default hidden dimension in the provided CDC configs is `512`, and the input dimension is not hard-coded. Forward outputs include clustering logits, calibration logits, probabilities, predictions, and confidence tensors.

`CDCTrainer`:

- treats `trainer.batch_size` as the physical dataloader batch size and `cdc.meta_batch_size` as the virtual CDC statistical batch size,
- accumulates physical batches into `CDCMetaBatch` objects before reliable sample selection and calibration target construction,
- uses the weak view for no-gradient calibration-head confidence and reliable pseudo-label selection over the whole meta-batch,
- stores physical-dataloader strong tensors on CPU inside the virtual meta-batch,
- uses the stored strong view for clustering-head cross-entropy on selected reliable samples, optimized in `cdc.sub_batch_size` chunks,
- uses the calibration view for detached mini-cluster target construction over the whole meta-batch and calibration-head loss in chunks,
- supports `cdc.meta_batch_drop_last` for dropping or processing a final partial meta-batch,
- keeps calibration loss gradients out of DINOv2, GLoCA, and the clustering head,
- keeps DINOv2 frozen,
- supports optimizer groups for CDC clustering head, CDC calibration head, optional GLoCA body parameters, and optional GLoCA alpha parameters,
- saves resumable checkpoints under `checkpoints/`,
- logs losses, configured and actual meta-batch sizes, selected sample counts, reliable sample ratio, confidence stats, pseudo-label entropy, skipped meta-batch/chunk counts, CDC initialization mode, checkpoint/eval schedule fields, separated resource timing totals, and optional GLoCA diagnostics.
- uses tqdm for live batch and epoch progress, and reports deterministic calibration-head NMI, ARI, and ACC only when `trainer.eval_interval` schedules trainer-level evaluation.

The provided CDC PlantWild configs are retained for smoke/debugging and historical reproducibility. They use this virtual-batch pattern:

```yaml
trainer:
  batch_size: 128
  num_workers: 4
  pin_memory: true
  persistent_workers: true
cdc:
  meta_batch_size: 2048
  sub_batch_size: 128
  per_class_selected_num: auto
  calibration_k: 160
  meta_batch_drop_last: false
```

With `per_class_selected_num: auto`, reliable selection is computed from the actual meta-batch size. For PlantWild-115, a full 2048-sample meta-batch selects up to `2048 // 115 = 17` samples per predicted class. Small smoke subsets can produce a smaller final partial meta-batch, so their selected-per-class count can be lower. PlantWild CDC runs should be interpreted as smoke/debug runs, not as main reported results.

CDC initialization attempts the reference-style deterministic embedding prototype initialization with local torch K-Means:

```text
embeddings -> row z-score/normalize -> K-Means(hidden_dim) for W1
hidden activations -> row z-score/normalize -> K-Means(n_clusters) for W2
```

When `cdc.orthogonalize_init: true`, CDC additionally runs the original CDC `orth_train`-style prototype row refinement after both K-Means stages and before copying weights into the clustering and calibration heads. The local port is CUDA-optional and keeps the reference defaults of scale `5.0` and `2000` optimization epochs unless `cdc.orthogonalize_scale` or `cdc.orthogonalize_epochs` are provided. Logs record `cdc_init_mode: prototype_kmeans_orthogonalized` and the per-layer orthogonalization losses. If the prototype path is not feasible, for example because a smoke subset has fewer samples than `hidden_dim`, CDC keeps random initialization and writes `cdc_init_mode: random` plus a fallback reason to `logs.json`.

Final CDC prediction uses the calibration head by default, matching the CDC-Cal interpretation. The runner writes the standard canonical files plus `checkpoint.ckpt`, `confidence.pt`, `calibrated_confidence.pt`, and `pseudo_labels.pt`. GLoCA runs also write `attention.pt`.

CDC appends method-specific metrics:

```text
clustering_confidence_mean
clustering_confidence_std
calibrated_confidence_mean
calibrated_confidence_std
reliable_sample_ratio
calibration_threshold
calibration_ece
calibration_mce
pseudo_label_count
pseudo_label_entropy
cdc_pretrain_epochs
cdc_refine_epochs
cdc_init_mode
```

Checked smoke commands:

```text
python scripts/run_cdc.py configs/cdc/cdc_plantwild_cls_smoke.yaml
python scripts/run_cdc.py configs/cdc/cdc_plantwild_gloca_gated_smoke.yaml
```

Both checked smoke runs complete end-to-end and write canonical outputs. In the smoke subset, prototype initialization falls back to random because `n_samples=230` is smaller than `hidden_dim=512`. The one-epoch smoke runs produce finite confidence tensors and non-collapsed assignment IDs, but they are not research-result runs.

## ProPos/GLoCA Diagnostics

The focused diagnostic script is:

```text
scripts/run_propos_gloca_diagnostics.py
```

The checked-in diagnostic config defines five runs on PlantWild seed 42:

```text
A: CLS
B: frozen gloca_gated
C: trainable gloca_gated with lower GLoCA body LR and alpha LR 1x
D: trainable gloca_gated with lower GLoCA body LR and alpha LR 10x
E: trainable gloca_gated with body LR 1x and alpha LR 10x
```

The script writes:

```text
propos_gloca_diagnostic_summary.csv
propos_gloca_diagnostic_report.md
```

GLoCA scalar diagnostics live in `src/diagnostics/gloca.py`:

```text
gloca_alpha_value
gloca_alpha_grad_norm
gloca_param_grad_norm
gloca_delta_norm_mean
gloca_delta_norm_std
gloca_embedding_cls_cosine_mean
gloca_embedding_cls_cosine_std
gloca_attention_entropy_mean
gloca_attention_max_mean
```

## Attention Visualization

`scripts/visualize_gloca_attention.py` visualizes saved GLoCA attention for runs with `attention.pt`.

Inputs:

```text
--run-dir       directory containing attention.pt, assignments.json, and optionally config.yaml
--dataset       plantseg, plantvillage, or plantwild
--n-per-class   number of samples per class
--output-dir    optional destination
--seed          deterministic sample selection seed
--alpha         overlay opacity
--cmap          colormap
```

The script validates:

- `attention.pt` exists,
- `assignments.json` exists,
- attention has shape `[B, N]`,
- the run dataset matches `--dataset`,
- `image_ids`, `labels`, and `assignments` have equal length,
- attention columns match `patch_grid[0] * patch_grid[1]`.

It maps attention through the repository's deterministic eval resize/center-crop geometry and overlays the heatmap on the original dataset image.

## Baseline Sweep

The baseline sweep entry point is:

```text
scripts/run_baselines.py
```

`configs/baselines/baselines_full.yaml` defines:

```text
datasets: plantvillage, plantwild, plantseg
seeds: 42, 69, 67
runs:
  kmeans_cls
  spherical_kmeans_cls
  kmeans_gloca_gated_untrained
  dec_cls
  idec_cls
```

Although the checked-in full baseline sweep still includes PlantWild, PlantWild rows are smoke/debug or legacy context only. The main reported baseline comparisons should use PlantVillage and PlantSeg.

`configs/baselines/baselines_smoke.yaml` defines a PlantWild seed-42 smoke sweep with two K-Means runs and `limit_per_class: 2`.

The baseline script writes:

```text
baseline_summary.csv
baseline_summary_agg.csv
```

The aggregate grouping keys are:

```text
backbone
gloca
head
dataset
```

## Output Contract

Runner output directories use:

```text
<experiment.output_dir>/<experiment.name>/<head.name>/<dataset.name>/seed_<seed>/
```

Standard files:

```text
config.yaml
assignments.json
metrics.csv
embeddings.pt
logs.json
```

Conditional files:

```text
attention.pt       # written when attention exists
checkpoint.ckpt    # written by trainable runners
confidence.pt      # written by CDC
calibrated_confidence.pt  # written by CDC
pseudo_labels.pt   # written by CDC
```

Trainable runners additionally write periodic resumable checkpoints:

```text
checkpoints/
  epoch_0010.ckpt
  epoch_0020.ckpt
  latest.ckpt
```

`latest.ckpt` is a real copied file for Windows compatibility, not a symlink. The final `checkpoint.ckpt` artifact remains separate and unchanged for canonical output compatibility.

Trainable runners append checkpoint-time metrics to:

```text
checkpoint_metrics.csv
```

This file is append-only during training and records rows for interval checkpoint epochs. It does not replace the canonical one-row final `metrics.csv`.

Output writing is centralized in `src/experiments/outputs.py`. Assignment payloads are validated before writing.

## Assignment Schema

`src/evaluation/assignment_schema.py` defines required assignment fields:

```text
head
backbone
gloca
dataset
seed
n_clusters
image_ids
labels
assignments
patch_grid
```

These arrays must have equal length:

```text
image_ids
labels
assignments
```

## Metrics Schema

Canonical per-run metric fields are:

```text
experiment
head
backbone
dataset
seed
gloca
n_clusters
n_images
ari
nmi
acc
silhouette
n_nonempty_clusters
cluster_size_min
cluster_size_max
cluster_size_entropy
embedding_variance_mean
embedding_norm_mean
embedding_norm_std
attention_entropy
attention_max
attention_top5_mass
attention_variance
backbone_cache_time_s
head_train_time_s
total_time_s
inference_time_s
peak_gpu_mb
uses_cached_backbone_features
```

DEC/IDEC append:

```text
input_dim
hidden_dims
latent_dim
pretrain_epochs
refine_epochs
pretrain_lr
refine_lr
lambda_recon
alpha
target_update_mode
target_update_interval
```

ProPos appends:

```text
loss_psa_final
loss_psl_final
loss_total_final
kmeans_interval
warmup_epochs
lambda_psl
sigma
temperature
ema_momentum
ema_momentum_final
projection_dim
n_empty_cluster_batches
n_invalid_psl_batches
kmeans_backend
kmeans_init
```

CDC appends:

```text
clustering_confidence_mean
clustering_confidence_std
calibrated_confidence_mean
calibrated_confidence_std
reliable_sample_ratio
calibration_threshold
calibration_ece
calibration_mce
pseudo_label_count
pseudo_label_entropy
cdc_pretrain_epochs
cdc_refine_epochs
cdc_init_mode
```

Canonical aggregate fields are:

```text
backbone
gloca
head
dataset
n_runs
seeds
ari_mean
ari_std
nmi_mean
nmi_std
acc_mean
acc_std
silhouette_mean
silhouette_std
total_time_s_mean
total_time_s_std
peak_gpu_mb
```

Clustering metrics use:

```text
ARI: sklearn adjusted_rand_score
NMI: sklearn normalized_mutual_info_score
ACC: Hungarian-aligned clustering accuracy
Silhouette: sklearn silhouette_score with sample_size <= 2000
```

Diagnostics include cluster occupancy, normalized cluster-size entropy, embedding variance/norm statistics, and attention entropy/max/top-5 mass/variance when attention exists.

## Historical PlantWild ProPos Findings

`docs/report_1.md` records a historical PlantWild 200-epoch ProPos analysis for these seed-42 runs:

```text
outputs/propos_full/propos_cls_full_200ep/propos/plantwild/seed_42
outputs/propos_full/propos_gloca_gated_full_200ep/propos/plantwild/seed_42
```

The two runs share dataset, seed, backbone, head family, evaluation code, and output schema. Their ProPos schedules differ in `kmeans_interval`: CLS uses `1`, and GLoCA-gated uses `2`.

Because PlantWild is now smoke/debug-only and PlantSeg has been identified as a near duplicate of PlantWild v2, these PlantWild ProPos findings should not be used as main reported experiment evidence.

Final metrics:

| Run | GLoCA | ARI | NMI | ACC | Silhouette | Final total loss | Final PSA | Final PSL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ProPos CLS, 200 ep | disabled | 0.326506 | 0.666694 | 0.400853 | 0.114619 | -1.523476 | -1.764363 | 2.408874 |
| ProPos GLoCA-gated, 200 ep | gloca_gated | 0.326628 | 0.654514 | 0.417653 | 0.132384 | -1.580791 | -1.823843 | 2.430524 |

Best observed epoch-history values:

| Run | Best ARI epoch | Best ARI | Best ACC epoch | Best ACC | Best NMI epoch | Best NMI |
|---|---:|---:|---:|---:|---:|---:|
| ProPos CLS | 58 | 0.350289 | 70 | 0.433670 | 155 | 0.674839 |
| ProPos GLoCA-gated | 66 | 0.352669 | 118 | 0.443332 | 66 | 0.674425 |

GLoCA learning diagnostics from the 200-epoch GLoCA run include:

```text
alpha: 0.0 initialization to -8.4046
embedding-to-CLS cosine: about 0.537 near the end of training
training attention entropy: about 4.82 near the end of training diagnostics
final metrics attention_entropy: 4.868949
final metrics attention_max: 0.039434
final metrics attention_top5_mass: 0.150596
```

The current interpretation is:

- ProPos is a working implemented method in this repository.
- `GLoCA-gated + ProPos` is stable under the recorded PlantWild run.
- GLoCA changes the embedding substantially under ProPos.
- GLoCA improves ProPos total loss and PSA loss in the recorded run.
- The learned patch correction is not clearly better aligned with disease-label clustering than the CLS path.
- The bottleneck shown by these runs is objective alignment, not absence of GLoCA learning.

The report also records qualitative attention-map findings. GLoCA attention sometimes overlaps visible disease evidence and sometimes emphasizes background, borders, artifacts, non-lesion object regions, or other visual factors. Attention localization and final cluster alignment are related diagnostics, but they are not equivalent: disease-localizing attention can still produce fragmented clusters, and cohesive clusters can appear without lesion-centered attention.

The report records a preprocessing observation as well: full-frame-preserving resize-pad and resize-stretch trials do not improve the observed ProPos-CLS trajectory over the current crop/resize pipeline in the noted preliminary runs.

## Tests

The test suite is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Current tests cover:

- folder dataset and datamodule behavior,
- DEC/IDEC target update parsing and tiny cached training,
- conservative GLoCA initialization,
- GLoCA alpha gradients,
- direct CLS behavior with no adapter,
- GLoCA diagnostics scalar outputs,
- attention visualization helpers,
- ProPos projection/predictor shapes,
- ProPos PSA and PSL behavior,
- ProPos EMA target updates,
- frozen DINOv2 behavior in the clustering model,
- target adapter EMA updates,
- ProPos optimizer parameter groups,
- GLoCA freeze controls,
- ProPos pseudo-label lookup by dataset index,
- torch K-Means and spherical K-Means behavior,
- ProPos import structure,
- CDC head outputs, reliable-sample edge cases, calibration stop-gradient behavior, initialization fallback logs, CDC named dataloader views, CDC trainer view consumption, CDC optimizer freezing behavior, and CDC script parsing,
- resumable checkpoint helpers, checkpoint-gated evaluation scheduling, and CDC/ProPos/DEC-IDEC resume behavior,
- assignment array length validation,
- runtime matmul precision settings,
- experiment script config-path parsing,
- config template section coverage.

## Present Policy Summary

- The main comparison pattern is same dataset, same seed, same backbone, same clustering method, and CLS versus GLoCA embedding.
- The main reported experiment datasets are PlantVillage and PlantSeg.
- PlantWild / PlantWild v2 is smoke/debug-only and should not be reported as independent main evidence.
- DINOv2 is frozen in the implemented runners.
- The shared embedding path is `ClusteringBaseModel.encode_view()`.
- Clustering methods consume the final embedding tensor only.
- K-Means and spherical K-Means are deterministic cached-embedding baselines.
- DEC and IDEC are standalone DINO CLS autoencoder baselines.
- ProPos is the live two-view trainable clustering method.
- GLoCA variants live in `src/models/gloca.py`.
- Output writing lives in `src/experiments/outputs.py`.
- Assignment and metrics schemas live in `src/evaluation/assignment_schema.py`.
- Long histories, training details, diagnostics, and method notes live in `logs.json`.
