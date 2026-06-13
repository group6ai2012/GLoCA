# GLoCA Modular Implementation Plan — v6.3 ProPos/GLoCA Diagnostics and CDC Direction

This plan replaces the old v5 planning document as the active implementation roadmap. It is aligned with the implemented repository state recorded in `CURRENT_STATE.md` and updates the future direction after the baseline cleanup, ProPos implementation, and GLoCA diagnostic work.

The current working tree already includes an explicit PyTorch baseline layer, an implemented ProPos runner/trainer, torch spherical K-Means, and ProPos/GLoCA diagnostics. The next new deep clustering method remains **CDC — Calibrated Deep Clustering**, while ProPos remains the current active method for diagnosing whether GLoCA attention can make embeddings more disease-centered.

Structural cleanup update: experiment scripts are now YAML-config-only, K-Means and spherical K-Means live as pure PyTorch implementations in `src/models/clustering/kmeans.py`, and production code must not reintroduce `sklearn.cluster.KMeans`.

The core design philosophy is unchanged:

```text
DINO CLS + clustering method
vs.
DINO + GLoCA + same clustering method
```

GLoCA remains independent from clustering methods. Clustering methods consume only the final embedding tensor. Dataset loading, DINOv2 extraction, GLoCA composition, clustering objectives, metrics, and output writing should stay modular and swappable.

Post-hoc XAI remains intentionally excluded from this stage. GLoCA attention maps are saved for later PlantSeg/XAI analysis, but PlantSeg masks, Grad-CAM, Eigen-CAM, CRAFT, and deletion metrics are not part of the current implementation stage.

---

## 0. Version 6.3 Update Summary

Version 6.3 updates the plan after the baseline suite, explicit-trainer cleanup, ProPos implementation, GLoCA initialization fix, and ProPos/GLoCA diagnostic tooling.

Major changes from v6.1:

- Treat `CURRENT_STATE.md` as the source of truth for implemented code.
- Mark data, DINOv2, GLoCA, K-Means, DEC/IDEC, StudentT diagnostics, schemas, baseline sweeps, torch spherical K-Means, and ProPos as implemented.
- Mark ProPos as the current active live-image clustering method under diagnostic study, not merely a future phase.
- Keep **CDC** as the next new deep clustering method to port after the current ProPos/GLoCA diagnostic pass.
- Keep **CoHiClust** as the later hierarchical method.
- Demote SPICE to optional legacy/reference work only.
- Preserve the central comparison: same clustering method with and without GLoCA.
- Preserve the rule that DEC/IDEC are standalone DINO CLS baselines, not GLoCA heads.
- Preserve StudentT as a diagnostic/smoke head only.
- Preserve the explicit-trainer direction: current trainable logic lives in `src/training/` and is orchestrated by runners.
- Add CDC-specific calibration outputs without breaking the canonical metrics schema.
- Use already cloned official repositories as reference implementations, but port only the necessary method logic into the GLoCA codebase.

Current and future main methods:

```text
ProPos      # implemented; current GLoCA diagnostic method
CDC         # next new method to port
CoHiClust   # later hierarchical method
```

Main paper comparisons remain:

```text
DINO CLS + ProPos     vs. DINO + GLoCA + ProPos
DINO CLS + CDC        vs. DINO + GLoCA + CDC
DINO CLS + CoHiClust  vs. DINO + GLoCA + CoHiClust
```

---

## 1. Current Implemented State

The repository currently supports experiments around frozen DINOv2 features, optional GLoCA embeddings, and clustering baselines for plant disease discovery.

The current implemented comparison surface is:

```text
DINO CLS + clustering baseline
DINO + GLoCA + same clustering baseline or diagnostic head
```

The active code layout is:

```text
src/
  data/
    folder_dataset.py
    registry.py
    datamodule.py
    transforms.py

  features/
    dinov2.py
    dinov2_backbone.py

  models/
    base.py
    gloca.py

    clustering/
      base.py
      kmeans.py
      student_t.py
      propos.py

    baselines/
      dec_idec.py

  training/
    kmeans.py
    dec_idec_trainer.py
    propos_trainer.py

  runners/
    common.py
    kmeans.py
    student_t.py
    dec_idec.py
    propos.py
    embedding_export.py
    diagnostics.py

  evaluation/
    assignment_schema.py
    clustering_metrics.py

  experiments/
    config.py
    outputs.py
    registry.py

  utils.py

scripts/
  prepare_plantseg_folder_dataset.py
  run_baselines.py
  run_propos.py
  run_propos_gloca_diagnostics.py
```

Important current-state constraints:

- `scripts/run_baselines.py` is the supported baseline sweep entry point.
- `scripts/run_propos.py` is the supported ProPos entry point.
- `scripts/run_propos_gloca_diagnostics.py` is the supported ProPos/GLoCA diagnostic entry point.
- Experiment scripts accept exactly one positional YAML config path and must not grow hidden CLI-defined experiment defaults.
- K-Means and spherical K-Means are pure PyTorch through `src.models.clustering.kmeans`; do not reintroduce `sklearn.cluster.KMeans`.
- Deprecated one-off Phase 5 scripts have been removed and must not be reintroduced.
- Current trainable logic lives in `src/training/` and is orchestrated by runners.
- `src/models/heads/` and `src/models/adapters/` have been removed or replaced.
- Active imports should use `src.models.clustering`, `src.models.gloca`, `src.models.baselines`, `src.runners`, and `src.training`.
- GLoCA lives in `src/models/gloca.py`.
- GLoCA variants currently implemented: `cls`, `gloca_sum`, `gloca_gated`.
- Direct CLS is represented by `gloca.enabled: false`, not by a disabled adapter class.
- DINOv2 is the only active backbone family.
- Current supported backbone variant: `facebook/dinov2-small`.
- DINOv2 fine-tuning is not implemented in the current stage.
- PlantSeg class count issues have been resolved in the current implementation state.
- CDC is not implemented yet in the current working tree.

---

## 2. Research Direction

### 2.1 Main research question

The project is not trying to prove that a new clustering algorithm alone is better. It is trying to test whether **GLoCA improves plant disease discovery when placed between frozen DINOv2 and strong clustering methods**.

The central experiment must always be controlled:

```text
same dataset
same seed
same DINOv2 backbone
same clustering method
same output schema
only difference: CLS embedding vs. GLoCA embedding
```

### 2.2 Why replace SPICE with CDC

SPICE was previously planned as the first main deep clustering method, but it is now older and less defensible as the lead future direction.

CDC is a better fit for the next stage because:

- CDC is recent and published at ICLR 2025.
- CDC directly targets overconfidence in deep clustering.
- CDC uses a dual-head design: a calibration head and a clustering head.
- The calibration head estimates reliable confidence for pseudo-label selection.
- Calibrated confidence is useful for this project because plant disease discovery needs not only cluster assignments but also a defensible notion of assignment reliability.
- CDC can naturally produce extra diagnostic outputs such as calibrated confidence, reliable sample ratio, and calibration error.

Therefore, CDC should replace SPICE as the first main deep clustering method.

### 2.3 SPICE status

SPICE is no longer part of the main implementation roadmap.

Allowed future use:

```text
optional legacy reference
optional reproduction baseline
optional ablation if time remains
```

Not allowed in this stage:

```text
main proof of GLoCA
primary future direction
required success criterion
```

---

## 3. Design Philosophy To Preserve

The following rules are non-negotiable and should guide all future implementation.

### 3.1 GLoCA independence

GLoCA must remain independent from clustering methods.

Correct composition:

```text
frozen DINOv2 backbone
→ optional GLoCA
→ final embedding tensor
→ clustering method
```

Incorrect composition:

```text
CDC-specific GLoCA
ProPos-specific GLoCA
CoHiClust-specific GLoCA
method-specific adapter branches
clustering method reads raw patch tokens directly
```

The clustering method should consume only:

```python
embedding: Tensor
```

Method-specific code may receive labels only for evaluation, never for training.

### 3.2 Same-method comparison

Every main method must be evaluated in matched pairs:

```text
DINO CLS + method
DINO + GLoCA + method
```

For example:

```text
DINO CLS + CDC
DINO + GLoCA-gated + CDC
```

Do not compare:

```text
DINO CLS + K-Means
vs.
DINO + GLoCA + CDC
```

as the primary proof, because that confounds the adapter with the clustering method.

### 3.3 Conservative GLoCA

The main GLoCA variant remains `gloca_gated`, a conservative residual correction anchored on the CLS path:

```text
z_cls = W_cls(CLS)
z_patch = gated_attention_pool(patch_tokens)
delta = MLP(concat(z_cls, z_patch))
h = normalize(z_cls + alpha * delta)
```

`alpha` should remain learnable and initialized to `0.0` or a very small value. This preserves DINO geometry at initialization and prevents early destruction of the embedding space.

For the default `gloca_gated` variant, the residual skip should be anchored at `z_cls`. When `input_dim == embedding_dim` and `alpha_init == 0.0`, the untrained adapter should behave as close as possible to normalized DINO CLS. This makes the adapter conservative, debuggable, and suitable for matched CLS-vs-GLoCA comparisons.

A more expressive fused-residual design may still be valuable as a future ablation, but it is not the default safe variant:

```text
z_cls = W_cls(CLS)
z_patch = gated_attention_pool(patch_tokens)
z_fused = Fusion(z_cls, z_patch)
h = normalize(z_fused + ResidualMLP(z_fused))
```

This fused-residual design places the skip connection after global-local fusion rather than at `z_cls`. It may learn stronger global-local interactions, but it no longer guarantees exact CLS preservation at initialization unless carefully initialized. Therefore, it should be treated as a future experimental direction after the conservative `gloca_gated` path is verified.

### 3.3.1 GLoCA local-information hypothesis under ProPos

Under ProPos, GLoCA attention does not directly optimize for disease spots. ProPos optimizes prototype scattering and positive sampling alignment over the final embedding. GLoCA can help only by changing the embedding so that the ProPos prototypes become more aligned with the desired clustering factor.

The intended mechanism is:

```text
DINO CLS provides a stable global representation.
GLoCA attention pools patch-token information.
The gated local correction can shift the embedding toward disease-relevant local evidence.
ProPos then forms prototypes from these embeddings.
```

This may help if local lesion patterns, textures, color changes, or necrosis regions are more relevant than global leaf/object properties. It may hurt or fail to help if the unsupervised objective instead finds species, background, acquisition source, lighting, or leaf shape to be the dominant stable clustering factor.

Important interpretation rule:

```text
Short trainable-GLoCA ProPos runs should not be interpreted as proving that GLoCA is unpromising.
They mainly diagnose whether the adapter is stable, whether alpha opens, whether attention sharpens, and whether the local correction improves or damages the embedding.
```

Useful diagnostics:

```text
gloca_alpha_value
gloca_delta_norm_mean/std
gloca_embedding_cls_cosine_mean/std
gloca_attention_entropy_mean
gloca_attention_max_mean
E-step metrics
final clustering metrics
```

### 3.4 Simple framework over large abstractions

Prefer cohesive, concise modules over a plugin-heavy framework.

Use:

```text
src/models/clustering/<method>.py
src/runners/<method>.py
src/training/<method>_trainer.py             # method-specific explicit trainer when needed
```

Avoid:

```text
large plugin systems
method-specific dataset code
method-specific backbone wrappers
fake trainer abstractions
reintroduced one-off scripts from removed phases
```

### 3.5 Cached-vs-live training honesty

Cached deterministic embeddings are allowed for single-view baselines.

Live-image training is required for methods that depend on stochastic views or online representation learning.

Report timing separately:

```text
backbone_cache_time_s
head_train_time_s
total_time_s
inference_time_s
uses_cached_backbone_features
```

Do not compare cached and live methods using one ambiguous `train_time_s` field.

---

## 4. Implemented Baseline Layer

The baseline layer is already implemented and should be treated as completed infrastructure.

Default datasets:

```text
plantvillage
plantwild
plantseg
```

Default seeds:

```text
42
69
67
```

Default baseline runs:

```text
kmeans_cls
spherical_kmeans_cls
kmeans_gloca_gated_untrained
dec_cls
idec_cls
```

Diagnostic runs available only when explicitly requested:

```text
student_t_cls
student_t_gloca_gated
```

The supported sweep writes:

```text
baseline_summary.csv
baseline_summary_agg.csv
```

`baseline_summary_agg.csv` groups by:

```text
backbone
gloca
head
dataset
```

### 4.1 K-Means status

Implemented runner:

```python
run_kmeans(config)
```

Current behavior:

- deterministic single-view image transform,
- frozen DINOv2 embedding extraction,
- optional untrained GLoCA embedding extraction,
- K-Means fit on cached embeddings,
- optional spherical normalization through `baseline.spherical: true`.

Reported `head` values:

```text
kmeans
spherical_kmeans
```

### 4.2 DEC / IDEC status

Implemented runner:

```python
run_dec_idec(config)
```

True DEC/IDEC are standalone DINO CLS baselines. They do not use GLoCA.

Current default config:

```yaml
baseline:
  input_dim: 384
  hidden_dims: [512, 512, 2048]
  latent_dim: 64
  pretrain_epochs: 20
  refine_epochs: 10
  pretrain_lr: 1.0e-3
  refine_lr: 1.0e-4
  lambda_recon: 0.1
  alpha: 1.0
  target_update_interval: null
```

`target_update_interval: null`, `"none"`, `"fixed"`, or `0` means the target distribution is fixed after K-Means initialization. A positive integer refreshes the target every N refinement epochs.

DEC/IDEC remain important baselines but are not main GLoCA heads.

### 4.3 StudentT status

Implemented runner:

```python
run_student_t(config)
```

StudentT is retained as a diagnostic/smoke head only. It is not part of the default baseline sweep.

StudentT can still be run explicitly through:

```text
scripts/run_baselines.py --only student_t_cls
scripts/run_baselines.py --only student_t_gloca_gated
```

StudentT is not true DEC. It is a centroid-based Student-t soft clustering head over an existing embedding.

---

## 5. Current Data, Backbone, and GLoCA Contracts

### 5.1 Dataset contract

The active dataset abstraction is `FolderImageDataset`.

Expected folder layout:

```text
data/raw/<dataset_name>/
  <class_name>/
    image_001.jpg
    image_002.png
```

Single-view sample:

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

Two-view training sample:

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

`ClusteringDataModule` resolves:

```text
training_views: auto
```

as:

```text
single_view -> 1
contrastive_two_view -> 2
```

Prediction/export dataloaders always use deterministic single-view transforms.

Known dataset registry names:

```text
plantseg
plantvillage
plantwild
```

The removed tensor-image cache path should not be reintroduced as a default speed path. Dataloader throughput should be tuned with `trainer.num_workers`; cached deterministic backbone features remain limited to the baseline runners that explicitly report `backbone_cache_time_s`.

### 5.2 Backbone contract

DINOv2 is fixed as the backbone family:

```yaml
backbone:
  family: dinov2
  variant: facebook/dinov2-small
  freeze: true
  image_size: 224
  output: cls_patch_tokens
```

`DINOv2Backbone`:

- loads through HuggingFace `AutoModel`,
- freezes parameters when `freeze: true`,
- keeps the underlying model in eval mode,
- returns CLS, patch tokens, and patch grid,
- extracts CLS from `last_hidden_state[:, 0, :]`,
- extracts patch tokens from `last_hidden_state[:, 1:, :]`.

At 224x224 with DINOv2-S/14:

```text
CLS:          [B, 384]
Patch tokens: [B, 256, 384]
Patch grid:   (16, 16)
```

### 5.3 GLoCA contract

All active GLoCA code lives in:

```text
src/models/gloca.py
```

Implemented variants:

```text
cls
gloca_sum
gloca_gated
```

Direct CLS is represented by:

```yaml
gloca:
  enabled: false
```

All GLoCA adapters return:

```python
{
    "embedding": Tensor,
    "attention": Tensor | None,
    "patch_grid": tuple[int, int] | None,
}
```

The final `embedding` is the only tensor passed into clustering heads/runners.

---

## 6. Output and Metrics Contract

### 6.1 Output files

Each runner writes one output directory:

```text
outputs/<...>/<experiment>/<head>/<dataset>/seed_<seed>/
  config.yaml
  assignments.json
  metrics.csv
  embeddings.pt
  attention.pt         # only when attention exists
  checkpoint.ckpt      # trainable methods
  logs.json
```

The path prefix depends on the configured `experiment.output_dir`. The stable suffix is:

```text
<experiment.name>/<head.name>/<dataset.name>/seed_<seed>
```

### 6.2 Assignment schema

`assignments.json` is validated by:

```text
src/evaluation/assignment_schema.py
```

Required fields:

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

The following arrays must have equal length:

```text
image_ids
labels
assignments
```

The assignment schema intentionally does not require:

```text
model
method
adapter
embedding_source
```

These are derivable from `backbone`, `gloca`, and `head`.

### 6.3 Canonical per-run metrics schema

Canonical `metrics.csv` fields:

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

DEC/IDEC rows append:

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

CDC rows may append method-specific calibration fields:

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

These fields are appended, not substituted for canonical fields.

### 6.4 Aggregate metrics schema

Canonical aggregate fields:

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

`peak_gpu_mb` is the maximum observed peak memory in the grouped runs.

CDC aggregate rows may additionally include:

```text
calibration_ece_mean
calibration_ece_std
reliable_sample_ratio_mean
reliable_sample_ratio_std
calibrated_confidence_mean
```

---

## 7. Updated Stage-Level Roadmap

The old v5 phases 1–5 are now implemented infrastructure. The future roadmap begins after the current baseline suite.

```text
Phase 1 — Data layer                          DONE
Phase 2 — DINOv2 backbone wrapper             DONE
Phase 3 — GLoCA module                        DONE
Phase 4 — StudentTHead smoke test             DONE / DIAGNOSTIC
Phase 5 — Baseline suite                      DONE
Phase 6 — ProPos                              IMPLEMENTED / CURRENT DIAGNOSTIC METHOD
Phase 7 — CDC                                 NEXT NEW METHOD
Phase 8 — CoHiClust                           FUTURE
Phase 9 — Unified dispatcher / main sweeps     FUTURE
Phase 10 — Result analysis and report tables   FUTURE
```

The immediate work is to finish interpreting ProPos/GLoCA diagnostics and then move to CDC as the next new method.

---

## 8. External Official Repository Policy

The official repositories for CDC, ProPos, and CoHiClust have already been cloned locally. They should be treated as reference implementations, not as the final architecture of this project.

Rules:

- Do not copy an entire external repo into `src/`.
- Do not let external repo assumptions dictate the GLoCA repository layout.
- Port only the method-specific losses, heads, initialization, and training logic needed for the GLoCA comparison.
- Preserve the existing dataset, backbone, GLoCA, evaluation, and output contracts.
- Add source references in `logs.json` where useful, for reproducibility.
- Keep external code adaptation isolated and readable.

Recommended local organization if reference repos are stored inside the project:

```text
external/
  CDC/          # official cloned reference repo, not imported directly by production runners
  ProPos/       # official cloned reference repo
  CoHiClust/    # official cloned reference repo
```

Recommended production code organization:

```text
src/models/clustering/cdc.py
src/training/cdc_trainer.py
src/runners/cdc.py
```

Do not hard-code external paths. If a coding agent needs them, it should discover them from the local repository tree before porting.

---

## 9. Phase 6 — CDC

### 9.1 CDC role

CDC becomes the next new deep clustering method for GLoCA evaluation after the current ProPos diagnostic pass.

Role:

```text
main calibrated deep clustering method
replacement for old SPICE plan
next new method after implemented ProPos
```

Required comparison:

```text
DINO CLS + CDC
vs.
DINO + GLoCA-gated + CDC
```

Optional comparison:

```text
DINO + GLoCA-sum + CDC
```

only if `gloca_gated` works and there is time for ablation.

### 9.2 CDC conceptual contract

CDC should be implemented as a clustering method over the final embedding tensor.

CDC should include:

```text
clustering head
calibration head
confidence calibration objective
reliable high-confidence sample selection
pseudo-label self-training / refinement
initialization strategy from the official method where practical
```

In GLoCA composition:

```text
image / augmented views
→ frozen DINOv2
→ optional GLoCA
→ embedding
→ CDC clustering head + calibration head
```

The CDC implementation must not bypass `ClusteringBaseModel.encode_view()`.

### 9.3 CDC module structure

Add:

```text
src/models/clustering/cdc.py
src/training/cdc_trainer.py
src/runners/cdc.py
```

Possible internal classes/functions:

```text
CDCHead
CalibrationHead
ClusteringHead
cdc_confidence_loss
cdc_pseudo_label_loss
select_reliable_samples
compute_calibration_metrics
initialize_cdc_heads
```

If the official implementation uses a large training framework, translate it into the local contract rather than importing it wholesale.

### 9.4 CDC training path

CDC should use live-image training, not a cached single-view baseline, because its training behavior depends on neural heads, confidence calibration, and iterative pseudo-label selection.

Training path:

```text
stochastic training transform
image or views
frozen DINOv2 forward
optional GLoCA forward
CDC loss
optimizer step for GLoCA + CDC heads only
```

Backbone rule:

```text
DINOv2 remains frozen.
```

Trainable components:

```text
CDC heads
GLoCA parameters when gloca.enabled: true
```

Frozen components:

```text
DINOv2 backbone
```

### 9.5 CDC prediction/export path

Prediction path:

```text
model.eval()
torch.no_grad()
deterministic single-view transform
one image -> one embedding -> one CDC assignment
```

Export:

```text
assignments.json
metrics.csv
embeddings.pt
checkpoint.ckpt
logs.json
attention.pt       # only for GLoCA runs
```

CDC may also export optional confidence tensors:

```text
confidence.pt
calibrated_confidence.pt
```

If saved, these should be documented in `logs.json` and should not become required for non-CDC methods.

### 9.6 CDC metrics

CDC must report all canonical metrics.

Additional CDC diagnostics should be logged either in `metrics.csv` or `logs.json`:

```text
clustering confidence distribution
calibrated confidence distribution
reliable sample ratio
number of pseudo-labeled samples
pseudo-label entropy
calibration threshold / schedule
calibration ECE after Hungarian label alignment
maximum calibration error, if implemented
training stage losses
```

Important: calibration metrics are evaluation diagnostics. They must not use ground-truth labels during training.

### 9.7 CDC acceptance criteria

Minimum acceptance:

```text
DINO CLS + CDC runs end-to-end on PlantWild.
DINO + GLoCA-gated + CDC runs end-to-end on PlantWild.
Both runs write canonical outputs.
Both runs write non-null ARI, NMI, ACC, silhouette, and cluster diagnostics.
CDC confidence values are finite.
CDC calibrated confidence values are finite if implemented.
Assignments use more than one cluster.
DINOv2 remains frozen.
Prediction uses deterministic single-view transforms.
```

Research acceptance:

```text
CDC performs competitively with or better than existing baselines.
GLoCA + CDC does not catastrophically degrade DINO CLS + CDC.
GLoCA + CDC improves at least one of ARI, NMI, ACC, silhouette, or calibration diagnostics on at least one plant dataset.
Attention diagnostics remain finite and interpretable.
```

Do not over-tune CDC on a single seed. First make the matched pair reliable, then run the normal seed sweep.

---

## 10. Phase 6 — ProPos

### 10.1 ProPos role

ProPos is now implemented as the current live-image, two-view trainable clustering method.

Role:

```text
current contrastive / prototype-based deep clustering method
current method for diagnosing GLoCA under a live two-view clustering objective
main implemented method before CDC
```

Required comparison:

```text
DINO CLS + ProPos
vs.
DINO + GLoCA-gated + ProPos
```

### 10.2 ProPos training path

ProPos uses stochastic two-view training.

Training path:

```text
view_1, view_2
→ frozen DINOv2 shared encoder
→ optional GLoCA shared adapter
→ embeddings
→ ProPos PSA + PSL losses
```

Prediction path remains deterministic single-view.

Implemented behavior:

```text
PSA uses official-style -2 cosine alignment over noisy online projections.
PSL uses prototype scattering with online-online negatives and online-target positive diagonal replacement.
E-step repeatedly updates hard p(k|x) pseudo-labels through spherical K-Means.
DINOv2 remains frozen.
GLoCA is trainable by default when enabled.
```

### 10.3 ProPos implementation structure

Implemented files:

```text
src/models/clustering/propos.py
src/training/propos_trainer.py
src/runners/propos.py
scripts/run_propos.py
scripts/run_propos_gloca_diagnostics.py
```

Current implementation intentionally remains single-GPU and omits official large-scale infrastructure:

```text
no DDP
no distributed gather
no queue
no SyncBN
no shuffled BN
no LARS
```

These are acceptable for the current laptop-scale GLoCA comparison as long as they are documented in `logs.json` and applied equally to CLS and GLoCA matched runs.

### 10.4 ProPos / GLoCA diagnostic direction

The current diagnostic result should be interpreted as an optimization/schedule signal, not as proof that GLoCA is unpromising.

Important distinction:

```text
CLS and frozen GLoCA mainly expose the global CLS path.
Trainable GLoCA exposes patch-token information through attention and a gated local correction.
The trainable local path may need more time to learn which tokens are useful.
```

Recommended next ProPos/GLoCA diagnostic runs:

```text
Use one or two longer PlantWild seed-42 runs before broad seed sweeps.
Prefer max_epochs around 25 and warmup around 3 for diagnostics.
Use kmeans_interval=2 if runtime is a concern.
Track alpha, delta norm, CLS cosine, attention entropy/max, E-step metrics, and final clustering metrics.
```

A promising GLoCA/ProPos pattern would be:

```text
alpha gradually moves away from 0
CLS cosine decreases slowly, not abruptly
attention entropy decreases moderately
metrics improve after the attention branch has time to specialize
```

A concerning pattern would be:

```text
CLS cosine drops sharply
attention becomes too sharp too early
cluster metrics stay flat or degrade
```

### 10.5 ProPos acceptance criteria

```text
DINO CLS + ProPos runs end-to-end.
DINO + GLoCA-gated + ProPos runs end-to-end.
Both runs export assignments and metrics.
Two-view training is verified.
Single-view prediction is verified.
DINOv2 remains frozen.
ProPos/GLoCA diagnostics are finite when enabled.
```

---

## 11. Phase 8 — CoHiClust

### 11.1 CoHiClust role

CoHiClust remains a main future method because it offers a different clustering structure from CDC and ProPos.

Role:

```text
main hierarchical/coarse-to-fine clustering method
third main GLoCA evaluation method
```

Required comparison:

```text
DINO CLS + CoHiClust
vs.
DINO + GLoCA-gated + CoHiClust
```

### 11.2 CoHiClust implementation structure

Add:

```text
src/models/clustering/cohiclust.py
src/training/cohiclust_trainer.py
src/runners/cohiclust.py
```

### 11.3 CoHiClust output policy

The canonical assignment remains one fine-level cluster assignment per image.

If CoHiClust produces hierarchy information, save it in method-specific outputs:

```text
hierarchy.json
coarse_assignments.pt
fine_assignments.pt
```

But `assignments.json` must still contain the canonical fine assignment:

```text
assignments
```

### 11.4 CoHiClust acceptance criteria

```text
DINO CLS + CoHiClust runs end-to-end.
DINO + GLoCA-gated + CoHiClust runs end-to-end.
Fine assignments are exported in canonical `assignments.json`.
Hierarchy information is optional but documented if saved.
DINOv2 remains frozen.
```

---

## 12. Phase 9 — Unified Dispatcher and Sweeps

The current supported baseline entry point is:

```text
scripts/run_baselines.py
```

For main deep clustering methods, add a separate dispatcher only after CDC works.

Recommended future entry point:

```text
scripts/run_experiments.py
```

or:

```text
scripts/run_main_methods.py
```

Do not replace `scripts/run_baselines.py`; keep it stable for baseline reproduction.

Dispatcher responsibilities:

```text
resolve config
resolve dataset
resolve method
resolve GLoCA setting
resolve seed
call the correct runner
aggregate outputs
```

Supported methods after implementation:

```text
kmeans
spherical_kmeans
dec
idec
student_t       # explicit diagnostic only
cdc
propos
cohiclust
```

Recommended main sweep grid:

```text
datasets: plantvillage, plantwild, plantseg
seeds: 42, 69, 67
backbone: facebook/dinov2-small
gloca settings: disabled, gloca_gated
methods: cdc, propos, cohiclust
```

Optional ablation grid:

```text
gloca_sum
gloca_fused_residual    # future expressive variant, not default
alpha initialization variants
attention temperature variants
CDC calibration threshold variants
```

Run ablations only after the main matched-pair comparisons are stable.

---

## 13. Phase 10 — Result Analysis and Report Tables

After CDC, ProPos, and CoHiClust are implemented, produce report-ready tables.

Main comparison table:

```text
Dataset | Method | Embedding | ARI | NMI | ACC | Silhouette | Peak GPU | Total Time
```

CDC calibration table:

```text
Dataset | Embedding | ACC | ECE | Reliable Sample Ratio | Calibrated Confidence | Cluster Entropy
```

GLoCA diagnostic table:

```text
Dataset | Method | Attention Entropy | Attention Max | Top-5 Mass | Cluster Entropy
```

Baseline context table:

```text
Dataset | Baseline | ARI | NMI | ACC | Silhouette
```

Keep interpretation disciplined:

- K-Means is the simple baseline ladder.
- DEC/IDEC are generic bottleneck baselines.
- StudentT is diagnostic only.
- CDC/ProPos/CoHiClust are the main GLoCA proof methods.
- Calibration metrics are especially important for CDC but should not replace clustering metrics.

---

## 14. Registry Policy

Keep simple dictionaries first.

Suggested future registry:

```python
CLUSTERING_METHODS = {
    "student_t": StudentTHead,
    "cdc": CDCHead,
    "propos": ProPosHead,
    "cohiclust": CoHiClustHead,
}
```

Do not include true DEC/IDEC here as GLoCA heads. They remain standalone baselines.

GLoCA variations:

```python
GLOCA_VARIATIONS = {
    "gloca_sum": GLoCASumAdapter,
    "gloca_gated": GLoCAGatedAdapter,
}
```

Direct CLS:

```yaml
gloca:
  enabled: false
```

Backwards compatibility may support:

```yaml
gloca:
  name: gloca_gated       # preferred
  variation: gloca_gated  # backward compatible
```

---

## 15. Training Wrapper Policy

Trainable logic should use explicit, method-specific PyTorch trainers in `src/training/` when that makes the phase structure clearer.

Current policy:

```text
Use explicit PyTorch loops for baseline trainers and live-image methods.
Keep method-specific trainers small and inspectable.
Do not build a generic callback/plugin training framework.
Do not reintroduce fake trainer abstractions for cached-feature methods.
```

Implemented examples:

```text
src/training/dec_idec_trainer.py
src/training/propos_trainer.py
```

Future CDC and CoHiClust should follow the same principle: use explicit, method-specific trainers that make E-step, pseudo-label, calibration, hierarchy, and export logic easy to inspect.

---

## 16. Attention Monitoring

Do not enable attention regularization in this stage.

Monitor only:

```text
attention_entropy
attention_max
attention_top5_mass
attention_variance
```

Optional future regularization:

```text
L_entropy = (H(attention) - target_entropy)^2
loss = clustering_loss + lambda * L_entropy
```

Keep this disabled unless attention collapse is observed after the main heads are already running.

---

## 17. What Not To Build Yet

Do not build these in the current stage:

```text
post-hoc XAI metrics
PlantSeg mask loading in core clustering runners
CRAFT
Grad-CAM++
Eigen-CAM
deletion evaluation
broad benchmark suite
extra backbone families
DINOv2 fine-tuning
hierarchy-aware multi-level attention
complex plugin system
attention regularization
early stopping by default before a concrete failure mode is observed
gloca_fused_residual as a default/main variant before conservative GLoCA is validated
true DEC/IDEC as GLoCA heads
fake trainer abstraction for cached-feature DEC/IDEC
SPICE as a required main method
```

SPICE can remain a literature reference or optional legacy baseline, but it should not block CDC, ProPos, or CoHiClust.

---

## 18. Updated Success Criteria For This Stage

The stage succeeds when:

```text
1. The current baseline suite remains reproducible through scripts/run_baselines.py.
2. DINO CLS + ProPos and GLoCA + ProPos remain reproducible through scripts/run_propos.py.
3. ProPos/GLoCA diagnostic outputs remain finite and interpretable.
4. DINO CLS + CDC runs end-to-end.
5. DINO + GLoCA-gated + CDC runs end-to-end.
6. CDC confidence/calibration diagnostics are finite and logged.
7. DINO CLS + CoHiClust and GLoCA + CoHiClust run end-to-end.
8. All methods write the same canonical output schema.
9. Method-specific diagnostics are appended without breaking aggregate summaries.
10. DINOv2 is frozen in all runs.
11. Cached-feature methods report cache time and head-training time separately.
12. Live contrastive/calibrated methods do not use offline single-view token caches as their training path.
13. Final tables compare same-method CLS vs GLoCA pairs.
```

Main final comparisons:

```text
DINO CLS + CDC        vs. GLoCA + CDC
DINO CLS + ProPos     vs. GLoCA + ProPos
DINO CLS + CoHiClust  vs. GLoCA + CoHiClust
```

Baseline context:

```text
DINO CLS + K-Means
DINO CLS + spherical K-Means
DINO CLS + true DEC
DINO CLS + true IDEC
Untrained GLoCA + K-Means diagnostic
StudentT diagnostics only when explicitly requested
```

---

## 19. Immediate Codex Implementation Checklist

Use this checklist for the next coding agent runs.

### Step 1 — Protect current implemented state

```text
Read CURRENT_STATE.md.
Run or inspect scripts/run_baselines.py and scripts/run_propos.py.
Do not reintroduce removed scripts.
Use explicit method-specific trainers for trainable methods.
Do not change canonical metrics or assignment schema unless appending method-specific fields.
```

### Step 2 — Finish ProPos/GLoCA diagnostic interpretation

```text
Use scripts/run_propos_gloca_diagnostics.py for short diagnostic matrices.
Use one or two longer PlantWild seed-42 runs to test whether attention needs more time.
Track alpha, delta norm, CLS cosine, attention entropy/max, E-step metrics, and final clustering metrics.
Do not conclude that local information is useless from short runs alone.
```

Recommended longer diagnostic starting point:

```yaml
propos:
  max_epochs: 25
  warmup_epochs: 3
  kmeans_interval: 2
  gloca_lr_multiplier: 0.1
  gloca_alpha_lr_multiplier: 10.0
  log_gloca_diagnostics: true
```

### Step 3 — Inspect official CDC repo

```text
Locate the locally cloned official CDC repository.
Identify the CDC head, calibration head, loss functions, initialization strategy, and training loop.
Do not copy the full repo into src/.
Write a short porting note in logs or docs.
```

### Step 4 — Add local CDC modules

```text
Create src/models/clustering/cdc.py.
Create src/training/cdc_trainer.py or a small runner-local trainer if needed.
Create src/runners/cdc.py.
Use ClusteringBaseModel.encode_view().
Keep DINO frozen.
Train CDC heads and GLoCA only.
```

### Step 5 — Add CDC config and registry entries

```text
Register cdc as a clustering method.
Add configs for cdc_cls and cdc_gloca_gated.
Use PlantWild first.
Use a small limit_per_class option for smoke testing if needed.
```

### Step 6 — Validate CDC matched pair

```text
Run DINO CLS + CDC.
Run DINO + GLoCA-gated + CDC.
Check outputs.
Check finite metrics.
Check confidence/calibration diagnostics.
Check more than one non-empty cluster.
```

### Step 7 — Expand only after stability

```text
Run stable methods on plantwild, plantvillage, and plantseg.
Use seeds 42, 69, 67 for final sweeps.
Generate aggregate summaries.
Compare CLS vs GLoCA only under the same method.
```

### Step 8 — Move to CoHiClust later

```text
Only after CDC and ProPos are stable, port CoHiClust.
Preserve the same output and comparison contracts.
```

---

## 20. Summary

The project has moved past the old Phase 5 planning state. The current repository already has a stable data/backbone/GLoCA/baseline foundation and an implemented ProPos path.

The updated future direction is:

```text
baseline foundation already complete
→ ProPos implemented as the current live-image diagnostic method
→ finish ProPos/GLoCA attention-schedule diagnostics
→ CDC as the next new calibrated deep clustering method
→ CoHiClust as the later hierarchical method
→ unified sweeps and report tables
```

The philosophy remains unchanged:

```text
GLoCA is not a clustering method.
GLoCA is an embedding adapter.
The proof is same clustering method, same setting, with vs. without GLoCA.
```

The current ProPos diagnostics should be interpreted as optimization evidence about the local patch-attention branch, not as a final judgment on whether GLoCA is useful.
