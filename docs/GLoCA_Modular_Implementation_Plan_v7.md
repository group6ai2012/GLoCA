# GLoCA Modular Implementation Plan — v7 CDC, Attention Evidence, and Preprocessing Closure

This plan replaces `GLoCA_Modular_Implementation_Plan_v6_3_Explicit_Trainers.md` as the active implementation roadmap. It is aligned with the current repository state in `CURRENT_STATE.md` and the updated ProPos/GLoCA report interpretation.

The project direction is now sharper:

```text
Baseline infrastructure: implemented
ProPos: implemented and validated as a live-image diagnostic/baseline method
GLoCA diagnostics: implemented and qualitatively inspected
Preprocessing ablation: explored enough to keep current crop/resize as default
CDC: next main method to implement
CoHiClust: later hierarchy/interpretability method
```

The core comparison philosophy remains unchanged:

```text
DINO CLS + clustering method
vs.
DINO + GLoCA + same clustering method
```

GLoCA remains an embedding adapter, not a clustering method. Clustering methods must consume only the final embedding tensor produced by the shared model path.

---

## 0. Version 7 Update Summary

Version 7 updates the roadmap after four new developments:

1. **GLoCA diagnostics have been modularized.**
   Scalar GLoCA diagnostics now live in `src/diagnostics/gloca.py`, not inside the ProPos trainer. This keeps model code focused on raw adapter computation and trainer code focused on training orchestration.

2. **Attention heatmap visualization has been added and inspected.**
   `scripts/visualize_gloca_attention.py` can visualize saved `attention.pt` overlays. Qualitative inspection of the 200-epoch ProPos/GLoCA PlantWild run showed that GLoCA attention is heterogeneous: sometimes it captures disease evidence cleanly, sometimes it focuses on artifacts/background, and sometimes attention quality does not match cluster quality.

3. **The high-entropy interpretation has been revised.**
   High attention entropy does not, by itself, prove that GLoCA is failing to attend to disease-relevant regions. Attention can remain moderately distributed while still highlighting lesion regions. Entropy is useful as a scalar diagnostic but must be interpreted with heatmaps, attention max/top-k mass, cluster assignment behavior, and image content.

4. **The preprocessing ablation is closed for now.**
   Full-frame preserving `resize_pad` and `resize_stretch` trials did not improve the observed ProPos-CLS trajectory over the current crop/resize pipeline. The current preprocessing remains the practical default. Preprocessing remains a caveat for qualitative attention interpretation, not the next main optimization axis.

The resulting research interpretation is:

```text
GLoCA is learning under ProPos.
GLoCA sometimes attends to disease evidence.
GLoCA attention quality and cluster quality are not equivalent.
The current bottleneck is objective/pseudo-label alignment, not alpha starvation, absent learning, or simply bad attention.
CDC is the next main method because calibrated confidence and reliable sample selection directly test this bottleneck.
```

---

## 1. Current Implemented State To Preserve

Treat `CURRENT_STATE.md` as the source of truth for implemented code. The active repository already includes:

```text
src/
  data/
  diagnostics/gloca.py
  evaluation/
  experiments/
  features/dinov2_backbone.py
  models/base.py
  models/gloca.py
  models/clustering/kmeans.py
  models/clustering/propos.py
  models/clustering/student_t.py
  models/baselines/dec_idec.py
  runners/
  training/propos_trainer.py
  training/dec_idec_trainer.py

scripts/
  run_baselines.py
  run_propos.py
  run_propos_gloca_diagnostics.py
  visualize_gloca_attention.py
```

Supported YAML-driven entry points:

```text
python scripts/run_baselines.py <config_path>
python scripts/run_propos.py <config_path>
python scripts/run_propos_gloca_diagnostics.py <config_path>
python scripts/visualize_gloca_attention.py --run-dir <run_dir> --dataset <name> --n-per-class <n>
```

Do not reintroduce removed one-off experiment scripts or hidden CLI experiment defaults. New experiment behavior should be controlled by YAML configs and explicit runners.

Implemented method status:

```text
K-Means / spherical K-Means: implemented baseline layer
DEC / IDEC: implemented DINO-CLS standalone baselines
StudentT: implemented diagnostic centroid head
ProPos: implemented live-image trainable method and diagnostic baseline
CDC: not implemented yet; next new method
CoHiClust: not implemented yet; later method
```

---

## 2. Research Interpretation After ProPos, Attention, and Preprocessing

### 2.1 ProPos is no longer the next optimization target

ProPos should now be treated as:

```text
implemented
working
validated as a live-image trainable baseline/diagnostic method
useful evidence that GLoCA can learn under a clustering objective
not the next main optimization target
```

The 200-epoch PlantWild ProPos runs show:

```text
DINO CLS + ProPos and GLoCA-gated + ProPos reach comparable peak ARI/ACC.
GLoCA-gated does not catastrophically degrade metrics.
GLoCA alpha moves strongly away from zero.
Embedding-to-CLS cosine falls substantially.
GLoCA improves ProPos objective loss terms.
Disease-label ARI/NMI/ACC do not improve reliably with later loss improvement.
```

Therefore, more ProPos epochs are not the next priority. ProPos should be rerun only for:

```text
paper-grade matched schedules
multi-seed reproducibility
attention visualization examples
small diagnostic checks after infrastructure changes
```

### 2.2 Revised attention interpretation

The qualitative attention-map review found at least four regimes:

| Regime | Observed example | Interpretation |
|---|---|---|
| Disease-localizing attention + cohesive cluster | Apple mosaic virus | Best-case evidence that GLoCA can capture disease cues and the embedding can cluster them together. |
| Disease-localizing attention + fragmented clusters | Apple black rot, blueberry rust | GLoCA can attend to lesions, but the final embedding/clustering objective may still separate images by other factors. |
| Weak or mislocalized attention + cohesive cluster | Carrot cavity spot | Cluster cohesion can arise from global object/source cues even when attention is not lesion-centered. |
| Artifact/background attention + mostly cohesive cluster | Banana anthracnose | Dataset artifacts, borders, text, backgrounds, or acquisition source can dominate attention while clusters remain visually/source coherent. |

The key interpretation rule is:

```text
Attention localization is neither necessary nor sufficient for disease-aligned clustering.
```

Consequences:

- Do not treat high attention entropy as proof that GLoCA is not paying attention to useful regions.
- Do not treat good lesion-localizing heatmaps as proof that disease-label clustering will improve.
- Do not treat fragmented clusters as proof that GLoCA attention failed.
- Interpret attention maps jointly with cluster IDs, labels, image content, alpha movement, delta norm, CLS cosine, attention max/top-k mass, and final metrics.

### 2.3 High entropy caveat

At DINOv2-S/14 224px resolution, the patch grid is `16 x 16`, i.e. 256 patch tokens. A fully uniform distribution has entropy about:

```text
log(256) ≈ 5.545
```

The recorded late GLoCA training attention entropy around `4.8` is still relatively high, but it is not uniform. More importantly, a scalar entropy value does not reveal whether the attended mass overlaps disease evidence. Several heatmaps show disease-relevant attention even with nontrivial entropy.

Use entropy as a coarse concentration statistic only. For research claims, prefer wording such as:

```text
Attention remains moderately distributed, but qualitative overlays show that GLoCA sometimes captures visible disease evidence. The scalar entropy alone is insufficient to determine whether GLoCA attends to points of interest.
```

### 2.4 Preprocessing ablation conclusion

PlantWild images are heterogeneous and often non-square, so preprocessing was a valid concern. However, preliminary ablations showed:

```text
resize_pad_224: worse observed ProPos-CLS trajectory than current crop/resize
resize_stretch_224: not better than current crop/resize
current crop/resize: remains the practical default
```

Interpretation:

```text
The issue is not simply "cropping bad, padding good".
Current cropping likely acts as a useful zoom prior for plant/lesion regions.
Full-frame preservation can reduce effective lesion scale at 224px and may introduce padding/border cues.
```

Policy:

- Keep the existing training and prediction preprocessing as default.
- Do not promote `resize_pad` or `resize_stretch` to default.
- Do not spend more immediate compute on randomized-crop tuning unless a clear failure mode appears.
- Retain preprocessing as a qualitative caveat for attention visualization.
- Higher-resolution or object-aware preprocessing can be revisited later, but it is not the next main research step.

---

## 3. Design Philosophy To Preserve

### 3.1 GLoCA independence

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
method-specific patch-token branches
clustering method reading raw patch tokens directly
```

The clustering method should consume only:

```python
embedding: Tensor
```

Ground-truth labels are evaluation-only and must never be used during training.

### 3.2 Same-method matched pairs

Every main method must be evaluated as matched CLS-vs-GLoCA pairs:

```text
same dataset
same seed
same DINOv2 backbone
same clustering method
same schedule
same preprocessing
same evaluation code
same output schema
only difference: embedding source
```

Main pairs:

```text
DINO CLS + ProPos        vs. DINO + GLoCA-gated + ProPos
DINO CLS + CDC           vs. DINO + GLoCA-gated + CDC
DINO CLS + CoHiClust     vs. DINO + GLoCA-gated + CoHiClust
```

Do not use cross-method comparisons as primary proof, e.g.:

```text
DINO CLS + K-Means vs. DINO + GLoCA + CDC
```

### 3.3 Conservative GLoCA remains default

The default adapter remains `gloca_gated`:

```text
z_cls = W_cls(CLS)
z_patch = W_patch(gated_attention_pool(patch_tokens))
delta = MLP(concat(z_cls, z_patch))
h = normalize(z_cls + alpha * delta)
```

Rules:

- `alpha` remains learnable.
- `alpha_init` remains `0.0` or very small by default.
- With `input_dim == embedding_dim` and `alpha_init == 0.0`, the untrained adapter should remain close to normalized DINO CLS.
- The fused-residual variant remains a future ablation only, not a default.

### 3.4 Diagnostics modularity

Current policy:

```text
Raw GLoCA tensors: src/models/gloca.py
Scalar GLoCA diagnostics: src/diagnostics/gloca.py
Method orchestration: src/training/<method>_trainer.py and src/runners/<method>.py
Shared output writing: src/experiments/outputs.py
```

Do not move scalar aggregation back into `src/training/propos_trainer.py` or bloat `src/models/gloca.py` with report-level metrics.

### 3.5 Simple explicit trainers

Use explicit, method-specific PyTorch trainers for trainable methods:

```text
src/training/propos_trainer.py
src/training/cdc_trainer.py
src/training/cohiclust_trainer.py
```

Avoid:

```text
large plugin systems
generic callback frameworks
method-specific dataset wrappers
method-specific backbone wrappers
fake trainer abstractions for cached-feature baselines
```

---

## 4. Completed Infrastructure

The following infrastructure should be treated as complete unless a bug is found:

```text
Data layer
DINOv2 frozen backbone wrapper
GLoCA adapters
GLoCA scalar diagnostics
Attention visualization utility
K-Means / spherical K-Means baselines
DEC / IDEC baselines
StudentT diagnostic head
ProPos model/trainer/runner
ProPos/GLoCA diagnostic matrix
Assignment schema
Metrics schema
Output writer
Baseline sweep
```

Do not refactor completed infrastructure unless it directly unblocks CDC or fixes a reproducibility bug.

---

## 5. Output and Metrics Contract

All runners must preserve the canonical output directory suffix:

```text
<experiment.name>/<head.name>/<dataset.name>/seed_<seed>/
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
attention.pt       # when GLoCA attention exists
checkpoint.ckpt    # trainable methods
```

Canonical metrics fields remain:

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

Method-specific fields may be appended, but canonical fields must not be renamed or removed.

CDC may append:

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

CDC may save optional tensors:

```text
confidence.pt
calibrated_confidence.pt
pseudo_labels.pt
```

These files are optional CDC artifacts, not required outputs for other methods.

---

## 6. Updated Stage-Level Roadmap

```text
Phase 1 — Data layer                              DONE
Phase 2 — DINOv2 backbone wrapper                 DONE
Phase 3 — GLoCA module                            DONE
Phase 4 — StudentT diagnostic head                 DONE / DIAGNOSTIC
Phase 5 — Baseline suite                          DONE
Phase 6 — ProPos                                  DONE / IMPLEMENTED BASELINE + DIAGNOSTIC
Phase 6.1 — GLoCA diagnostics + attention maps     DONE
Phase 6.2 — Preprocessing ablation                 CLOSED FOR NOW
Phase 7 — CDC                                     NEXT MAIN METHOD
Phase 8 — CoHiClust                               FUTURE HIERARCHICAL METHOD
Phase 9 — Unified main-method sweeps               FUTURE
Phase 10 — Result analysis and report tables       FUTURE
```

Immediate work:

```text
1. Keep current preprocessing default.
2. Freeze ProPos as implemented diagnostic/baseline evidence.
3. Implement CDC matched pair: CLS + CDC and GLoCA-gated + CDC.
4. Run CDC smoke tests, then PlantWild seed-42 matched runs.
5. Only after CDC is stable, decide whether CoHiClust should be ported next.
```

---

## 7. Phase 7 — CDC

### 7.1 CDC role

CDC is the next main method because it directly targets the current bottleneck:

```text
pseudo-label reliability
confidence calibration
objective alignment between unsupervised training signal and disease-label clusters
```

The core hypothesis is:

```text
GLoCA can inject patch-token information.
ProPos can optimize this information, but not reliably align it to disease labels.
CDC tests whether calibrated confidence and reliable sample selection make the training signal more disease-aligned.
```

Required comparison:

```text
DINO CLS + CDC
vs.
DINO + GLoCA-gated + CDC
```

Optional later ablation:

```text
DINO + GLoCA-sum + CDC
```

Only run optional GLoCA variants after the default matched pair works.

### 7.2 CDC implementation files

Add:

```text
src/models/clustering/cdc.py
src/training/cdc_trainer.py
src/runners/cdc.py
scripts/run_cdc.py
```

Add configs:

```text
configs/cdc/cdc_plantwild_cls_smoke.yaml
configs/cdc/cdc_plantwild_gloca_gated_smoke.yaml
configs/cdc/cdc_plantwild_cls_seed42.yaml
configs/cdc/cdc_plantwild_gloca_gated_seed42.yaml
```

After stability, add multi-seed configs or a sweep config.

### 7.3 CDC conceptual contract

CDC should operate over final embeddings from the shared model path:

```text
image / augmented image
→ frozen DINOv2
→ optional GLoCA
→ final embedding
→ CDC clustering head + calibration head
```

CDC should include:

```text
clustering head
calibration head
feature/prototype-based head initialization if practical
confidence calibration loss
reliable sample selection
pseudo-label self-training / refinement
calibration and confidence diagnostics
```

Do not let CDC read patch tokens directly. Patch information must enter CDC only through the GLoCA embedding.

### 7.4 CDC training policy

CDC should use live-image training, not a cached deterministic single-view training path, because its head behavior, calibration, and pseudo-label selection depend on online model predictions.

Frozen:

```text
DINOv2 backbone
```

Trainable:

```text
CDC clustering head
CDC calibration head
GLoCA adapter when enabled
```

Optimizer groups should support:

```text
CDC head parameters
GLoCA body parameters
GLoCA alpha parameter
```

Use conservative learning-rate separation similar to ProPos if needed:

```text
cdc head lr: base lr
GLoCA body lr: lower multiplier by default
GLoCA alpha lr: optionally higher multiplier
```

### 7.5 CDC initialization policy

Port only the necessary initialization logic from the official CDC reference. If prototype-based MLP initialization is too large for the first pass, implement CDC with a simpler initialization first, but document the deviation clearly in `logs.json` and `docs/CURRENT_STATE.md`.

Preferred path:

```text
1. Extract deterministic embeddings from the initial model.
2. Initialize CDC clustering/calibration heads with prototype or K-Means-derived weights where practical.
3. Verify finite confidence values and non-collapsed assignments.
```

Acceptable first-pass fallback:

```text
random head initialization + explicit logs noting no CDC prototype initialization
```

Do not silently omit important CDC details.

### 7.6 CDC confidence and calibration outputs

CDC should log:

```text
clustering_confidence_mean/std/min/max
calibrated_confidence_mean/std/min/max
reliable_sample_ratio
pseudo_label_count
pseudo_label_entropy
classwise/reliable sample counts if available
calibration loss
clustering loss
total loss
```

Evaluation-only diagnostics may include ECE/MCE after Hungarian label alignment. These must not be used for training.

### 7.7 CDC prediction/export path

Prediction should use deterministic single-view transforms:

```text
model.eval()
torch.no_grad()
single image -> final embedding -> CDC calibrated prediction / assignment
```

Export:

```text
assignments.json
metrics.csv
embeddings.pt
logs.json
checkpoint.ckpt
attention.pt                  # GLoCA runs only
confidence.pt                 # optional CDC artifact
calibrated_confidence.pt      # optional CDC artifact
```

### 7.8 CDC acceptance criteria

Minimum engineering acceptance:

```text
DINO CLS + CDC runs end-to-end on a PlantWild smoke config.
DINO + GLoCA-gated + CDC runs end-to-end on a PlantWild smoke config.
Both runs write canonical outputs.
DINOv2 remains frozen.
Assignments use more than one cluster.
Confidence values are finite.
Calibrated confidence values are finite if implemented.
No canonical schema fields are removed or renamed.
```

Research acceptance for first real run:

```text
PlantWild seed-42 CLS + CDC completes.
PlantWild seed-42 GLoCA-gated + CDC completes.
CDC performs competitively with existing baselines.
GLoCA + CDC does not catastrophically degrade CLS + CDC.
GLoCA + CDC improves at least one of ARI, NMI, ACC, silhouette, reliable-sample behavior, or calibration diagnostics.
Attention diagnostics remain finite and interpretable.
```

Do not over-tune CDC on one seed before establishing stable matched-pair behavior.

---

## 8. Phase 6 Status — ProPos After Diagnostics

### 8.1 ProPos role

ProPos is implemented and validated. Its role is now:

```text
implemented live-image baseline
GLoCA diagnostic evidence
not the next method to optimize
```

### 8.2 ProPos findings to preserve

Preserve these conclusions in future docs and reports:

```text
GLoCA-gated + ProPos is stable.
GLoCA learns under ProPos.
Alpha starvation is not the current bottleneck.
High entropy does not prove attention failure.
Disease-localizing attention does not guarantee cluster alignment.
Cohesive clusters can arise without lesion-centered attention.
The current bottleneck is objective/pseudo-label alignment.
```

### 8.3 When to rerun ProPos

Only rerun ProPos for:

```text
matched schedule reproducibility
multi-seed final tables
smoke tests after infrastructure changes
qualitative attention examples
```

Do not keep tuning:

```text
epoch count
attention entropy regularization
preprocessing policies
GLoCA LR schedules
```

unless CDC or another method reveals a specific failure mode that points back to ProPos.

---

## 9. Phase 8 — CoHiClust

### 9.1 CoHiClust role

CoHiClust remains a future method, but it is no longer ahead of CDC.

Role:

```text
hierarchical/coarse-to-fine clustering method
interpretability and structure analysis method
later GLoCA evaluation method
```

Potential value:

```text
shows whether PlantWild organizes by host species, disease type, image source, or visual hierarchy
may explain why disease-local attention can still fragment into multiple clusters
can provide hierarchy outputs beyond flat ARI/NMI/ACC
```

Required comparison when implemented:

```text
DINO CLS + CoHiClust
vs.
DINO + GLoCA-gated + CoHiClust
```

### 9.2 CoHiClust implementation files

Add later:

```text
src/models/clustering/cohiclust.py
src/training/cohiclust_trainer.py
src/runners/cohiclust.py
scripts/run_cohiclust.py
```

### 9.3 CoHiClust output policy

Canonical assignment output remains one fine-level assignment per image:

```text
assignments.json -> assignments
```

Optional hierarchy artifacts:

```text
hierarchy.json
coarse_assignments.pt
fine_assignments.pt
hierarchy_distances.pt
```

Do not make hierarchy artifacts required for non-CoHiClust methods.

---

## 10. Preprocessing Policy

### 10.1 Default policy

Keep the current preprocessing pipeline as default:

Training:

```text
RandomResizedCrop
RandomHorizontalFlip
ColorJitter
RandomGrayscale
ToTensor
ImageNet normalization
```

Prediction/export:

```text
Resize(round(image_size * 256 / 224))
CenterCrop(image_size)
ToTensor
ImageNet normalization
```

### 10.2 Closed ablations

Do not promote these to default:

```text
resize_pad_224
resize_stretch_224
```

Reason:

```text
They did not improve the observed ProPos-CLS trajectory and may reduce effective lesion/object scale at 224px.
```

### 10.3 Future preprocessing ideas

Only revisit if a concrete failure mode requires it:

```text
higher image size, e.g. 336 or 448, if DINOv2 positional handling and memory are clean
conservative random-resized-crop scale floor
multi-crop export ensemble
foreground/object-aware crop
valid-patch masking for padded transforms
```

These are future diagnostics, not immediate roadmap items.

---

## 11. Attention Monitoring Policy

Keep attention monitoring enabled for GLoCA runs when affordable:

```text
attention_entropy
attention_max
attention_top5_mass
attention_variance
gloca_attention_entropy_mean
gloca_attention_max_mean
```

Use attention visualization for qualitative audit:

```text
scripts/visualize_gloca_attention.py
```

Interpretation rules:

```text
High entropy alone is not a failure signal.
Low entropy alone is not a success signal.
Disease-localizing attention is positive evidence but not proof of cluster alignment.
Artifact/background attention is a warning sign but may coexist with cohesive clusters.
Attention should be interpreted alongside cluster IDs and metrics.
```

Do not add attention regularization yet.

Deferred regularization ideas:

```text
entropy target regularization
attention sparsity regularization
foreground-aware attention masking
lesion/PlantSeg-supervised XAI metrics
```

These require stronger evidence before implementation.

---

## 12. Unified Dispatcher and Sweeps

Do not add a unified dispatcher until CDC is stable.

Keep current entry points stable:

```text
scripts/run_baselines.py
scripts/run_propos.py
scripts/run_propos_gloca_diagnostics.py
scripts/visualize_gloca_attention.py
```

After CDC works, optionally add:

```text
scripts/run_cdc.py
scripts/run_main_methods.py
```

The main-method dispatcher, if added, should call existing runners rather than reimplement logic.

Recommended final sweep grid after CDC and CoHiClust are stable:

```text
datasets: plantwild, plantvillage, plantseg
seeds: 42, 69, 67
backbone: facebook/dinov2-small
gloca settings: disabled, gloca_gated
methods: propos, cdc, cohiclust
```

Ablations only after main pairs are stable:

```text
gloca_sum
fused-residual GLoCA
alpha initialization variants
CDC calibration threshold variants
higher image size
preprocessing alternatives
```

---

## 13. Report Tables

After CDC is implemented, report tables should include:

Main method table:

```text
Dataset | Method | Embedding | ARI | NMI | ACC | Silhouette | Total Time | Peak GPU
```

CDC calibration table:

```text
Dataset | Embedding | ACC | ECE | Reliable Sample Ratio | Calibrated Confidence | Cluster Entropy
```

GLoCA diagnostics table:

```text
Dataset | Method | Alpha | CLS Cosine | Delta Norm | Attention Entropy | Attention Max | Top-5 Mass
```

Qualitative attention taxonomy:

```text
Regime | Example class | Cluster behavior | Interpretation
```

Preprocessing note:

```text
Current crop/resize remains default; resize-pad and resize-stretch were not promoted after preliminary ProPos-CLS ablation.
```

Keep report interpretation disciplined:

- K-Means is baseline context.
- DEC/IDEC are generic DINO-CLS baselines.
- StudentT is diagnostic only.
- ProPos is implemented live-image diagnostic/baseline evidence.
- CDC is the next main alignment test.
- CoHiClust is later hierarchy/structure analysis.
- Attention maps support qualitative interpretation but do not replace clustering metrics.

---

## 14. What Not To Build Yet

Do not build in the immediate next stage:

```text
post-hoc XAI metrics
PlantSeg mask loading in core clustering runners
Grad-CAM / Grad-CAM++
Eigen-CAM
CRAFT
deletion/insertion evaluation
attention regularization
entropy loss
foreground-supervised attention loss
early stopping as default
DINOv2 fine-tuning
new backbone families
complex plugin framework
full external repo import
SPICE as a required main method
more preprocessing sweeps without a concrete failure mode
GLoCA fused-residual as default
CDC-specific GLoCA
CoHiClust before CDC
```

---

## 15. Immediate Codex Implementation Checklist

### Step 1 — Protect current state

```text
Read CURRENT_STATE.md.
Run existing tests before large changes if possible.
Do not alter canonical output schemas.
Do not change default preprocessing.
Do not change ProPos behavior while implementing CDC.
```

Minimum tests to keep green:

```bash
pytest tests/test_gloca.py tests/test_propos.py
```

Also run any CDC tests added in this phase.

### Step 2 — Inspect official CDC reference

```text
Locate the local official CDC repo.
Identify clustering head, calibration head, initialization, losses, reliable-sample selection, and training loop.
Write a short implementation note.
Do not copy the full repo into src/.
```

### Step 3 — Add CDC model code

```text
Create src/models/clustering/cdc.py.
Implement CDCHead or separate ClusteringHead + CalibrationHead.
Implement confidence utilities and reliable-sample selection.
Implement finite forward outputs over embeddings.
```

### Step 4 — Add CDC trainer

```text
Create src/training/cdc_trainer.py.
Use ClusteringBaseModel.encode_view().
Keep DINOv2 frozen.
Train CDC heads and optional GLoCA only.
Log losses, confidence stats, reliable sample ratio, and pseudo-label stats.
```

### Step 5 — Add CDC runner and script

```text
Create src/runners/cdc.py.
Create scripts/run_cdc.py.
Use YAML config only.
Write canonical outputs.
Append CDC-specific fields without breaking schemas.
```

### Step 6 — Add configs

```text
configs/cdc/cdc_plantwild_cls_smoke.yaml
configs/cdc/cdc_plantwild_gloca_gated_smoke.yaml
configs/cdc/cdc_plantwild_cls_seed42.yaml
configs/cdc/cdc_plantwild_gloca_gated_seed42.yaml
```

### Step 7 — Validate smoke pair

```text
Run CLS + CDC smoke.
Run GLoCA-gated + CDC smoke.
Check finite metrics.
Check more than one non-empty cluster.
Check confidence values.
Check DINO parameters remain frozen.
Check GLoCA diagnostics for GLoCA run.
```

### Step 8 — Run PlantWild seed-42 matched pair

```text
Run DINO CLS + CDC on PlantWild seed 42.
Run DINO + GLoCA-gated + CDC on PlantWild seed 42.
Use identical schedule and preprocessing.
Compare metrics and calibration diagnostics.
```

### Step 9 — Update docs

```text
Update CURRENT_STATE.md after CDC is implemented.
Add CDC results to report only after real runs complete.
Do not rewrite the ProPos interpretation unless new evidence contradicts it.
```

---

## 16. Success Criteria For v7

v7 succeeds when:

```text
1. Existing baseline and ProPos paths remain reproducible.
2. GLoCA diagnostics and attention visualization remain available.
3. The current preprocessing pipeline remains default.
4. DINO CLS + CDC runs end-to-end.
5. DINO + GLoCA-gated + CDC runs end-to-end.
6. CDC outputs canonical files and finite metrics.
7. CDC confidence/calibration diagnostics are logged.
8. DINOv2 remains frozen.
9. GLoCA remains method-independent.
10. CLS-vs-GLoCA comparisons are same-method matched pairs.
```

Research success is not defined as GLoCA winning every metric. It is defined as producing controlled evidence about whether GLoCA patch-token information becomes more useful when the clustering method has a calibrated reliable-sample mechanism.

---

## 17. Summary

The project has moved beyond the question of whether ProPos runs or whether GLoCA learns under ProPos. It does.

The stronger current finding is:

```text
GLoCA can learn and sometimes attend to disease evidence, but ProPos does not reliably convert that local evidence into disease-label-aligned clusters.
```

The preprocessing branch also does not justify further immediate effort:

```text
Full-frame preserving resize-pad/stretch did not improve the observed ProPos-CLS trajectory.
Current crop/resize remains the default.
```

The next main method should therefore be CDC:

```text
CDC directly tests the pseudo-label reliability and confidence-calibration bottleneck that ProPos exposed.
```

CoHiClust remains valuable, but it should follow CDC as a hierarchy/structure analysis method rather than precede the next alignment test.
