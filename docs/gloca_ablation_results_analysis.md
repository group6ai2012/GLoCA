# GLoCA Ablation Results Analysis

Generated from local artifacts on 2026-06-28.

This document analyzes the completed GLoCA ablation runs and compares each ablation with the matched original GLoCA architecture on the same dataset, seed, backbone, clustering method, training schedule, and output schema.

The matched comparison target is:

```text
outputs/main_experiments/gloca_gated/cdc/plantseg/seed_42
```

The ablation runs are:

```text
outputs/ablation_studies/gloca_mean_pool/cdc/plantseg/seed_42
outputs/ablation_studies/gloca_no_patches/cdc/plantseg/seed_42
outputs/ablation_studies/gloca_no_cls_residual/cdc/plantseg/seed_42
```

All runs are CDC on PlantSeg seed 42 with frozen `facebook/dinov2-small`, 114 clusters, 11,458 images, 200 epochs, the CDC weak/strong/calibration view setup, and deterministic final prediction. The intended difference is only the GLoCA adapter architecture.

## Executive Summary

The ablation results make the original `gloca_gated` design look well-motivated, but not absolutely dominant on this one run.

`gloca_mean_pool` performs worst. Replacing gated attention pooling with uniform mean pooling drops ARI by 0.071, NMI by 0.035, and ACC by 0.059 relative to the original `gloca_gated` run. It also produces exactly uniform attention, so it removes the adaptive patch selection mechanism that appears important for this CDC PlantSeg result.

`gloca_no_patches` also underperforms the original adapter. It keeps the residual CLS correction path but removes patch information. The drop is smaller than mean pooling but still material: ARI falls by 0.033, NMI by 0.025, and ACC by 0.028. This indicates that patch-token information is contributing useful signal beyond a trainable CLS-only correction.

`gloca_no_cls_residual` is the surprise. It slightly beats the original `gloca_gated` run on final ARI, NMI, ACC, and silhouette. However, it does this with much worse calibration error. ECE rises from 0.047 to 0.147 and MCE rises from 0.097 to 0.199. This variant may improve disease-label clustering on this seed, but it gives up the conservative CLS-anchored behavior that the project uses to keep CLS-vs-GLoCA comparisons controlled.

The main interpretation is:

```text
Patch information matters.
Adaptive gated patch pooling matters.
The CLS residual anchor is conservative and stabilizing, but it may also limit peak CDC PlantSeg seed-42 clustering performance.
```

Because this is one dataset and one seed, `gloca_no_cls_residual` should be treated as a hypothesis-generating result, not a new default.

## Adapter Differences

The ablations are implemented as method-independent GLoCA variants in `src/models/gloca.py`.

| Variant | Architecture change | What it tests |
|---|---|---|
| `gloca_gated` | Original conservative adapter: `normalize(z_cls + alpha * delta)` with gated patch attention | Main reference architecture |
| `gloca_mean_pool` | Replaces gated attention pooling with uniform mean patch pooling | Whether learned patch attention matters |
| `gloca_no_patches` | Removes the patch branch; uses CLS-only correction | Whether patch tokens add useful information |
| `gloca_no_cls_residual` | Removes residual CLS output path; uses `normalize(delta)` | Whether the CLS anchor helps or constrains the adapter |

The original project guardrail favors `gloca_gated` because it preserves a controlled CLS-anchored comparison: when `input_dim == embedding_dim` and `alpha_init == 0.0`, it starts close to normalized DINO CLS. The `gloca_no_cls_residual` ablation intentionally breaks that conservative output form, so it should be interpreted carefully even when its final metrics are strong.

## Final Metrics

Canonical final metrics come from each run's `metrics.csv`.

| Variant | ARI | NMI | ACC | Silhouette | Nonempty clusters | Cluster min / max | Cluster entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gloca_gated` | 0.433 | 0.727 | 0.512 | 0.112 | 89 | 1 / 473 | 0.911 |
| `gloca_mean_pool` | 0.362 | 0.692 | 0.453 | 0.096 | 95 | 1 / 449 | 0.926 |
| `gloca_no_patches` | 0.401 | 0.702 | 0.484 | 0.097 | 100 | 1 / 428 | 0.938 |
| `gloca_no_cls_residual` | 0.442 | 0.731 | 0.523 | 0.166 | 95 | 29 / 295 | 0.943 |

Relative to the original `gloca_gated` run:

| Ablation | Delta ARI | Delta NMI | Delta ACC | Delta silhouette | Delta cluster entropy |
|---|---:|---:|---:|---:|---:|
| `gloca_mean_pool` | -0.071 | -0.035 | -0.059 | -0.016 | +0.015 |
| `gloca_no_patches` | -0.033 | -0.025 | -0.028 | -0.016 | +0.027 |
| `gloca_no_cls_residual` | +0.008 | +0.004 | +0.010 | +0.054 | +0.031 |

The final clustering metrics rank the variants as:

```text
ARI:        no_cls_residual > gloca_gated > no_patches > mean_pool
NMI:        no_cls_residual > gloca_gated > no_patches > mean_pool
ACC:        no_cls_residual > gloca_gated > no_patches > mean_pool
Silhouette: no_cls_residual > gloca_gated > no_patches ~= mean_pool
```

This ordering points to two conclusions. First, the worst result is the ablation that removes learned attention and averages all patches uniformly. Second, patch information helps: the CLS-only ablation is below the original adapter, despite having a trainable correction path.

## Calibration and Reliable Samples

CDC-specific calibration and pseudo-label statistics:

| Variant | ECE | MCE | Reliable ratio | Pseudo labels | Clustering confidence mean | Calibrated confidence mean |
|---|---:|---:|---:|---:|---:|---:|
| `gloca_gated` | 0.047 | 0.097 | 0.464 | 5,316 | 0.876 | 0.559 |
| `gloca_mean_pool` | 0.072 | 0.104 | 0.442 | 5,059 | 0.855 | 0.525 |
| `gloca_no_patches` | 0.046 | 0.093 | 0.443 | 5,076 | 0.863 | 0.529 |
| `gloca_no_cls_residual` | 0.147 | 0.199 | 0.555 | 6,355 | 0.912 | 0.669 |

Relative to `gloca_gated`:

| Ablation | Delta ECE | Delta MCE | Delta reliable ratio | Delta pseudo labels |
|---|---:|---:|---:|---:|
| `gloca_mean_pool` | +0.025 | +0.007 | -0.022 | -257 |
| `gloca_no_patches` | -0.001 | -0.003 | -0.021 | -240 |
| `gloca_no_cls_residual` | +0.101 | +0.102 | +0.091 | +1,039 |

The `gloca_no_cls_residual` result has the best clustering metrics but the worst calibration. It selects 1,039 more pseudo-labels than the original adapter and raises the reliable-sample ratio from 0.464 to 0.555, but the large ECE/MCE increase means the broader confidence is not better calibrated.

`gloca_no_patches` is interesting in the opposite direction: it slightly improves ECE/MCE while worsening ARI/NMI/ACC. That reinforces a recurring project lesson: calibration scalars and disease-label clustering metrics must be interpreted jointly.

## Checkpoint Trajectories

Checkpoint metrics come from `checkpoint_metrics.csv`. They are useful for training behavior; final ranking should still use `metrics.csv`.

| Variant | Best ARI | Final ARI | Best NMI | Final NMI | Best ACC | Final ACC | Final loss | Final reliable ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gloca_gated` | 0.433 at epoch 200 | 0.433 | 0.727 at epoch 200 | 0.727 | 0.515 at epoch 170 | 0.512 | -2.750 | 0.464 |
| `gloca_mean_pool` | 0.365 at epoch 160 | 0.362 | 0.692 at epoch 160 | 0.692 | 0.455 at epoch 190 | 0.453 | -2.603 | 0.442 |
| `gloca_no_patches` | 0.401 at epoch 190 | 0.401 | 0.704 at epoch 180 | 0.702 | 0.485 at epoch 180 | 0.484 | -2.632 | 0.443 |
| `gloca_no_cls_residual` | 0.447 at epoch 180 | 0.442 | 0.733 at epoch 180 | 0.731 | 0.527 at epoch 180 | 0.523 | -3.140 | 0.555 |

The original `gloca_gated` run keeps improving through the final checkpoint for ARI and NMI. The `gloca_no_cls_residual` variant peaks earlier at epoch 180 and gives back a small amount by epoch 200, but it still finishes ahead of the original on clustering metrics.

The mean-pool and no-patches variants peak below the original throughout the trajectory. Their underperformance is not just a bad final checkpoint artifact.

## GLoCA Diagnostics

Diagnostics come from nested `gloca_diagnostics` entries in `logs.json`.

| Variant | Alpha first -> last | CLS cosine first -> last | Delta norm first -> last | Attention entropy first -> last | Attention max first -> last |
|---|---:|---:|---:|---:|---:|
| `gloca_gated` | 0.001 -> 0.243 | 1.000 -> 0.677 | 8.5 -> 146.5 | 5.526 -> 4.270 | 0.006 -> 0.062 |
| `gloca_mean_pool` | 0.001 -> 0.243 | 1.000 -> 0.702 | 8.5 -> 129.4 | 5.545 -> 5.545 | 0.004 -> 0.004 |
| `gloca_no_patches` | -0.001 -> -0.337 | 1.000 -> 0.723 | 10.6 -> 73.5 | n/a | n/a |
| `gloca_no_cls_residual` | 0.000 -> 0.000 | 0.021 -> -0.010 | 8.7 -> 12.2 | 5.526 -> 4.500 | 0.006 -> 0.053 |

The diagnostics explain the architecture behavior:

- `gloca_mean_pool` has uniform attention by design. Its attention entropy stays at `log(256) ~= 5.545`, and attention max remains `1/256 = 0.00390625`. It cannot focus on disease-relevant patches.
- `gloca_no_patches` has no attention output and no patch branch. It still moves away from CLS through a trainable CLS-only correction, but the movement is weaker and less useful than the original patch-aware adapter.
- `gloca_no_cls_residual` has alpha fixed at 0.0 in the diagnostic table because the output path ignores `z_cls + alpha * delta` and directly uses `delta`. This variant is not CLS-anchored in the same way as `gloca_gated`.

The original `gloca_gated` and `gloca_no_cls_residual` both learn non-uniform patch attention. The stronger performance of `gloca_no_cls_residual` on this seed is not because it avoids patch attention; it uses the same gated attention mechanism but removes the conservative residual output anchor.

## Interpretation by Ablation

### `gloca_mean_pool`

Mean pooling is the clearest negative ablation. It keeps the patch branch but removes adaptive patch selection. Compared with `gloca_gated`, it loses 0.071 ARI and 0.059 ACC. Since it performs worse than both `gloca_gated` and `gloca_no_patches`, uniform use of patch tokens appears actively less useful than either learned gated patch pooling or ignoring patches entirely.

Interpretation:

```text
Patch information is not automatically useful.
CDC benefits from patch information when GLoCA can weight patches adaptively.
Uniform patch averaging likely dilutes disease-relevant evidence with background or non-diagnostic plant tissue.
```

### `gloca_no_patches`

The no-patches variant removes the main reason to use GLoCA: patch-token information. It still trains a CLS-only residual correction, and it does better than mean pooling, but it remains below the original adapter.

Interpretation:

```text
The trained correction alone helps less than the original patch-aware adapter.
The PlantSeg seed-42 gain is not explained only by giving CDC extra trainable parameters after CLS.
Patch tokens contribute meaningful disease-aligned signal.
```

The small calibration improvement does not overturn the clustering result. The variant is slightly better calibrated but less label-aligned.

### `gloca_no_cls_residual`

This variant removes the conservative CLS residual output path and directly normalizes the learned delta. It is the best final clustering result in this ablation set, but it is also the least controlled comparison against direct CLS.

Interpretation:

```text
The CLS residual anchor is not required for high PlantSeg seed-42 CDC clustering performance.
Removing the anchor may let CDC use a more disease-aligned patch-informed representation.
But the cost is worse calibration and weaker continuity with the direct CLS baseline.
```

This is a useful research lead, not a default-setting result. The variant should be repeated across seeds and datasets before changing the main architecture.

## Overall Conclusion

The ablation study supports the main design idea behind GLoCA: the useful component is not just "more parameters" and not just "some patch pooling." The useful component is adaptive patch-aware embedding correction.

Evidence:

- `gloca_mean_pool` performs worst, showing that uniform patch averaging is not enough.
- `gloca_no_patches` underperforms the original adapter, showing that patch tokens add useful signal.
- `gloca_gated` remains the strongest conservative architecture because it improves over both attention and patch-removal ablations while preserving a CLS-anchored comparison.
- `gloca_no_cls_residual` slightly outperforms `gloca_gated` on PlantSeg seed 42, but its calibration is much worse and its architecture no longer preserves the same conservative CLS residual semantics.

The best current interpretation is:

```text
For CDC on PlantSeg seed 42, learned gated patch attention is important.
The conservative CLS residual anchor is scientifically useful for controlled comparisons, but it may constrain peak clustering performance.
```

## Limitations and Next Steps

- These ablations are only PlantSeg seed 42. Do not generalize them as final architecture conclusions.
- The comparison is method-specific to CDC. It should not be transferred to ProPos without matched ProPos ablation runs.
- `gloca_no_cls_residual` should be repeated on PlantSeg seeds 67 and 69 and on PlantVillage before promoting it beyond hypothesis status.
- Because `gloca_no_cls_residual` worsens calibration, follow-up analysis should inspect whether its higher reliable-sample ratio represents useful added coverage or overconfident pseudo-label expansion.
- Attention diagnostics should remain diagnostic only. Lower entropy or stronger top-k mass is not proof of disease localization.
