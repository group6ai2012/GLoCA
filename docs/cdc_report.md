# CDC + GLoCA Local Artifact Report

This report summarizes locally available CDC experiment artifacts. It uses existing outputs only: `metrics.csv`, `checkpoint_metrics.csv`, `logs.json`, `config.yaml`, and related saved artifacts where surfaced through those summaries. No new analysis scripts or reusable tooling were added.

## 1. Discovered CDC Runs

Primary reportable datasets are PlantVillage and PlantSeg. PlantWild / PlantWild v2 is retained for smoke tests, debugging, and historical artifact inspection only, because PlantSeg has been identified as a near duplicate of PlantWild v2.

| Dataset | Seed | GLoCA | Run group | ARI | NMI | ACC | Silhouette | ECE / MCE |
|---|---:|---|---|---:|---:|---:|---:|---:|
| plantseg | 42 | disabled | `outputs/main_experiments/base/cdc/plantseg/seed_42` | 0.339 | 0.681 | 0.429 | 0.062 | 0.045 / 0.123 |
| plantseg | 42 | gloca_gated | `outputs/main_experiments/gloca_gated/cdc/plantseg/seed_42` | 0.425 | 0.722 | 0.509 | 0.107 | 0.039 / 0.112 |
| plantvillage | 42 | disabled | `outputs/main_experiments/base/cdc/plantvillage/seed_42` | 0.589 | 0.821 | 0.624 | 0.098 | 0.074 / 0.305 |
| plantvillage | 42 | gloca_gated | `outputs/main_experiments/gloca_gated/cdc/plantvillage/seed_42` | 0.649 | 0.875 | 0.667 | 0.209 | 0.179 / 0.383 |
| plantwild | 42 | disabled | `outputs/cdc_full/cdc_cls_seed42/cdc/plantwild/seed_42` | 0.325 | 0.677 | 0.425 | 0.073 | 0.060 / 0.101 |
| plantwild | 42 | gloca_gated | `outputs/cdc_full/cdc_gloca_gated_bottelneck_seed42/cdc/plantwild/seed_42` | 0.356 | 0.688 | 0.452 | 0.068 | 0.046 / 0.086 |
| plantwild smoke | 42 | disabled | `outputs/cdc_smoke/cdc_cls_smoke/cdc/plantwild/seed_42` | 0.011 | 0.522 | 0.100 | -0.107 | 0.091 / 0.091 |
| plantwild smoke | 42 | gloca_gated | `outputs/cdc_smoke/cdc_gloca_gated_smoke/cdc/plantwild/seed_42` | 0.006 | 0.492 | 0.109 | -0.100 | 0.099 / 0.099 |

Additional local CDC artifacts include PlantVillage GLoCA variants under `outputs/cdc_full/...` and `outputs/main_experiments_old/...`, plus several CLS-only PlantWild smoke/profile runs. These are useful as historical or implementation context, but not as primary matched evidence.

## 2. Matched-Pair Deltas

Clean matched pairs were found for PlantSeg and PlantVillage under `outputs/main_experiments`. These match dataset, seed, backbone, head, CDC schedule/config, and differ by embedding source: direct CLS versus `gloca_gated`.

| Dataset | Delta ARI | Delta NMI | Delta ACC | Delta Silhouette | Delta ECE | Delta MCE | Delta Reliable Ratio | Delta Pseudo Labels | Delta Conf-ACC Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plantseg | +0.086 | +0.041 | +0.079 | +0.045 | -0.006 | -0.011 | +0.076 | +867 | +0.008 |
| plantvillage | +0.060 | +0.054 | +0.043 | +0.111 | +0.105 | +0.078 | +0.167 | +9,076 | +0.110 |

PlantWild has no strict matched full pair in the available artifacts. The closest `cdc_full` comparison favors GLoCA on ARI/NMI/ACC, but differs in schedule: CLS uses 100 epochs and `calibration_k=230`, while GLoCA uses 200 epochs and `calibration_k=200`. PlantWild is therefore excluded from main reported evidence.

## 3. Training and Log Evidence

GLoCA moved substantially in the full GLoCA runs.

| Dataset | Alpha movement | Embedding-to-CLS cosine | Attention entropy trend |
|---|---:|---:|---:|
| plantseg | ~0.0006 to 0.210 | ~1.000 to 0.689 | ~5.53 to 4.30 |
| plantvillage | ~0.030 to 0.206 | ~0.953 to 0.589 | ~3.97 to 3.89 |
| plantwild full | ~-0.0008 to -0.061 | ~0.998 to 0.711 | ~5.52 to 4.81 |

Confidence and pseudo-label selection broadened during training:

| Dataset / Run | Selected pseudo-labels first to final | Reliable ratio first to final | Calibrated confidence first to final |
|---|---:|---:|---:|
| plantseg CLS | 245 to 4,400 | 0.021 to 0.384 | 0.048 to 0.417 |
| plantseg GLoCA | 245 to 5,267 | 0.021 to 0.460 | 0.048 to 0.505 |
| plantvillage CLS | 23,124 to 31,720 | 0.426 to 0.584 | 0.440 to 0.600 |
| plantvillage GLoCA | 23,886 to 40,796 | 0.440 to 0.751 | 0.452 to 0.789 |

Loss curves behaved normally in the inspected logs. Total loss became more negative, calibration loss dropped, and checkpoint metrics generally improved. The logs contain JSON `NaN` placeholders for unevaluated ARI/NMI/ACC epochs, but no observed loss collapse or instability in the reported runs.

Best checkpoint/history values:

| Dataset / Run | Best ARI | Best NMI | Best ACC |
|---|---:|---:|---:|
| plantseg CLS | 0.341 at epoch 140 | 0.683 at epoch 140 | 0.432 at epoch 90 |
| plantseg GLoCA | 0.419 at epoch 140 | 0.719 at epoch 140 | 0.504 at epoch 140 |
| plantvillage CLS | 0.604 at epoch 15 | 0.824 at epoch 45 | 0.624 at epoch 50 |
| plantvillage GLoCA | 0.650 at epoch 45 | 0.875 at epoch 50 | 0.668 at epoch 45 |

## 4. Interpretation by Dataset

### PlantSeg

GLoCA helped CDC clustering and calibration in the matched seed-42 run. ARI, NMI, ACC, and silhouette all improved. ECE and MCE decreased, so calibration improved by the available scalar metrics. Reliable sample ratio and pseudo-label count increased. Cluster-size entropy improved, although nonempty clusters decreased slightly from 93 to 91.

Attention entropy was below uniform `log(256) ~= 5.545`, so attention was non-uniform. This does not establish lesion localization without heatmaps.

### PlantVillage

GLoCA helped CDC clustering, pseudo-label breadth, and cluster occupancy in the matched seed-42 run. ARI, NMI, ACC, silhouette, nonempty cluster count, and cluster-size entropy all improved.

Calibration worsened. ECE increased from 0.074 to 0.179, MCE increased from 0.305 to 0.383, and the calibrated-confidence-minus-ACC gap widened from +0.066 to +0.176. The GLoCA run is more overconfident, so it should not be described as a calibration improvement.

### PlantWild

PlantWild results are smoke/debug or historical context only. Available full-looking comparisons are schedule-mismatched, and PlantWild / PlantWild v2 should not be treated as independent main evidence because of the near-duplicate relationship with PlantSeg.

## 5. Overall Conclusion

The reportable PlantSeg and PlantVillage seed-42 results support a limited conclusion: `CDC + gloca_gated` can improve disease-label clustering over `CDC + CLS` when the clustering method is held fixed.

They do not fully support the stronger hypothesis that CDC's calibrated pseudo-label selection consistently makes GLoCA's patch-token information more disease-aligned. PlantSeg supports that interpretation more strongly because clustering and calibration both improve. PlantVillage complicates it: clustering improves, but calibration gets worse and confidence becomes more overconfident.

Observed evidence:

- GLoCA learned and moved away from CLS in the CDC runs.
- CDC selected more reliable pseudo-labels with GLoCA on PlantSeg and PlantVillage.
- Disease-label clustering metrics improved on both reportable matched pairs.
- Calibration improved on PlantSeg but worsened on PlantVillage.

Interpretation:

- The current evidence is favorable for `CDC + GLoCA` as a clustering combination on these two datasets.
- The evidence is not yet enough to claim that calibrated confidence is the causal reason GLoCA becomes disease-aligned.

## 6. Caveats

- Only seed 42 is available for the clean matched CDC comparisons.
- PlantWild is smoke/debug-only and should not be reported as independent main evidence.
- PlantWild full comparisons are schedule-mismatched.
- Some artifacts are smoke/profile/debug runs with random initialization, one epoch, or zero reliable samples.
- Attention entropy below uniform indicates non-uniform attention, not lesion localization.
- PlantVillage GLoCA worsens calibration despite improving clustering.
- PlantSeg GLoCA timing fields are implausible: final metrics report `total_time_s=46.9` and `head_train_time_s=1.0`, inconsistent with checkpoint epoch wall times around 160 seconds. Do not compare that timing field.
- The default shell Python environment did not have `torch`, so `.pt` tensors were not directly loaded during this inspection. The analysis relies on CSV/JSON/config summaries that already expose the requested metrics.

## 7. Next Recommended Experiment

Run the same matched CDC CLS-vs-GLoCA comparison on PlantVillage and PlantSeg for additional seeds, keeping dataset, seed, DINOv2 backbone, CDC schedule, initialization, optimizer settings, evaluation, and output schema controlled.
