# ProPos + GLoCA Local Artifact Report

This report summarizes locally available ProPos experiment artifacts under `outputs/main_experiments`. It uses existing outputs only: `metrics.csv`, `checkpoint_metrics.csv`, `logs.json`, `config.yaml`, `assignments.json`, and saved attention artifacts as summarized in metrics/log files. No new analysis scripts or reusable tooling were added.

The main reported datasets are PlantVillage and PlantSeg. PlantWild / PlantWild v2 is smoke/debug-only and should not be used as independent main evidence.

## 1. Discovered ProPos Runs

| Dataset | Seed | GLoCA | Run path | ARI | NMI | ACC | Silhouette |
|---|---:|---|---|---:|---:|---:|---:|
| plantseg | 42 | disabled | `outputs/main_experiments/base/propos/plantseg/seed_42` | 0.330 | 0.668 | 0.423 | 0.118 |
| plantseg | 42 | gloca_gated | `outputs/main_experiments/gloca_gated/propos/plantseg/seed_42` | 0.325 | 0.663 | 0.414 | 0.136 |
| plantvillage | 42 | disabled | `outputs/main_experiments/base/propos/plantvillage/seed_42` | 0.578 | 0.819 | 0.617 | 0.221 |
| plantvillage | 42 | gloca_gated | `outputs/main_experiments/gloca_gated/propos/plantvillage/seed_42` | 0.506 | 0.787 | 0.560 | 0.255 |

Both reportable datasets have matched seed-42 pairs. The pairs match dataset, seed, backbone, ProPos schedule, K-Means interval, warmup, loss weights, optimizer settings, evaluation cadence, and output schema. The intended controlled difference is the embedding source:

```text
normalized DINO CLS
vs.
GLoCA-gated embedding
```

## 2. Matched-Pair Deltas

| Dataset | Delta ARI | Delta NMI | Delta ACC | Delta Silhouette | Delta Total Loss | Delta PSA | Delta PSL | Delta Cluster Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plantseg | -0.006 | -0.005 | -0.009 | +0.018 | -0.013 | -0.018 | +0.055 | -0.003 |
| plantvillage | -0.073 | -0.032 | -0.056 | +0.034 | -0.150 | -0.131 | -0.189 | -0.006 |

Higher ARI, NMI, ACC, and silhouette are better. Lower ProPos losses are better. GLoCA improves ProPos objective terms, especially on PlantVillage, but worsens disease-label clustering metrics on both reportable datasets.

Additional final-output diagnostics:

| Dataset | Nonempty Clusters CLS / GLoCA | Cluster Min CLS / GLoCA | Cluster Max CLS / GLoCA | Attention Entropy | Attention Top-5 Mass |
|---|---:|---:|---:|---:|---:|
| plantseg | 114 / 114 | 29 / 7 | 254 / 256 | 4.869 | 0.141 |
| plantvillage | 38 / 38 | 441 / 415 | 2590 / 2687 | 4.494 | 0.202 |

Both GLoCA runs keep all clusters nonempty. Cluster-size entropy is slightly lower with GLoCA in both matched pairs.

## 3. Training and Log Evidence

GLoCA learned and moved away from CLS.

| Dataset | Alpha Movement | Embedding-to-CLS Cosine | Attention Entropy Trend | Attention Max Trend |
|---|---:|---:|---:|---:|
| plantseg | 0.0 to -0.161 | 0.959 to 0.754 | 4.696 to 4.765 | 0.031 to 0.038 |
| plantvillage | 0.0 to -0.098 | 0.233 to 0.508 | 2.471 to 4.664 | 0.277 to 0.041 |

Checkpoint and epoch-history best values:

| Dataset / Run | Best ARI | Best NMI | Best ACC |
|---|---:|---:|---:|
| plantseg CLS | 0.347 at epoch 87 | 0.674 at epoch 87 | 0.431 at epoch 92 |
| plantseg GLoCA | 0.329 at epoch 85 | 0.665 at epoch 88 | 0.414 at epoch 69 |
| plantvillage CLS | 0.578 at epoch 22 | 0.824 at epoch 26 | 0.624 at epoch 22 |
| plantvillage GLoCA | 0.557 at epoch 19 | 0.803 at epoch 26 | 0.580 at epoch 19 |

Loss behavior:

| Dataset / Run | Final Total Loss | Final PSA | Final PSL |
|---|---:|---:|---:|
| plantseg CLS | -1.491 | -1.768 | 2.770 |
| plantseg GLoCA | -1.504 | -1.787 | 2.825 |
| plantvillage CLS | -1.572 | -1.728 | 1.564 |
| plantvillage GLoCA | -1.722 | -1.860 | 1.375 |

No collapse or instability was found in the inspected logs:

- `n_empty_cluster_batches = 0` for all four runs.
- `n_invalid_psl_batches = 0` for all four runs.
- E-step history kept all clusters nonempty throughout.
- Logs showed no NaN or Inf values in the inspected JSON histories.
- DINOv2 remained frozen according to `backbone_requires_grad_false: true`.

The E-step history does not expose CDC-style confidence, calibrated confidence, or reliable-sample statistics. ProPos pseudo-label behavior is represented through E-step K-Means assignment quality and cluster occupancy instead.

## 4. Interpretation by Dataset

### PlantSeg

GLoCA does not improve ProPos disease-label clustering on the matched seed-42 run. ARI, NMI, and ACC are all slightly lower with GLoCA. Silhouette is higher, and total/PSA loss are slightly better, but these objective improvements do not translate into better label-aligned clustering.

GLoCA attention is non-uniform, with final attention entropy 4.869 versus uniform `log(256) ~= 5.545`, but entropy alone does not establish lesion localization.

### PlantVillage

GLoCA more clearly hurts ProPos disease-label clustering on the matched seed-42 run. ARI drops by 0.073, NMI by 0.032, and ACC by 0.056. At the same time, total loss, PSA loss, PSL loss, and silhouette improve.

This is an objective-alignment warning: GLoCA helps optimize ProPos-style objectives, but the resulting representation is less aligned with disease labels in the final clustering output.

## 5. Overall Conclusion

The `outputs/main_experiments` ProPos results do not support the hypothesis that GLoCA improves disease-label clustering under ProPos.

Observed evidence:

- GLoCA learns and changes embeddings under ProPos.
- GLoCA improves or partially improves ProPos objective terms.
- All runs remain stable and non-collapsed.
- Final ARI, NMI, and ACC are worse with GLoCA on both reportable datasets.
- Best checkpoint/history ARI, NMI, and ACC also favor CLS on both datasets.

Interpretation:

- The bottleneck for ProPos + GLoCA remains objective alignment, not absent GLoCA learning.
- Lower ProPos loss and higher silhouette are not sufficient evidence of better disease-label clustering.
- These findings contrast with the CDC artifacts, where GLoCA improves ARI/NMI/ACC on PlantSeg and PlantVillage, although CDC calibration is mixed.

## 6. Caveats

- Only seed 42 is available for these matched ProPos comparisons.
- Canonical `metrics.csv` final export and checkpoint-time metrics differ in places; this report uses `metrics.csv` as the primary final result and checkpoint/history records for training behavior.
- Timing fields should not be overinterpreted. Base ProPos `total_time_s` is much smaller than summed epoch wall time in logs, so runtime comparisons are not reliable.
- Attention entropy below uniform means attention is non-uniform, not that it localizes disease evidence.
- PlantWild / PlantWild v2 is smoke/debug-only and excluded from main reported ProPos evidence.

## 7. Next Recommended Experiment

Run additional matched seeds on PlantSeg and PlantVillage before making a final statistical claim. The current seed-42 evidence is negative for GLoCA under ProPos.
