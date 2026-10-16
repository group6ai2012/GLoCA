# Main Experiment Results Analysis

> This historical analysis is superseded by the complete current report at [`docs/full_result_report.md`](full_result_report.md), which includes the full three-seed baselines, five-seed CDC runs, one-seed ProPos runs, three-seed GLoCA ablations, and updated CDC occupancy/calibration analysis for the main and ablation comparisons.

Generated from local artifacts on 2026-06-27.

This report analyzes the completed main experiment artifacts for the reportable datasets in the current project state:

```text
plantvillage
plantseg
```

PlantWild / PlantWild v2 is excluded from primary evidence because `docs/CURRENT_STATE.md` identifies it as smoke/debug/historical context and notes that PlantSeg is a near duplicate of PlantWild v2. The report therefore treats PlantWild as non-independent evidence and does not use it for the main conclusion.

The scientific comparison remains:

```text
DINO CLS + same clustering method
vs.
DINO + GLoCA + same clustering method
```

GLoCA is interpreted only as an embedding adapter. It is not treated as a clustering method.

## Executive Summary

The strongest current evidence is for CDC. Across three matched seeds on PlantSeg, `CDC + gloca_gated` improves ARI, NMI, ACC, and silhouette over `CDC + CLS` on every seed. On PlantVillage, `CDC + gloca_gated` improves NMI, ACC, and silhouette on every seed; ARI improves clearly on seed 42, is slightly negative on seed 67, and is slightly positive on seed 69. Averaged over the three seeds, CDC improves on both reportable datasets with GLoCA.

The baseline artifacts show that untrained `gloca_gated` stays extremely close to the CLS spherical K-Means baseline. This supports the conservative-initialization interpretation: the adapter does not create a large uncontrolled embedding shift before training.

The ProPos artifacts in `outputs/main_experiments` are only available for seed 42. They are useful as limited matched evidence, but not as a three-seed conclusion. On seed 42, ProPos with GLoCA worsens ARI, NMI, and ACC on both PlantSeg and PlantVillage while improving silhouette. This is an objective-alignment warning: a representation can look better by internal clustering geometry while becoming less aligned with disease labels.

Overall, the three-seed CDC results support a cautious positive claim for `CDC + GLoCA` under controlled CLS-vs-GLoCA comparisons. The current ProPos result does not support a positive GLoCA claim for ProPos.

## Experiment Contract and Artifact Coverage

The report uses existing artifacts only:

| Source | Use |
|---|---|
| `docs/CURRENT_STATE.md` | Current project scope, interpretation guardrails, and PlantWild exclusion |
| `outputs/baselines_full/baseline_summary.csv` | Per-seed baseline metrics |
| `outputs/baselines_full/baseline_summary_agg.csv` | Three-seed baseline aggregates |
| `outputs/main_experiments/**/metrics.csv` | Canonical final metrics for CDC and ProPos |
| `outputs/main_experiments/**/checkpoint_metrics.csv` | Checkpoint-time metric trajectories |
| `outputs/main_experiments/**/logs.json` | Training diagnostics, GLoCA scalar diagnostics, timing, and stability signals |
| `docs/cdc_report.md`, `docs/propos_report.md` | Historical seed-42 context only |

Matched artifact coverage under `outputs/main_experiments`:

| Method | Dataset | CLS seeds | GLoCA seeds | Main interpretation status |
|---|---|---:|---:|---|
| CDC | plantseg | 42, 67, 69 | 42, 67, 69 | Three-seed matched evidence |
| CDC | plantvillage | 42, 67, 69 | 42, 67, 69 | Three-seed matched evidence |
| ProPos | plantseg | 42 | 42 | Limited seed-42 evidence |
| ProPos | plantvillage | 42 | 42 | Limited seed-42 evidence |

The CDC runs are therefore the only trainable main-method artifacts that currently support a three-seed matched claim. ProPos should not be described as a three-seed result unless seed 67 and seed 69 artifacts are added.

## Baseline Three-Seed Results

Baseline results come from `outputs/baselines_full/baseline_summary_agg.csv`. Values are mean +/- standard deviation over seeds 42, 67, and 69.

| Dataset | Head | GLoCA | ARI | NMI | ACC | Silhouette |
|---|---|---|---:|---:|---:|---:|
| plantseg | DEC | disabled | 0.263 +/- 0.007 | 0.633 +/- 0.003 | 0.366 +/- 0.005 | 0.167 +/- 0.003 |
| plantseg | IDEC | disabled | 0.263 +/- 0.007 | 0.633 +/- 0.002 | 0.366 +/- 0.005 | 0.167 +/- 0.004 |
| plantseg | K-Means | disabled | 0.285 +/- 0.005 | 0.651 +/- 0.001 | 0.385 +/- 0.003 | 0.059 +/- 0.004 |
| plantseg | spherical K-Means | disabled | 0.292 +/- 0.005 | 0.653 +/- 0.001 | 0.394 +/- 0.005 | 0.063 +/- 0.003 |
| plantseg | spherical K-Means | gloca_gated, untrained | 0.293 +/- 0.005 | 0.653 +/- 0.002 | 0.394 +/- 0.004 | 0.063 +/- 0.003 |
| plantvillage | DEC | disabled | 0.532 +/- 0.038 | 0.786 +/- 0.017 | 0.568 +/- 0.032 | 0.094 +/- 0.006 |
| plantvillage | IDEC | disabled | 0.531 +/- 0.038 | 0.786 +/- 0.017 | 0.567 +/- 0.032 | 0.093 +/- 0.006 |
| plantvillage | K-Means | disabled | 0.541 +/- 0.028 | 0.786 +/- 0.002 | 0.573 +/- 0.017 | 0.097 +/- 0.006 |
| plantvillage | spherical K-Means | disabled | 0.516 +/- 0.017 | 0.784 +/- 0.000 | 0.563 +/- 0.008 | 0.094 +/- 0.006 |
| plantvillage | spherical K-Means | gloca_gated, untrained | 0.516 +/- 0.018 | 0.784 +/- 0.000 | 0.563 +/- 0.008 | 0.094 +/- 0.007 |

The untrained GLoCA spherical K-Means baseline is nearly identical to direct CLS spherical K-Means. Per-seed deltas are tiny:

| Dataset | Seed | Delta ARI | Delta NMI | Delta ACC | Delta silhouette |
|---|---:|---:|---:|---:|---:|
| plantseg | 42 | +0.000515 | -0.000086 | -0.000175 | -0.000107 |
| plantseg | 67 | +0.000592 | +0.000851 | +0.000698 | +0.000793 |
| plantseg | 69 | +0.000917 | +0.000752 | +0.001571 | -0.000125 |
| plantvillage | 42 | -0.000044 | +0.000076 | +0.000074 | -0.000011 |
| plantvillage | 67 | -0.000063 | -0.000140 | -0.000147 | +0.000037 |
| plantvillage | 69 | -0.000715 | -0.000481 | -0.000147 | -0.000549 |

This matters because it supports a controlled interpretation of trainable GLoCA results. The adapter starts close enough to the CLS path that later changes are training effects, not a large arbitrary initialization shift.

## CDC Matched Three-Seed Analysis

The CDC comparison is the strongest current result because both reportable datasets have matched CLS and GLoCA runs on seeds 42, 67, and 69.

Canonical final metrics from `metrics.csv`:

| Dataset | Seed | Embedding | ARI | NMI | ACC | Silhouette |
|---|---:|---|---:|---:|---:|---:|
| plantseg | 42 | CLS | 0.338 | 0.684 | 0.430 | 0.067 |
| plantseg | 42 | GLoCA | 0.433 | 0.727 | 0.512 | 0.112 |
| plantseg | 67 | CLS | 0.356 | 0.685 | 0.447 | 0.064 |
| plantseg | 67 | GLoCA | 0.377 | 0.691 | 0.466 | 0.105 |
| plantseg | 69 | CLS | 0.338 | 0.689 | 0.441 | 0.069 |
| plantseg | 69 | GLoCA | 0.399 | 0.719 | 0.487 | 0.108 |
| plantvillage | 42 | CLS | 0.589 | 0.821 | 0.624 | 0.098 |
| plantvillage | 42 | GLoCA | 0.649 | 0.875 | 0.667 | 0.209 |
| plantvillage | 67 | CLS | 0.591 | 0.826 | 0.618 | 0.098 |
| plantvillage | 67 | GLoCA | 0.588 | 0.860 | 0.647 | 0.186 |
| plantvillage | 69 | CLS | 0.618 | 0.837 | 0.639 | 0.100 |
| plantvillage | 69 | GLoCA | 0.621 | 0.863 | 0.669 | 0.199 |

Per-seed GLoCA-minus-CLS deltas:

| Dataset | Seed | Delta ARI | Delta NMI | Delta ACC | Delta silhouette |
|---|---:|---:|---:|---:|---:|
| plantseg | 42 | +0.094867 | +0.042768 | +0.082126 | +0.045591 |
| plantseg | 67 | +0.020909 | +0.006143 | +0.019026 | +0.041200 |
| plantseg | 69 | +0.061535 | +0.029697 | +0.046780 | +0.038715 |
| plantvillage | 42 | +0.059881 | +0.053933 | +0.043053 | +0.110668 |
| plantvillage | 67 | -0.003199 | +0.034479 | +0.028800 | +0.088557 |
| plantvillage | 69 | +0.003771 | +0.026216 | +0.030531 | +0.098513 |

Aggregate CDC means. Values are mean +/- sample standard deviation across seeds 42, 67, and 69:

| Dataset | Embedding | ARI | NMI | ACC | Silhouette |
|---|---|---:|---:|---:|---:|
| plantseg | CLS | 0.344 +/- 0.010 | 0.686 +/- 0.003 | 0.439 +/- 0.009 | 0.066 +/- 0.003 |
| plantseg | GLoCA | 0.403 +/- 0.028 | 0.712 +/- 0.019 | 0.489 +/- 0.023 | 0.108 +/- 0.004 |
| plantvillage | CLS | 0.599 +/- 0.016 | 0.828 +/- 0.008 | 0.627 +/- 0.010 | 0.099 +/- 0.001 |
| plantvillage | GLoCA | 0.619 +/- 0.031 | 0.866 +/- 0.008 | 0.661 +/- 0.012 | 0.198 +/- 0.011 |

Aggregate GLoCA-minus-CLS deltas. Values are mean +/- sample standard deviation across matched per-seed deltas:

| Dataset | Delta ARI | Delta NMI | Delta ACC | Delta silhouette |
|---|---:|---:|---:|---:|
| plantseg | +0.059 +/- 0.037 | +0.026 +/- 0.019 | +0.049 +/- 0.032 | +0.042 +/- 0.003 |
| plantvillage | +0.020 +/- 0.035 | +0.038 +/- 0.014 | +0.034 +/- 0.008 | +0.099 +/- 0.011 |

### CDC Interpretation

PlantSeg is the cleanest CDC result. GLoCA improves all four reported clustering metrics on every seed. The gains are not one-seed artifacts: ARI improves by about +0.021 to +0.095, ACC by about +0.019 to +0.082, and silhouette by about +0.039 to +0.046.

PlantVillage is more nuanced. NMI, ACC, and silhouette improve on every seed. ARI is mixed: seed 42 improves substantially, seed 67 decreases slightly by -0.003, and seed 69 increases slightly by +0.004. The right interpretation is not "GLoCA always improves every metric on PlantVillage"; it is "GLoCA consistently improves NMI, ACC, and geometry, while ARI is nearly flat outside seed 42."

The average PlantVillage ARI gain remains positive, but it is smaller than the PlantSeg gain. The average PlantVillage silhouette gain is large, increasing from 0.099 to 0.198, so CDC + GLoCA produces a much more separated embedding/clustering geometry even when ARI is not uniformly higher.

## CDC Occupancy and Calibration

CDC final cluster and calibration summaries:

| Dataset | Seed | Embedding | Nonempty clusters | Min / max cluster | Cluster entropy | ECE | MCE | Reliable ratio | Pseudo labels |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| plantseg | 42 | CLS | 93 | 1 / 582 | 0.903 | 0.048 | 0.153 | 0.381 | 4,371 |
| plantseg | 42 | GLoCA | 89 | 1 / 473 | 0.911 | 0.047 | 0.097 | 0.464 | 5,316 |
| plantseg | 67 | CLS | 99 | 6 / 529 | 0.914 | 0.040 | 0.100 | 0.384 | 4,397 |
| plantseg | 67 | GLoCA | 97 | 1 / 401 | 0.931 | 0.080 | 0.144 | 0.456 | 5,220 |
| plantseg | 69 | CLS | 97 | 1 / 480 | 0.918 | 0.034 | 0.064 | 0.385 | 4,408 |
| plantseg | 69 | GLoCA | 94 | 2 / 546 | 0.919 | 0.074 | 0.118 | 0.464 | 5,317 |
| plantvillage | 42 | CLS | 35 | 1 / 3348 | 0.953 | 0.074 | 0.305 | 0.584 | 31,720 |
| plantvillage | 42 | GLoCA | 37 | 510 / 3356 | 0.974 | 0.179 | 0.383 | 0.751 | 40,796 |
| plantvillage | 67 | CLS | 37 | 426 / 3339 | 0.973 | 0.084 | 0.175 | 0.600 | 32,603 |
| plantvillage | 67 | GLoCA | 38 | 471 / 2709 | 0.984 | 0.196 | 0.289 | 0.747 | 40,569 |
| plantvillage | 69 | CLS | 37 | 1 / 3343 | 0.963 | 0.077 | 0.263 | 0.604 | 32,789 |
| plantvillage | 69 | GLoCA | 37 | 825 / 3350 | 0.980 | 0.187 | 0.307 | 0.772 | 41,931 |

The reliable-sample ratio and pseudo-label count increase with GLoCA on every CDC seed for both datasets. That is evidence that CDC selects more samples as reliable under GLoCA.

Calibration should be interpreted more carefully. On PlantSeg seed 42, GLoCA slightly improves ECE and clearly improves MCE. On PlantSeg seeds 67 and 69, ECE and MCE worsen even though ARI/NMI/ACC improve. On PlantVillage, ECE and MCE worsen on all three seeds with GLoCA. Therefore, the three-seed evidence does not support the claim that GLoCA improves CDC calibration. It supports a narrower claim: GLoCA improves disease-label clustering under CDC while also increasing reliable-sample coverage, but calibration error is mixed or worse.

## ProPos Seed-42 Matched Analysis

ProPos has matched artifacts only for seed 42. These runs should be treated as limited evidence, not a three-seed conclusion. Because there is only one matched seed per dataset, no across-seed standard deviation is reported for ProPos.

Canonical final metrics from `metrics.csv`:

| Dataset | Seed | Embedding | ARI | NMI | ACC | Silhouette | Nonempty clusters | Cluster entropy |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| plantseg | 42 | CLS | 0.330 | 0.668 | 0.423 | 0.118 | 114 | 0.984 |
| plantseg | 42 | GLoCA | 0.325 | 0.663 | 0.414 | 0.136 | 114 | 0.981 |
| plantvillage | 42 | CLS | 0.578 | 0.819 | 0.617 | 0.221 | 38 | 0.983 |
| plantvillage | 42 | GLoCA | 0.506 | 0.787 | 0.560 | 0.255 | 38 | 0.976 |

GLoCA-minus-CLS deltas:

| Dataset | Delta ARI | Delta NMI | Delta ACC | Delta silhouette |
|---|---:|---:|---:|---:|
| plantseg | -0.005502 | -0.005461 | -0.009338 | +0.018015 |
| plantvillage | -0.072500 | -0.031776 | -0.056127 | +0.034007 |

The ProPos seed-42 result is negative for disease-label clustering with GLoCA. It lowers ARI, NMI, and ACC on both reportable datasets. At the same time, silhouette improves, which reinforces the project guardrail that better internal clustering geometry is not the same thing as better disease-label discovery.

This result also contrasts with CDC. The same embedding adapter can be helpful under one clustering objective and harmful under another. That makes method-specific objective alignment a more plausible explanation than a generic "GLoCA is always better" or "GLoCA is always worse" interpretation.

## Checkpoint Peak-vs-Final Behavior

Checkpoint histories come from `checkpoint_metrics.csv`. These are useful for training behavior, but `metrics.csv` remains the canonical final export. This distinction matters most for ProPos, where checkpoint-time final values and exported final metrics differ in places.

| Method | Dataset | Seed | Embedding | Best ARI | Final checkpoint ARI | Best NMI | Final checkpoint NMI | Best ACC | Final checkpoint ACC |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| CDC | plantseg | 42 | CLS | 0.342 | 0.338 | 0.684 | 0.684 | 0.432 | 0.430 |
| CDC | plantseg | 42 | GLoCA | 0.433 | 0.433 | 0.727 | 0.727 | 0.515 | 0.512 |
| CDC | plantseg | 67 | CLS | 0.359 | 0.356 | 0.686 | 0.685 | 0.449 | 0.447 |
| CDC | plantseg | 67 | GLoCA | 0.377 | 0.377 | 0.691 | 0.691 | 0.466 | 0.466 |
| CDC | plantseg | 69 | CLS | 0.339 | 0.338 | 0.689 | 0.689 | 0.442 | 0.441 |
| CDC | plantseg | 69 | GLoCA | 0.399 | 0.399 | 0.719 | 0.719 | 0.491 | 0.487 |
| CDC | plantvillage | 42 | CLS | 0.604 | 0.589 | 0.824 | 0.821 | 0.624 | 0.624 |
| CDC | plantvillage | 42 | GLoCA | 0.650 | 0.649 | 0.875 | 0.875 | 0.668 | 0.667 |
| CDC | plantvillage | 67 | CLS | 0.591 | 0.591 | 0.826 | 0.826 | 0.618 | 0.618 |
| CDC | plantvillage | 67 | GLoCA | 0.624 | 0.588 | 0.860 | 0.860 | 0.661 | 0.647 |
| CDC | plantvillage | 69 | CLS | 0.618 | 0.618 | 0.837 | 0.837 | 0.642 | 0.639 |
| CDC | plantvillage | 69 | GLoCA | 0.641 | 0.621 | 0.863 | 0.863 | 0.678 | 0.669 |
| ProPos | plantseg | 42 | CLS | 0.338 | 0.338 | 0.670 | 0.670 | 0.418 | 0.418 |
| ProPos | plantseg | 42 | GLoCA | 0.329 | 0.312 | 0.659 | 0.657 | 0.414 | 0.398 |
| ProPos | plantvillage | 42 | CLS | 0.566 | 0.533 | 0.824 | 0.805 | 0.622 | 0.560 |
| ProPos | plantvillage | 42 | GLoCA | 0.552 | 0.511 | 0.803 | 0.790 | 0.579 | 0.568 |

CDC is generally stable by the end of training. On PlantSeg, the best and final checkpoint values are very close, especially for GLoCA. On PlantVillage, GLoCA peaks earlier for ARI and ACC on seeds 67 and 69, but final NMI remains high and final ACC remains above CLS.

ProPos checkpoint histories strengthen the limited seed-42 interpretation: best checkpoint ARI/NMI/ACC still favor CLS over GLoCA on both datasets.

## GLoCA Diagnostics and Attention

Final attention scalars from `metrics.csv`:

| Method | Dataset | Seed | Attention entropy | Attention max | Attention top-5 mass |
|---|---|---:|---:|---:|---:|
| CDC | plantseg | 42 | 4.365 | 0.058 | 0.212 |
| CDC | plantseg | 67 | 4.481 | 0.061 | 0.216 |
| CDC | plantseg | 69 | 4.582 | 0.050 | 0.184 |
| CDC | plantvillage | 42 | 4.074 | 0.070 | 0.256 |
| CDC | plantvillage | 67 | 3.992 | 0.077 | 0.274 |
| CDC | plantvillage | 69 | 4.269 | 0.061 | 0.224 |
| ProPos | plantseg | 42 | 4.869 | 0.036 | 0.141 |
| ProPos | plantvillage | 42 | 4.494 | 0.058 | 0.202 |

For a 16 x 16 patch grid, uniform attention entropy is approximately `log(256) = 5.545`. These values are below uniform, so attention is non-uniform. However, this report does not interpret attention entropy as proof of disease localization. Attention maps are diagnostics, not final evidence of clustering correctness.

GLoCA scalar diagnostics from `logs.json`:

| Method | Dataset | Seed | Alpha first -> last | CLS cosine first -> last | Delta norm first -> last | Attention entropy first -> last |
|---|---|---:|---:|---:|---:|---:|
| CDC | plantseg | 42 | 0.001 -> 0.243 | 1.000 -> 0.677 | 8.5 -> 146.5 | 5.526 -> 4.270 |
| CDC | plantseg | 67 | 0.000 -> -0.230 | 1.000 -> 0.691 | 8.1 -> 140.6 | 5.524 -> 4.260 |
| CDC | plantseg | 69 | 0.000 -> 0.249 | 1.000 -> 0.706 | 8.4 -> 123.7 | 5.529 -> 4.417 |
| CDC | plantvillage | 42 | 0.030 -> 0.206 | 0.953 -> 0.589 | 192.9 -> 147.7 | 3.971 -> 3.890 |
| CDC | plantvillage | 67 | 0.030 -> 0.238 | 0.946 -> 0.609 | 244.4 -> 140.0 | 3.675 -> 3.709 |
| CDC | plantvillage | 69 | 0.031 -> 0.226 | 0.955 -> 0.611 | 194.3 -> 133.7 | 3.942 -> 3.940 |
| ProPos | plantseg | 42 | -0.020 -> -0.161 | 0.959 -> 0.754 | 127.5 -> 222.6 | 4.696 -> 4.765 |
| ProPos | plantvillage | 42 | -0.056 -> -0.098 | 0.233 -> 0.508 | 3091.9 -> 387.8 | 2.471 -> 4.664 |

The diagnostic pattern is clear: GLoCA learns and moves under both CDC and ProPos. The adapter is not starved. Alpha moves away from its initial state, delta norms are nontrivial, and the embedding-to-CLS cosine changes substantially. Therefore, failed or mixed GLoCA outcomes should not be attributed to "GLoCA did nothing."

The key distinction is whether the moved embedding improves label-aligned clustering under a fixed method. For CDC, it usually does. For seed-42 ProPos, it does not.

## Cross-Method Interpretation

The results separate three different claims:

1. GLoCA can learn under trainable clustering objectives.
2. GLoCA can improve a fixed clustering method.
3. GLoCA improves every clustering method.

The artifacts support the first claim. GLoCA diagnostics show movement away from CLS under both CDC and ProPos.

The artifacts support the second claim for CDC. On PlantSeg, the improvement is consistent across all primary metrics and all three seeds. On PlantVillage, the improvement is consistent for NMI, ACC, and silhouette, with ARI mixed but positive on average.

The artifacts do not support the third claim. ProPos seed 42 gets worse on ARI/NMI/ACC with GLoCA even though silhouette improves. That is exactly why the project rules require matched same-method comparisons.

The most plausible current interpretation is:

```text
GLoCA is useful when the downstream objective can turn patch-token information into disease-aligned clusters.
CDC currently does this better than ProPos in the available artifacts.
ProPos still shows an objective-alignment bottleneck with GLoCA.
```

This does not prove that CDC's calibrated confidence mechanism is the causal reason. CDC reliable-sample counts increase with GLoCA, but calibration error is mixed or worse. The reportable evidence should therefore say that CDC + GLoCA improves clustering, not that calibration quality uniformly improves.

## Limitations

- ProPos only has matched seed-42 artifacts locally. It is not a three-seed ProPos conclusion.
- ProPos has no across-seed standard deviation in this report because `outputs/main_experiments` contains only seed 42 ProPos artifacts.
- PlantWild is excluded from primary evidence because it is not in the current main reported scope and is near-duplicate with PlantSeg.
- `metrics.csv` is treated as the canonical final output. `checkpoint_metrics.csv` is used only for training trajectory and peak behavior.
- CDC calibration metrics do not uniformly improve with GLoCA, despite clustering gains.
- Attention entropy and top-k attention mass indicate attention concentration, not lesion localization or clustering correctness.
- No new `.pt` tensor analysis is included; this report uses CSV and JSON summaries already emitted by the experiments.
- Timing fields are not compared as scientific evidence because prior local reports note timing inconsistencies in some final exports.

## Next Steps

1. Run ProPos seed 67 and seed 69 matched CLS-vs-GLoCA experiments for PlantSeg and PlantVillage before making any ProPos-level statistical claim.
2. For CDC, inspect class-level confusion and cluster composition to identify which disease categories benefit from GLoCA and which remain ambiguous.
3. Add qualitative attention review only as diagnostic context, paired with cluster assignments and labels. Do not use attention heatmaps alone as correctness evidence.
4. Investigate why CDC reliable-sample coverage improves while ECE/MCE often worsen, especially on PlantVillage.
5. Preserve the current controlled comparison contract: same dataset, seed, backbone, clustering method, schedule, evaluation code, and output schema; only the embedding source should differ.

## Final Interpretation

The main three-seed evidence is positive for `CDC + GLoCA`. The strongest statement supported by the current artifacts is:

```text
When CDC is held fixed as the clustering method, replacing normalized DINO CLS embeddings with trained GLoCA embeddings improves reportable clustering performance on PlantSeg and improves most reportable metrics on PlantVillage across three seeds.
```

The current artifacts do not support a universal GLoCA claim. ProPos seed-42 evidence is negative on ARI/NMI/ACC, and calibration metrics under CDC are mixed. The best conclusion is therefore method-aware and conservative: GLoCA can help, but its value depends on whether the clustering objective aligns the adapted patch-informed representation with disease labels.
