# Full Result Report

Generated from completed local artifacts on 2026-07-11 using the `XDC` conda environment.

## Executive summary

This report consolidates the complete reportable experiment set currently present in the repository:

| Experiment family | Datasets | Seeds | Completed result files | Interpretation status |
|---|---|---|---|---|
| Baselines (standard + cached-CLS + ProPos CLS) | PlantSeg, PlantVillage | 42, 67, 69 | 54 | Full three-seed baseline sweep; 9 baseline heads |
| CDC | PlantSeg, PlantVillage | 42, 67, 69, 99, 123 | 20 | Full five-seed matched CLS/GLoCA comparison |
| ProPos + GLoCA | PlantSeg, PlantVillage | 42 | 2 | Exploratory matched result; not primary-paper evidence |
| GLoCA ablations | PlantSeg | 42, 67, 69 | 9 | Full three-seed ablation comparison |

The controlled comparison is direct DINO CLS versus a GLoCA embedding under the same downstream clustering method. DINOv2 is frozen. PlantWild artifacts are excluded from the primary report because the project current-state document treats PlantWild as smoke/debug context and notes its near-duplicate relationship with PlantSeg.

The strongest result is CDC: GLoCA improves the five-seed mean for all four metrics on both PlantSeg and PlantVillage. ProPos with direct DINO CLS is now a completed three-seed baseline; the seed-42 ProPos + GLoCA comparison remains exploratory and is not primary-paper evidence. The ablations test which parts of the gated adapter matter on PlantSeg.

## Evaluation and reporting conventions

Metrics are read from each run's canonical `metrics.csv`: adjusted Rand index (ARI), normalized mutual information (NMI), Hungarian-aligned clustering accuracy (ACC), and silhouette. Aggregate values are mean ± sample standard deviation across the listed seeds. `metrics.csv` is treated as the final exported result; checkpoint histories are not substituted for final metrics.

Attention maps and scalar diagnostics are supporting diagnostics, not proof of disease localization or clustering correctness. In particular, lower attention entropy means more concentrated attention, not necessarily better disease evidence.

## 1. Three-seed baselines

The three-seed baseline group contains DEC, IDEC, K-Means, spherical K-Means, spherical K-Means with untrained `gloca_gated`, GMM, HDBSCAN, a supervised MLP reference, and ProPos with direct DINO CLS. All are evaluated on PlantSeg and PlantVillage with seeds 42, 67, and 69.

### Aggregate baseline results

| Dataset | Method | ARI mean ± SD | NMI mean ± SD | ACC mean ± SD | SILHOUETTE mean ± SD | Seeds |
|---|---|---|---|---|---|---|
| plantseg | DEC | 0.263 ± 0.007 | 0.633 ± 0.003 | 0.366 ± 0.005 | 0.167 ± 0.003 | 42, 67, 69 |
| plantseg | IDEC | 0.263 ± 0.007 | 0.633 ± 0.002 | 0.366 ± 0.005 | 0.167 ± 0.004 | 42, 67, 69 |
| plantseg | K-Means | 0.285 ± 0.005 | 0.651 ± 0.001 | 0.385 ± 0.003 | 0.059 ± 0.004 | 42, 67, 69 |
| plantseg | Spherical K-Means | 0.292 ± 0.005 | 0.653 ± 0.001 | 0.394 ± 0.005 | 0.063 ± 0.003 | 42, 67, 69 |
| plantseg | Spherical K-Means + untrained GLoCA | 0.293 ± 0.005 | 0.653 ± 0.002 | 0.394 ± 0.004 | 0.063 ± 0.003 | 42, 67, 69 |
| plantvillage | DEC | 0.532 ± 0.038 | 0.786 ± 0.017 | 0.568 ± 0.032 | 0.094 ± 0.006 | 42, 67, 69 |
| plantvillage | IDEC | 0.531 ± 0.038 | 0.786 ± 0.017 | 0.567 ± 0.032 | 0.093 ± 0.006 | 42, 67, 69 |
| plantvillage | K-Means | 0.541 ± 0.028 | 0.786 ± 0.002 | 0.573 ± 0.017 | 0.097 ± 0.006 | 42, 67, 69 |
| plantvillage | Spherical K-Means | 0.516 ± 0.017 | 0.784 ± 0.000 | 0.563 ± 0.008 | 0.094 ± 0.006 | 42, 67, 69 |
| plantvillage | Spherical K-Means + untrained GLoCA | 0.516 ± 0.018 | 0.784 ± 0.000 | 0.563 ± 0.008 | 0.094 ± 0.007 | 42, 67, 69 |

### Complete per-seed baseline results

| Dataset | Seed | Embedding | ARI | NMI | ACC | SILHOUETTE |
|---|---|---|---|---|---|---|
| plantseg | 42 | DEC | 0.270 | 0.635 | 0.371 | 0.169 |
| plantseg | 42 | IDEC | 0.270 | 0.635 | 0.371 | 0.169 |
| plantseg | 42 | K-Means | 0.283 | 0.650 | 0.382 | 0.057 |
| plantseg | 42 | Spherical K-Means + untrained GLoCA | 0.297 | 0.651 | 0.399 | 0.063 |
| plantseg | 42 | Spherical K-Means | 0.297 | 0.651 | 0.400 | 0.063 |
| plantseg | 67 | DEC | 0.263 | 0.633 | 0.361 | 0.169 |
| plantseg | 67 | IDEC | 0.263 | 0.633 | 0.361 | 0.169 |
| plantseg | 67 | K-Means | 0.291 | 0.652 | 0.389 | 0.063 |
| plantseg | 67 | Spherical K-Means + untrained GLoCA | 0.293 | 0.654 | 0.391 | 0.066 |
| plantseg | 67 | Spherical K-Means | 0.293 | 0.654 | 0.391 | 0.065 |
| plantseg | 69 | DEC | 0.256 | 0.630 | 0.367 | 0.163 |
| plantseg | 69 | IDEC | 0.256 | 0.630 | 0.366 | 0.163 |
| plantseg | 69 | K-Means | 0.282 | 0.651 | 0.384 | 0.057 |
| plantseg | 69 | Spherical K-Means + untrained GLoCA | 0.288 | 0.655 | 0.393 | 0.060 |
| plantseg | 69 | Spherical K-Means | 0.287 | 0.654 | 0.391 | 0.060 |
| plantvillage | 42 | DEC | 0.488 | 0.767 | 0.531 | 0.087 |
| plantvillage | 42 | IDEC | 0.488 | 0.767 | 0.530 | 0.087 |
| plantvillage | 42 | K-Means | 0.535 | 0.786 | 0.575 | 0.090 |
| plantvillage | 42 | Spherical K-Means + untrained GLoCA | 0.534 | 0.784 | 0.571 | 0.090 |
| plantvillage | 42 | Spherical K-Means | 0.534 | 0.784 | 0.571 | 0.090 |
| plantvillage | 67 | DEC | 0.554 | 0.794 | 0.585 | 0.098 |
| plantvillage | 67 | IDEC | 0.554 | 0.794 | 0.585 | 0.097 |
| plantvillage | 67 | K-Means | 0.517 | 0.784 | 0.556 | 0.101 |
| plantvillage | 67 | Spherical K-Means + untrained GLoCA | 0.515 | 0.783 | 0.554 | 0.102 |
| plantvillage | 67 | Spherical K-Means | 0.515 | 0.784 | 0.554 | 0.102 |
| plantvillage | 69 | DEC | 0.553 | 0.798 | 0.587 | 0.097 |
| plantvillage | 69 | IDEC | 0.553 | 0.798 | 0.587 | 0.096 |
| plantvillage | 69 | K-Means | 0.572 | 0.788 | 0.590 | 0.099 |
| plantvillage | 69 | Spherical K-Means + untrained GLoCA | 0.499 | 0.784 | 0.563 | 0.090 |
| plantvillage | 69 | Spherical K-Means | 0.499 | 0.784 | 0.564 | 0.091 |

### CLS versus untrained GLoCA spherical K-Means

The untrained adapter is intended to preserve the normalized CLS path at initialization. The matched deltas below are correspondingly small.

| Dataset | Seed | Δ ARI | Δ NMI | Δ ACC | Δ SILHOUETTE |
|---|---|---|---|---|---|
| plantseg | 42 | +0.001 | -0.000 | -0.000 | -0.000 |
| plantseg | 67 | +0.001 | +0.001 | +0.001 | +0.001 |
| plantseg | 69 | +0.001 | +0.001 | +0.002 | -0.000 |
| plantvillage | 42 | -0.000 | +0.000 | +0.000 | -0.000 |
| plantvillage | 67 | -0.000 | -0.000 | -0.000 | +0.000 |
| plantvillage | 69 | -0.001 | -0.000 | -0.000 | -0.001 |

### Cached-CLS baseline extensions

The repository also contains three cached-CLS baselines evaluated with the same three seeds on PlantSeg and PlantVillage. GMM and HDBSCAN are unsupervised references. The supervised MLP uses labels and is therefore reported only as a separate supervised reference, not as primary evidence for unsupervised clustering.

### Aggregate cached-CLS results

| Dataset | Method | ARI mean ± SD | NMI mean ± SD | ACC mean ± SD | SILHOUETTE mean ± SD | Seeds |
|---|---|---|---|---|---|---|
| plantseg | GMM | 0.273 ± 0.008 | 0.642 ± 0.002 | 0.373 ± 0.006 | 0.044 ± 0.003 | 42, 67, 69 |
| plantseg | HDBSCAN | 0.016 ± 0.000 | 0.325 ± 0.000 | 0.139 ± 0.000 | -0.144 ± 0.000 | 42, 67, 69 |
| plantseg | Supervised MLP | 0.608 ± 0.002 | 0.817 ± 0.003 | 0.724 ± 0.008 | 0.045 ± 0.004 | 42, 67, 69 |
| plantvillage | GMM | 0.506 ± 0.030 | 0.785 ± 0.008 | 0.562 ± 0.037 | 0.093 ± 0.006 | 42, 67, 69 |
| plantvillage | HDBSCAN | 0.015 ± 0.000 | 0.143 ± 0.000 | 0.127 ± 0.000 | 0.154 ± 0.000 | 42, 67, 69 |
| plantvillage | Supervised MLP | 0.989 ± 0.000 | 0.985 ± 0.001 | 0.991 ± 0.001 | 0.253 ± 0.002 | 42, 67, 69 |

### Complete per-seed cached-CLS results

| Dataset | Seed | Embedding | ARI | NMI | ACC | SILHOUETTE |
|---|---|---|---|---|---|---|
| plantseg | 42 | GMM | 0.277 | 0.641 | 0.376 | 0.043 |
| plantseg | 42 | HDBSCAN | 0.016 | 0.325 | 0.139 | -0.144 |
| plantseg | 42 | Supervised MLP | 0.606 | 0.815 | 0.719 | 0.045 |
| plantseg | 67 | GMM | 0.279 | 0.645 | 0.378 | 0.048 |
| plantseg | 67 | HDBSCAN | 0.016 | 0.325 | 0.139 | -0.144 |
| plantseg | 67 | Supervised MLP | 0.608 | 0.820 | 0.733 | 0.050 |
| plantseg | 69 | GMM | 0.264 | 0.640 | 0.366 | 0.041 |
| plantseg | 69 | HDBSCAN | 0.016 | 0.325 | 0.139 | -0.144 |
| plantseg | 69 | Supervised MLP | 0.609 | 0.816 | 0.721 | 0.041 |
| plantvillage | 42 | GMM | 0.475 | 0.779 | 0.527 | 0.087 |
| plantvillage | 42 | HDBSCAN | 0.015 | 0.143 | 0.127 | 0.154 |
| plantvillage | 42 | Supervised MLP | 0.989 | 0.985 | 0.991 | 0.254 |
| plantvillage | 67 | GMM | 0.509 | 0.783 | 0.558 | 0.093 |
| plantvillage | 67 | HDBSCAN | 0.015 | 0.143 | 0.127 | 0.154 |
| plantvillage | 67 | Supervised MLP | 0.989 | 0.986 | 0.992 | 0.251 |
| plantvillage | 69 | GMM | 0.534 | 0.793 | 0.601 | 0.098 |
| plantvillage | 69 | HDBSCAN | 0.015 | 0.143 | 0.127 | 0.154 |
| plantvillage | 69 | Supervised MLP | 0.989 | 0.984 | 0.990 | 0.253 |

### ProPos with direct DINO CLS

ProPos with `gloca: disabled` is included as a three-seed trainable clustering baseline. Its results are distinct from the seed-42 ProPos + GLoCA exploration reported later; only the direct-CLS ProPos configuration is designated as baseline evidence.

#### Aggregate ProPos CLS baseline results

| Dataset | Method | ARI mean ± SD | NMI mean ± SD | ACC mean ± SD | SILHOUETTE mean ± SD | Seeds |
|---|---|---|---|---|---|---|
| plantseg | ProPos (DINO CLS) | 0.322 ± 0.012 | 0.667 ± 0.001 | 0.409 ± 0.013 | 0.117 ± 0.001 | 42, 67, 69 |
| plantvillage | ProPos (DINO CLS) | 0.573 ± 0.029 | 0.820 ± 0.012 | 0.618 ± 0.019 | 0.217 ± 0.012 | 42, 67, 69 |

#### Complete per-seed ProPos CLS baseline results

| Dataset | Seed | Embedding | ARI | NMI | ACC | SILHOUETTE |
|---|---|---|---|---|---|---|
| plantseg | 42 | ProPos (DINO CLS) | 0.330 | 0.668 | 0.423 | 0.118 |
| plantseg | 67 | ProPos (DINO CLS) | 0.328 | 0.666 | 0.408 | 0.117 |
| plantseg | 69 | ProPos (DINO CLS) | 0.309 | 0.666 | 0.397 | 0.115 |
| plantvillage | 42 | ProPos (DINO CLS) | 0.578 | 0.819 | 0.617 | 0.221 |
| plantvillage | 67 | ProPos (DINO CLS) | 0.598 | 0.833 | 0.638 | 0.226 |
| plantvillage | 69 | ProPos (DINO CLS) | 0.541 | 0.809 | 0.600 | 0.203 |

## 2. Five-seed CDC results

CDC is the primary multi-seed trainable-method result. Each reportable dataset has matched CLS and GLoCA runs for seeds 42, 67, 69, 99, and 123.

### Aggregate CDC results

| Dataset | Embedding | ARI mean ± SD | NMI mean ± SD | ACC mean ± SD | SILHOUETTE mean ± SD | Seeds |
|---|---|---|---|---|---|---|
| plantseg | CLS | 0.338 ± 0.012 | 0.685 ± 0.003 | 0.436 ± 0.009 | 0.065 ± 0.003 | 42, 67, 69, 99, 123 |
| plantseg | GLoCA | 0.394 ± 0.024 | 0.707 ± 0.015 | 0.481 ± 0.020 | 0.108 ± 0.005 | 42, 67, 69, 99, 123 |
| plantvillage | CLS | 0.600 ± 0.017 | 0.830 ± 0.007 | 0.626 ± 0.013 | 0.100 ± 0.002 | 42, 67, 69, 99, 123 |
| plantvillage | GLoCA | 0.619 ± 0.036 | 0.867 ± 0.014 | 0.660 ± 0.025 | 0.196 ± 0.009 | 42, 67, 69, 99, 123 |

### Complete per-seed CDC final metrics

| Dataset | Seed | Embedding | ARI | NMI | ACC | SILHOUETTE |
|---|---|---|---|---|---|---|
| plantseg | 42 | CLS | 0.338 | 0.684 | 0.430 | 0.067 |
| plantseg | 42 | GLoCA | 0.433 | 0.727 | 0.512 | 0.112 |
| plantseg | 67 | CLS | 0.356 | 0.685 | 0.447 | 0.064 |
| plantseg | 67 | GLoCA | 0.377 | 0.691 | 0.466 | 0.105 |
| plantseg | 69 | CLS | 0.338 | 0.689 | 0.441 | 0.069 |
| plantseg | 69 | GLoCA | 0.399 | 0.719 | 0.487 | 0.108 |
| plantseg | 99 | CLS | 0.323 | 0.679 | 0.423 | 0.061 |
| plantseg | 99 | GLoCA | 0.378 | 0.698 | 0.462 | 0.104 |
| plantseg | 123 | CLS | 0.336 | 0.686 | 0.441 | 0.065 |
| plantseg | 123 | GLoCA | 0.385 | 0.702 | 0.477 | 0.114 |
| plantvillage | 42 | CLS | 0.589 | 0.821 | 0.624 | 0.098 |
| plantvillage | 42 | GLoCA | 0.649 | 0.875 | 0.667 | 0.209 |
| plantvillage | 67 | CLS | 0.591 | 0.826 | 0.618 | 0.098 |
| plantvillage | 67 | GLoCA | 0.588 | 0.860 | 0.647 | 0.186 |
| plantvillage | 69 | CLS | 0.618 | 0.837 | 0.639 | 0.100 |
| plantvillage | 69 | GLoCA | 0.621 | 0.863 | 0.669 | 0.199 |
| plantvillage | 99 | CLS | 0.618 | 0.836 | 0.639 | 0.101 |
| plantvillage | 99 | GLoCA | 0.658 | 0.886 | 0.691 | 0.193 |
| plantvillage | 123 | CLS | 0.582 | 0.829 | 0.610 | 0.101 |
| plantvillage | 123 | GLoCA | 0.577 | 0.850 | 0.625 | 0.191 |

### Matched GLoCA minus CLS deltas

| Dataset | Seed | Δ ARI | Δ NMI | Δ ACC | Δ SILHOUETTE |
|---|---|---|---|---|---|
| plantseg | 42 | +0.095 | +0.043 | +0.082 | +0.046 |
| plantseg | 67 | +0.021 | +0.006 | +0.019 | +0.041 |
| plantseg | 69 | +0.062 | +0.030 | +0.047 | +0.039 |
| plantseg | 99 | +0.054 | +0.018 | +0.039 | +0.042 |
| plantseg | 123 | +0.049 | +0.017 | +0.036 | +0.049 |
| plantvillage | 42 | +0.060 | +0.054 | +0.043 | +0.111 |
| plantvillage | 67 | -0.003 | +0.034 | +0.029 | +0.089 |
| plantvillage | 69 | +0.004 | +0.026 | +0.031 | +0.099 |
| plantvillage | 99 | +0.039 | +0.051 | +0.052 | +0.092 |
| plantvillage | 123 | -0.005 | +0.021 | +0.015 | +0.090 |

On PlantSeg, the GLoCA delta is positive for every metric on every seed. On PlantVillage, NMI, ACC, and silhouette are positive on every seed; ARI is positive on four of five seeds and slightly negative on seed 67. This supports a cautious positive CDC claim without implying that every metric improves on every seed.

### CDC occupancy

Cluster occupancy is reported from the final exported assignments. Higher normalized cluster-size entropy indicates a more even cluster-size distribution; it is not a ground-truth clustering metric on its own.

#### Aggregate occupancy

| Dataset | Embedding | Nonempty clusters mean ± SD | Min cluster mean ± SD | Max cluster mean ± SD | Cluster entropy mean ± SD | Seeds |
|---|---|---|---|---|---|---|
| plantseg | CLS | 95.6 ± 2.4 | 2.6 ± 2.1 | 535.4 ± 51.5 | 0.909 ± 0.007 | 42, 67, 69, 99, 123 |
| plantseg | GLoCA | 93.6 ± 3.2 | 1.2 ± 0.4 | 460.2 ± 61.4 | 0.921 ± 0.009 | 42, 67, 69, 99, 123 |
| plantvillage | CLS | 36.6 ± 0.9 | 86.2 ± 190.0 | 3216.6 ± 276.0 | 0.964 ± 0.007 | 42, 67, 69, 99, 123 |
| plantvillage | GLoCA | 37.4 ± 0.5 | 605.2 ± 137.3 | 2991.6 ± 529.4 | 0.981 ± 0.006 | 42, 67, 69, 99, 123 |

#### Per-seed occupancy

| Dataset | Seed | Embedding | Nonempty clusters | Min cluster | Max cluster | Cluster entropy |
|---|---|---|---|---|---|---|
| plantseg | 42 | CLS | 93 | 1 | 582 | 0.903 |
| plantseg | 42 | GLoCA | 89 | 1 | 473 | 0.911 |
| plantseg | 67 | CLS | 99 | 6 | 529 | 0.914 |
| plantseg | 67 | GLoCA | 97 | 1 | 401 | 0.931 |
| plantseg | 69 | CLS | 97 | 1 | 480 | 0.918 |
| plantseg | 69 | GLoCA | 94 | 2 | 546 | 0.919 |
| plantseg | 99 | CLS | 94 | 3 | 594 | 0.905 |
| plantseg | 99 | GLoCA | 96 | 1 | 400 | 0.927 |
| plantseg | 123 | CLS | 95 | 2 | 492 | 0.906 |
| plantseg | 123 | GLoCA | 92 | 1 | 481 | 0.914 |
| plantvillage | 42 | CLS | 35 | 1 | 3348 | 0.953 |
| plantvillage | 42 | GLoCA | 37 | 510 | 3356 | 0.974 |
| plantvillage | 67 | CLS | 37 | 426 | 3339 | 0.973 |
| plantvillage | 67 | GLoCA | 38 | 471 | 2709 | 0.984 |
| plantvillage | 69 | CLS | 37 | 1 | 3343 | 0.963 |
| plantvillage | 69 | GLoCA | 37 | 825 | 3350 | 0.980 |
| plantvillage | 99 | CLS | 37 | 2 | 3330 | 0.964 |
| plantvillage | 99 | GLoCA | 37 | 610 | 3356 | 0.976 |
| plantvillage | 123 | CLS | 37 | 1 | 2723 | 0.966 |
| plantvillage | 123 | GLoCA | 38 | 610 | 2187 | 0.989 |

Across both datasets, GLoCA generally maintains or increases normalized cluster-size entropy. On PlantVillage, the minimum occupied-cluster size rises sharply under GLoCA, indicating that the final assignments avoid the singleton or near-singleton occupied clusters seen in several CLS runs. These occupancy changes are descriptive and must be read with ARI, NMI, and ACC rather than as evidence of correctness by themselves.

### CDC calibration and reliable-sample coverage

CDC's calibration fields are evaluated from the final exported calibration head. Reliable ratio and pseudo-label count quantify coverage of samples selected as reliable during training; ECE and MCE quantify calibration error, where lower is better.

#### Aggregate calibration and coverage

| Dataset | Embedding | ECE mean ± SD | MCE mean ± SD | Reliable ratio mean ± SD | Pseudo labels mean ± SD | Pseudo-label entropy mean ± SD | Clustering confidence mean ± SD | Calibrated confidence mean ± SD | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| plantseg | CLS | 0.041 ± 0.008 | 0.093 ± 0.037 | 0.385 ± 0.002 | 4408.4 ± 26.2 | 4.323 ± 0.029 | 0.836 ± 0.002 | 0.462 ± 0.002 | 42, 67, 69, 99, 123 |
| plantseg | GLoCA | 0.071 ± 0.014 | 0.119 ± 0.017 | 0.460 ± 0.005 | 5267.4 ± 60.9 | 4.376 ± 0.038 | 0.872 ± 0.006 | 0.552 ± 0.008 | 42, 67, 69, 99, 123 |
| plantvillage | CLS | 0.079 ± 0.010 | 0.227 ± 0.056 | 0.598 ± 0.008 | 32467.8 ± 425.2 | 3.521 ± 0.023 | 0.892 ± 0.006 | 0.701 ± 0.008 | 42, 67, 69, 99, 123 |
| plantvillage | GLoCA | 0.189 ± 0.021 | 0.320 ± 0.036 | 0.762 ± 0.012 | 41359.2 ± 629.1 | 3.575 ± 0.021 | 0.938 ± 0.005 | 0.848 ± 0.006 | 42, 67, 69, 99, 123 |

#### Per-seed calibration and coverage

| Dataset | Seed | Embedding | ECE | MCE | Reliable ratio | Pseudo labels | Pseudo-label entropy | Clustering confidence | Calibrated confidence |
|---|---|---|---|---|---|---|---|---|---|
| plantseg | 42 | CLS | 0.048 | 0.153 | 0.381 | 4371 | 4.294 | 0.836 | 0.462 |
| plantseg | 42 | GLoCA | 0.047 | 0.097 | 0.464 | 5316 | 4.329 | 0.876 | 0.559 |
| plantseg | 67 | CLS | 0.040 | 0.100 | 0.384 | 4397 | 4.341 | 0.834 | 0.459 |
| plantseg | 67 | GLoCA | 0.080 | 0.144 | 0.456 | 5220 | 4.415 | 0.866 | 0.546 |
| plantseg | 69 | CLS | 0.034 | 0.064 | 0.385 | 4408 | 4.365 | 0.834 | 0.463 |
| plantseg | 69 | GLoCA | 0.074 | 0.118 | 0.464 | 5317 | 4.362 | 0.879 | 0.561 |
| plantseg | 99 | CLS | 0.051 | 0.083 | 0.387 | 4434 | 4.306 | 0.840 | 0.463 |
| plantseg | 99 | GLoCA | 0.079 | 0.122 | 0.453 | 5185 | 4.415 | 0.866 | 0.541 |
| plantseg | 123 | CLS | 0.032 | 0.066 | 0.387 | 4432 | 4.309 | 0.836 | 0.464 |
| plantseg | 123 | GLoCA | 0.076 | 0.117 | 0.462 | 5299 | 4.357 | 0.872 | 0.553 |
| plantvillage | 42 | CLS | 0.074 | 0.305 | 0.584 | 31720 | 3.488 | 0.886 | 0.690 |
| plantvillage | 42 | GLoCA | 0.179 | 0.383 | 0.751 | 40796 | 3.551 | 0.936 | 0.843 |
| plantvillage | 67 | CLS | 0.084 | 0.175 | 0.600 | 32603 | 3.548 | 0.891 | 0.702 |
| plantvillage | 67 | GLoCA | 0.196 | 0.289 | 0.747 | 40569 | 3.588 | 0.934 | 0.843 |
| plantvillage | 69 | CLS | 0.077 | 0.263 | 0.604 | 32789 | 3.512 | 0.902 | 0.712 |
| plantvillage | 69 | GLoCA | 0.187 | 0.307 | 0.772 | 41931 | 3.571 | 0.943 | 0.856 |
| plantvillage | 99 | CLS | 0.066 | 0.215 | 0.601 | 32623 | 3.527 | 0.892 | 0.704 |
| plantvillage | 99 | GLoCA | 0.165 | 0.308 | 0.768 | 41682 | 3.560 | 0.943 | 0.854 |
| plantvillage | 123 | CLS | 0.092 | 0.179 | 0.600 | 32604 | 3.530 | 0.891 | 0.696 |
| plantvillage | 123 | GLoCA | 0.220 | 0.310 | 0.770 | 41818 | 3.603 | 0.935 | 0.845 |

GLoCA increases reliable-sample coverage and pseudo-label count on every matched main CDC seed. Calibration error is not uniformly improved: PlantSeg has mixed ECE/MCE outcomes, while PlantVillage ECE and MCE are consistently higher with GLoCA. The supported claim is therefore improved CDC clustering with greater reliable-sample coverage, not universally better calibration.

## 3. ProPos + GLoCA: seed-42 exploratory comparison

The three-seed ProPos CLS baseline is reported in Section 1. This section retains the two matched seed-42 ProPos + GLoCA artifacts for transparency, but they are exploratory diagnostics rather than primary final-paper evidence because no matched GLoCA seed-67 or seed-69 runs are available.

### Seed-42 matched final metrics

| Dataset | Seed | Embedding | ARI | NMI | ACC | SILHOUETTE |
|---|---|---|---|---|---|---|
| plantseg | 42 | CLS | 0.330 | 0.668 | 0.423 | 0.118 |
| plantseg | 42 | GLoCA | 0.325 | 0.663 | 0.414 | 0.136 |
| plantvillage | 42 | CLS | 0.578 | 0.819 | 0.617 | 0.221 |
| plantvillage | 42 | GLoCA | 0.506 | 0.787 | 0.560 | 0.255 |

### Matched GLoCA minus CLS deltas

| Dataset | Seed | Δ ARI | Δ NMI | Δ ACC | Δ SILHOUETTE |
|---|---|---|---|---|---|
| plantseg | 42 | -0.006 | -0.005 | -0.009 | +0.018 |
| plantvillage | 42 | -0.073 | -0.032 | -0.056 | +0.034 |

For the available seed, GLoCA lowers ARI, NMI, and ACC on both datasets while changing silhouette in opposite directions: it decreases on PlantSeg and increases slightly on PlantVillage. This remains useful method-development context, but is not carried forward as a primary final-paper result.

## 4. Three-seed GLoCA ablations

The ablation study uses CDC on PlantSeg with seeds 42, 67, and 69. Each variant changes one structural part of `gloca_gated` while keeping the dataset, seeds, clustering method, and evaluation metrics fixed:

- Mean-pool GLoCA replaces gated attention pooling with mean pooling.
- No-patches GLoCA removes the patch branch.
- No-CLS-residual GLoCA removes the residual CLS output path.

### Aggregate ablation results

| Dataset | Variant | ARI mean ± SD | NMI mean ± SD | ACC mean ± SD | SILHOUETTE mean ± SD | Seeds |
|---|---|---|---|---|---|---|
| plantseg | Gated GLoCA (reference) | 0.403 ± 0.028 | 0.712 ± 0.019 | 0.489 ± 0.023 | 0.108 ± 0.004 | 42, 67, 69 |
| plantseg | Mean-pool GLoCA | 0.362 ± 0.002 | 0.693 ± 0.001 | 0.455 ± 0.002 | 0.100 ± 0.003 | 42, 67, 69 |
| plantseg | No-CLS-residual GLoCA | 0.418 ± 0.022 | 0.722 ± 0.008 | 0.503 ± 0.019 | 0.165 ± 0.004 | 42, 67, 69 |
| plantseg | No-patches GLoCA | 0.382 ± 0.018 | 0.702 ± 0.005 | 0.475 ± 0.010 | 0.097 ± 0.002 | 42, 67, 69 |

### Complete per-seed ablation final metrics

| Dataset | Seed | Embedding | ARI | NMI | ACC | SILHOUETTE |
|---|---|---|---|---|---|---|
| plantseg | 42 | Gated GLoCA (reference) | 0.433 | 0.727 | 0.512 | 0.112 |
| plantseg | 42 | Mean-pool GLoCA | 0.362 | 0.692 | 0.453 | 0.096 |
| plantseg | 42 | No-CLS-residual GLoCA | 0.442 | 0.731 | 0.523 | 0.166 |
| plantseg | 42 | No-patches GLoCA | 0.401 | 0.702 | 0.484 | 0.097 |
| plantseg | 67 | Gated GLoCA (reference) | 0.377 | 0.691 | 0.466 | 0.105 |
| plantseg | 67 | Mean-pool GLoCA | 0.361 | 0.693 | 0.457 | 0.101 |
| plantseg | 67 | No-CLS-residual GLoCA | 0.397 | 0.717 | 0.485 | 0.160 |
| plantseg | 67 | No-patches GLoCA | 0.380 | 0.706 | 0.477 | 0.096 |
| plantseg | 69 | Gated GLoCA (reference) | 0.399 | 0.719 | 0.487 | 0.108 |
| plantseg | 69 | Mean-pool GLoCA | 0.364 | 0.693 | 0.457 | 0.102 |
| plantseg | 69 | No-CLS-residual GLoCA | 0.415 | 0.717 | 0.503 | 0.168 |
| plantseg | 69 | No-patches GLoCA | 0.365 | 0.697 | 0.464 | 0.100 |

### Ablation minus matched gated-GLoCA reference

| Variant | Seed | Δ ARI | Δ NMI | Δ ACC | Δ SILHOUETTE |
|---|---|---|---|---|---|
| Mean-pool GLoCA | 42 | -0.071 | -0.035 | -0.059 | -0.016 |
| No-CLS-residual GLoCA | 42 | +0.008 | +0.004 | +0.010 | +0.054 |
| No-patches GLoCA | 42 | -0.033 | -0.025 | -0.028 | -0.016 |
| Mean-pool GLoCA | 67 | -0.016 | +0.002 | -0.010 | -0.004 |
| No-CLS-residual GLoCA | 67 | +0.020 | +0.026 | +0.019 | +0.055 |
| No-patches GLoCA | 67 | +0.003 | +0.015 | +0.011 | -0.009 |
| Mean-pool GLoCA | 69 | -0.035 | -0.026 | -0.031 | -0.006 |
| No-CLS-residual GLoCA | 69 | +0.015 | -0.001 | +0.015 | +0.060 |
| No-patches GLoCA | 69 | -0.035 | -0.022 | -0.023 | -0.008 |

The ablation aggregates and deltas are controlled against the gated-GLoCA reference on the same three seeds. They are not independent clustering methods; they isolate adapter design choices within the same CDC composition.

### Ablation occupancy

The tables below compare final CDC assignment occupancy for the three adapter variants against the matched gated-GLoCA reference on PlantSeg seeds 42, 67, and 69.

#### Aggregate occupancy

| Dataset | Variant | Nonempty clusters mean ± SD | Min cluster mean ± SD | Max cluster mean ± SD | Cluster entropy mean ± SD | Seeds |
|---|---|---|---|---|---|---|
| plantseg | Gated GLoCA (reference) | 93.3 ± 4.0 | 1.3 ± 0.6 | 473.3 ± 72.5 | 0.921 ± 0.010 | 42, 67, 69 |
| plantseg | Mean-pool GLoCA | 94.7 ± 1.5 | 1.0 ± 0.0 | 453.3 ± 70.6 | 0.921 ± 0.008 | 42, 67, 69 |
| plantseg | No-CLS-residual GLoCA | 98.7 ± 4.7 | 10.3 ± 16.2 | 322.3 ± 57.1 | 0.945 ± 0.004 | 42, 67, 69 |
| plantseg | No-patches GLoCA | 100.0 ± 1.0 | 1.0 ± 0.0 | 483.3 ± 53.6 | 0.930 ± 0.007 | 42, 67, 69 |

#### Per-seed occupancy

| Dataset | Seed | Variant | Nonempty clusters | Min cluster | Max cluster | Cluster entropy |
|---|---|---|---|---|---|---|
| plantseg | 42 | Gated GLoCA (reference) | 89 | 1 | 473 | 0.911 |
| plantseg | 42 | Mean-pool GLoCA | 95 | 1 | 449 | 0.926 |
| plantseg | 42 | No-CLS-residual GLoCA | 95 | 29 | 295 | 0.943 |
| plantseg | 42 | No-patches GLoCA | 100 | 1 | 428 | 0.938 |
| plantseg | 67 | Gated GLoCA (reference) | 97 | 1 | 401 | 0.931 |
| plantseg | 67 | Mean-pool GLoCA | 96 | 1 | 385 | 0.925 |
| plantseg | 67 | No-CLS-residual GLoCA | 104 | 1 | 388 | 0.950 |
| plantseg | 67 | No-patches GLoCA | 101 | 1 | 487 | 0.929 |
| plantseg | 69 | Gated GLoCA (reference) | 94 | 2 | 546 | 0.919 |
| plantseg | 69 | Mean-pool GLoCA | 93 | 1 | 526 | 0.911 |
| plantseg | 69 | No-CLS-residual GLoCA | 97 | 1 | 284 | 0.944 |
| plantseg | 69 | No-patches GLoCA | 99 | 1 | 535 | 0.924 |

No-CLS-residual GLoCA has the highest occupancy entropy and avoids singleton clusters on seed 42, consistent with its stronger ARI/ACC and silhouette in this three-seed ablation. Mean-pool and no-patches variants have lower entropy or more small occupied clusters than the gated reference on several seeds; those patterns are descriptive rather than standalone quality rankings.

### Ablation calibration and reliable-sample coverage

These calibration fields use the same CDC head and final-output convention as the main CDC section, so they can be compared directly within the matched three-seed PlantSeg ablation subset.

#### Aggregate calibration and coverage

| Dataset | Variant | ECE mean ± SD | MCE mean ± SD | Reliable ratio mean ± SD | Pseudo labels mean ± SD | Pseudo-label entropy mean ± SD | Clustering confidence mean ± SD | Calibrated confidence mean ± SD | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| plantseg | Gated GLoCA (reference) | 0.067 ± 0.018 | 0.119 ± 0.023 | 0.461 ± 0.005 | 5284.3 ± 55.7 | 4.369 ± 0.043 | 0.874 ± 0.007 | 0.555 ± 0.008 | 42, 67, 69 |
| plantseg | Mean-pool GLoCA | 0.075 ± 0.003 | 0.114 ± 0.011 | 0.444 ± 0.002 | 5085.7 ± 24.1 | 4.376 ± 0.031 | 0.861 ± 0.006 | 0.530 ± 0.005 | 42, 67, 69 |
| plantseg | No-CLS-residual GLoCA | 0.165 ± 0.017 | 0.219 ± 0.020 | 0.556 ± 0.002 | 6370.0 ± 25.1 | 4.492 ± 0.020 | 0.912 ± 0.002 | 0.668 ± 0.003 | 42, 67, 69 |
| plantseg | No-patches GLoCA | 0.057 ± 0.011 | 0.122 ± 0.032 | 0.445 ± 0.003 | 5094.0 ± 32.9 | 4.425 ± 0.035 | 0.863 ± 0.003 | 0.532 ± 0.003 | 42, 67, 69 |

#### Per-seed calibration and coverage

| Dataset | Seed | Variant | ECE | MCE | Reliable ratio | Pseudo labels | Pseudo-label entropy | Clustering confidence | Calibrated confidence |
|---|---|---|---|---|---|---|---|---|---|
| plantseg | 42 | Gated GLoCA (reference) | 0.047 | 0.097 | 0.464 | 5316 | 4.329 | 0.876 | 0.559 |
| plantseg | 42 | Mean-pool GLoCA | 0.072 | 0.104 | 0.442 | 5059 | 4.396 | 0.855 | 0.525 |
| plantseg | 42 | No-CLS-residual GLoCA | 0.147 | 0.199 | 0.555 | 6355 | 4.472 | 0.912 | 0.669 |
| plantseg | 42 | No-patches GLoCA | 0.046 | 0.093 | 0.443 | 5076 | 4.464 | 0.863 | 0.529 |
| plantseg | 67 | Gated GLoCA (reference) | 0.080 | 0.144 | 0.456 | 5220 | 4.415 | 0.866 | 0.546 |
| plantseg | 67 | Mean-pool GLoCA | 0.077 | 0.126 | 0.446 | 5106 | 4.393 | 0.860 | 0.534 |
| plantseg | 67 | No-CLS-residual GLoCA | 0.180 | 0.239 | 0.558 | 6399 | 4.511 | 0.910 | 0.665 |
| plantseg | 67 | No-patches GLoCA | 0.059 | 0.118 | 0.448 | 5132 | 4.413 | 0.861 | 0.535 |
| plantseg | 69 | Gated GLoCA (reference) | 0.074 | 0.118 | 0.464 | 5317 | 4.362 | 0.879 | 0.561 |
| plantseg | 69 | Mean-pool GLoCA | 0.076 | 0.113 | 0.444 | 5092 | 4.340 | 0.867 | 0.532 |
| plantseg | 69 | No-CLS-residual GLoCA | 0.167 | 0.218 | 0.555 | 6356 | 4.495 | 0.914 | 0.670 |
| plantseg | 69 | No-patches GLoCA | 0.067 | 0.156 | 0.443 | 5074 | 4.397 | 0.866 | 0.531 |

No-CLS-residual GLoCA selects the most reliable samples and has the largest pseudo-label count, but it also has the highest ECE and MCE. Mean-pool and no-patches variants retain lower calibration error than the gated reference on some seeds, while generally selecting fewer reliable samples. The ablations reinforce the main result: reliable-sample coverage, calibration error, and disease-label clustering are related but not interchangeable objectives.

## 5. GLoCA learning diagnostics

The final CDC GLoCA logs include nontrivial parameter movement across all ten runs. The table reports first-to-last values from `gloca_diagnostics_history`.

| Dataset | Seed | Alpha first → last | CLS cosine first → last | Delta norm first → last | Attention entropy first → last |
|---|---|---|---|---|---|
| plantseg | 42 | 0.001 → 0.243 | 1.000 → 0.677 | 8.5 → 146.5 | 5.526 → 4.270 |
| plantseg | 67 | 0.000 → -0.230 | 1.000 → 0.691 | 8.1 → 140.6 | 5.524 → 4.260 |
| plantseg | 69 | 0.000 → 0.249 | 1.000 → 0.706 | 8.4 → 123.7 | 5.529 → 4.417 |
| plantseg | 99 | -0.000 → -0.223 | 1.000 → 0.709 | 8.5 → 147.6 | 5.524 → 4.436 |
| plantseg | 123 | 0.001 → 0.225 | 1.000 → 0.715 | 8.0 → 135.1 | 5.525 → 4.507 |
| plantvillage | 42 | 0.030 → 0.206 | 0.953 → 0.589 | 192.9 → 147.7 | 3.971 → 3.890 |
| plantvillage | 67 | 0.030 → 0.238 | 0.946 → 0.609 | 244.4 → 140.0 | 3.675 → 3.709 |
| plantvillage | 69 | 0.031 → 0.226 | 0.955 → 0.611 | 194.3 → 133.7 | 3.942 → 3.940 |
| plantvillage | 99 | -0.031 → -0.235 | 0.965 → 0.617 | 152.1 → 123.5 | 3.921 → 4.175 |
| plantvillage | 123 | -0.028 → -0.217 | 0.940 → 0.579 | 277.5 → 134.1 | 4.314 → 4.111 |

Alpha movement, nonzero delta norms, and falling CLS cosine show that GLoCA changes the representation during CDC training. The positive CDC results therefore should not be attributed to a permanently inactive adapter. Conversely, representation movement alone is not sufficient: the seed-42 ProPos results show that movement can fail to improve disease-label alignment.

## 6. Overall interpretation

The complete artifact set supports the following bounded conclusions:

1. Untrained `gloca_gated` is nearly equivalent to the direct CLS spherical K-Means baseline, supporting the conservative initialization contract.
2. With CDC held fixed, trained GLoCA improves the five-seed average of ARI, NMI, ACC, and silhouette on PlantSeg, and improves the five-seed average of all four metrics on PlantVillage.
3. ProPos with direct DINO CLS is a completed three-seed baseline. The available seed-42 ProPos + GLoCA comparison is negative for label-aligned metrics on both datasets and remains exploratory rather than primary-paper evidence.
4. The three-seed PlantSeg ablations provide structural evidence about the gated adapter, but they do not replace the matched CDC CLS baseline or establish a universal best adapter without considering variance and seed coverage.
5. Attention concentration and GLoCA scalar movement are diagnostics. They should be interpreted jointly with ARI, NMI, ACC, cluster occupancy, and the matched baseline.

## Limitations and next steps

- ProPos CLS is available across three seeds, but ProPos + GLoCA remains seed-42-only; seed 67 and seed 69 GLoCA runs would be required for a multi-seed ProPos + GLoCA claim.
- The main CDC comparison uses five seeds, while the baseline and ablation families use three seeds; standard deviations are therefore not directly comparable across families as uncertainty estimates.
- PlantWild is omitted from primary evidence by project policy.
- CDC calibration-quality fields should not be interpreted as uniformly improved merely because clustering metrics improve.
- Class-level confusion, cluster composition, and paired qualitative attention review remain useful follow-up analyses.

## Artifact sources

| Family | Artifact root | Canonical file |
|---|---|---|
| Baselines | `outputs/baselines_full/` | `metrics.csv` plus `baseline_summary*.csv` |
| Cached-CLS baselines | `outputs/cached_cls_baselines/` | `metrics.csv` plus `cached_cls_baseline_summary*.csv` |
| CDC | `outputs/main_experiments/{base,gloca_gated}/cdc/` | `metrics.csv`, `logs.json`, `assignments.json` |
| ProPos CLS baseline / ProPos + GLoCA exploratory | `outputs/main_experiments/{base,gloca_gated}/propos/` | `metrics.csv`, `logs.json`, `checkpoint.ckpt` |
| Ablations | `outputs/ablation_studies/` | `metrics.csv`, `logs.json`, `assignments.json` |

The report is reproducible with `conda run -n XDC python scripts/generate_full_result_report.py`.
