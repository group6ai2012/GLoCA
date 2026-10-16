# GLoCA experiment report

## Findings

This report brings together the saved PlantSeg and PlantVillage experiments. GLoCA is an embedding adapter between a frozen DINOv2 backbone and a clustering method. Its effect is assessed by comparing normalized DINO CLS and GLoCA embeddings with the **same clustering method**.

- CDC with GLoCA improves mean ARI, NMI, ACC, and silhouette on both datasets across five seeds. ACC improves at every seed. PlantVillage ARI is mixed (three improvements and two decreases), with a paired confidence interval spanning zero.
- The available seed-42 ProPos comparison lowers ARI, NMI, and ACC with GLoCA on both datasets, despite higher silhouette. The CDC result does not imply an improvement for every clustering objective.
- Untrained gated GLoCA stays close to spherical K-Means on CLS. A fixed CLS-plus-mean-patch baseline provides a separate reference without a trainable adapter.
- PlantSeg ablations show that adapter structure matters, but parameter counts and some historical head sizes differ. They do not isolate a capacity-controlled causal effect of patch locality.
- Greater reliable-sample coverage is accompanied by worse mean calibration error. Historical selected-pseudo-label correctness is unavailable.

For installation and commands, see the [user guide](USER_GUIDE.md) and [reproducibility guide](REPRODUCIBILITY.md). This is a curated report of existing results; current deterministic execution does not retroactively reproduce the historical numbers.

## Data, evaluation, and experiment settings

| Dataset | Images | Reference folder categories | Scope |
|---|---:|---:|---|
| PlantSeg | 11,458 | 115 | Disease-field categories from metadata; masks are not used |
| PlantVillage | 54,305 | 38 | Crop-condition categories, including healthy classes |

PlantWild is excluded because of its reported overlap with PlantSeg. It is not independent evidence for the conclusions here.

All reported methods use frozen DINOv2-small (ViT-S/14), with 224 × 224 RGB inputs and ImageNet normalization. Deterministic evaluation uses resize to 256 followed by a center crop to 224. Direct CLS is L2-normalized; gated GLoCA uses a 384-dimensional output, attention hidden size 384, fusion hidden size 768, and residual scale initialized at zero. Training augmentations and evaluation preprocessing are distinct.

CDC uses clustering and calibration MLP heads (384 → 512 → output count), prototype initialization with spherical K-Means and orthogonal refinement, 200 epochs, AdamW at 0.0001 with zero weight decay, physical batches of 256, virtual meta-batches of 2,048, and sub-batches of 256. Calibration mini-cluster counts are 160 on PlantSeg and 128 on PlantVillage. Strong-view clustering loss trains the clustering head and adapter; detached calibration targets and embeddings train the calibration head separately. Final assignments come from the calibration head.

ProPos uses 100 epochs, AdamW at 0.0005, predictor learning-rate multiplier 10, weight decay 0.0001, temperature 0.5, noise scale 0.001, PSL weight 0.1, symmetric losses, EMA momentum increasing from 0.996 to 1.0, and a clustering E-step every epoch. Its configured warmup is 10 epochs (applied while the zero-based epoch is at most 10). Final target-encoder embeddings are clustered with spherical K-Means.

K-Means references use 20 initializations and at most 300 iterations. GMM uses diagonal covariance, three initializations, and at most 300 iterations. HDBSCAN uses minimum cluster size 5 and does not drop noise from evaluation. DEC/IDEC use a 384 → 512 → 512 → 2048 → 64 encoder with a mirrored decoder, 20 pretraining epochs and 10 refinement epochs; IDEC adds reconstruction weight 0.1. The supervised MLP uses 384 → 768 → 384 → category count, class-weighted training for 100 epochs, and a stratified 70/15/15 split.

### Reading the metrics

ARI, NMI, ACC, silhouette, ECE, and reliable ratios use their native scales, not percentages. NMI, ACC, ECE, and reliable ratios are on [0, 1]; ARI and silhouette can be negative. Higher ARI/NMI/ACC is better; lower ECE is better. Silhouette is an internal geometry measure and does not establish disease-label agreement. It uses up to 2,000 sampled embeddings with sampling seed 0.

ACC uses Hungarian cluster-to-label alignment. HDBSCAN noise is remapped to one explicit assignment cluster and remains included in all canonical metrics. Its discovered cluster count can differ from the category count. ECE uses 15 equal-width confidence bins and Hungarian-aligned correctness.

Tables use final exported `metrics.csv` values. Means and sample standard deviations use the stated seeds (`ddof=1`). Single-seed ProPos comparisons have no estimated between-seed uncertainty. Peak checkpoint metrics are reported separately and are not substituted for final results. Unsupervised evaluation is transductive; the supervised MLP is a separate label-trained, test-split reference selected by validation performance.

### Coverage

| Family | Datasets | Seeds | Distinct final result files |
|---|---|---|---:|
| Unsupervised baselines, including ProPos CLS and fixed CLS + mean-patch | Both | 42, 67, 69 | 54 |
| Supervised MLP reference | Both | 42, 67, 69 | 6 |
| CDC: CLS and gated GLoCA | Both | 42, 67, 69, 99, 123 | 20 |
| ProPos with GLoCA | Both | 42 | 2 |
| Three adapter ablations | PlantSeg | 42, 67, 69 | 9 |

The report covers 91 distinct final result files. Reference runs repeated in comparison tables are not additional experiments. All 91 runs' ARI, NMI, and ACC were checked against their saved assignments when consolidating this report (maximum absolute discrepancy below 1e-12).

## Baseline results

### Unsupervised references

| Dataset | Method | ARI mean ± SD | NMI mean ± SD | ACC mean ± SD | Silhouette mean ± SD | Seeds |
|---|---|---|---|---|---|---|
| plantseg | DEC | 0.2630 ± 0.0069 | 0.6330 ± 0.0025 | 0.3663 ± 0.0049 | 0.1672 ± 0.0034 | 42, 67, 69 |
| plantseg | Fixed CLS + mean-patch K-Means | 0.2826 ± 0.0031 | 0.6491 ± 0.0070 | 0.3814 ± 0.0044 | 0.0568 ± 0.0010 | 42, 67, 69 |
| plantseg | GMM | 0.2732 ± 0.0081 | 0.6421 ± 0.0024 | 0.3734 ± 0.0065 | 0.0438 ± 0.0034 | 42, 67, 69 |
| plantseg | HDBSCAN | 0.0157 ± 0.0000 | 0.3254 ± 0.0000 | 0.1385 ± 0.0000 | -0.1442 ± 0.0000 | 42, 67, 69 |
| plantseg | IDEC | 0.2630 ± 0.0069 | 0.6330 ± 0.0025 | 0.3661 ± 0.0048 | 0.1669 ± 0.0035 | 42, 67, 69 |
| plantseg | K-Means | 0.2852 ± 0.0051 | 0.6513 ± 0.0008 | 0.3849 ± 0.0035 | 0.0591 ± 0.0036 | 42, 67, 69 |
| plantseg | ProPos (CLS) | 0.3223 ± 0.0115 | 0.6667 ± 0.0015 | 0.4094 ± 0.0132 | 0.1167 ± 0.0015 | 42, 67, 69 |
| plantseg | Spherical K-Means | 0.2921 ± 0.0050 | 0.6528 ± 0.0015 | 0.3938 ± 0.0050 | 0.0629 ± 0.0028 | 42, 67, 69 |
| plantseg | Spherical K-Means + untrained GLoCA | 0.2927 ± 0.0048 | 0.6533 ± 0.0020 | 0.3945 ± 0.0043 | 0.0631 ± 0.0032 | 42, 67, 69 |
| plantvillage | DEC | 0.5315 ± 0.0379 | 0.7864 ± 0.0170 | 0.5676 ± 0.0321 | 0.0940 ± 0.0059 | 42, 67, 69 |
| plantvillage | Fixed CLS + mean-patch K-Means | 0.5452 ± 0.0278 | 0.7924 ± 0.0081 | 0.5767 ± 0.0268 | 0.0986 ± 0.0051 | 42, 67, 69 |
| plantvillage | GMM | 0.5060 ± 0.0298 | 0.7851 ± 0.0075 | 0.5618 ± 0.0372 | 0.0927 ± 0.0057 | 42, 67, 69 |
| plantvillage | HDBSCAN | 0.0151 ± 0.0000 | 0.1433 ± 0.0000 | 0.1267 ± 0.0000 | 0.1541 ± 0.0000 | 42, 67, 69 |
| plantvillage | IDEC | 0.5314 ± 0.0378 | 0.7863 ± 0.0169 | 0.5675 ± 0.0322 | 0.0933 ± 0.0057 | 42, 67, 69 |
| plantvillage | K-Means | 0.5413 ± 0.0281 | 0.7859 ± 0.0017 | 0.5735 ± 0.0173 | 0.0970 ± 0.0059 | 42, 67, 69 |
| plantvillage | ProPos (CLS) | 0.5726 ± 0.0292 | 0.8205 ± 0.0120 | 0.6182 ± 0.0189 | 0.2167 ± 0.0124 | 42, 67, 69 |
| plantvillage | Spherical K-Means | 0.5161 ± 0.0172 | 0.7839 ± 0.0002 | 0.5629 ± 0.0084 | 0.0944 ± 0.0065 | 42, 67, 69 |
| plantvillage | Spherical K-Means + untrained GLoCA | 0.5159 ± 0.0176 | 0.7837 ± 0.0003 | 0.5628 ± 0.0085 | 0.0942 ± 0.0067 | 42, 67, 69 |

The fixed global/local reference is `normalize(concat(CLS, mean(patch tokens)))`, with 768 dimensions and no learned parameters. It is distinct from GLoCA. Untrained gated GLoCA and direct CLS spherical K-Means have closely similar results; this is an initialization reference rather than evidence of a trained adapter benefit. Comparing different methods in this table does not isolate GLoCA's effect.

HDBSCAN finds 95 non-noise clusters on PlantSeg with 7,128 noise images, and five non-noise clusters on PlantVillage with 693 noise images. Noise participates as an additional assignment cluster. Its near-zero cross-seed variation reflects this deterministic density-clustering reference, not a multi-seed training study.

### Supervised reference (test split only)

| Dataset | Method | ARI mean ± SD | NMI mean ± SD | ACC mean ± SD | Silhouette mean ± SD | Seeds |
|---|---|---|---|---|---|---|
| plantseg | Supervised MLP (test) | 0.6076 ± 0.0015 | 0.8172 ± 0.0026 | 0.7241 ± 0.0076 | 0.0455 ± 0.0044 | 42, 67, 69 |
| plantvillage | Supervised MLP (test) | 0.9892 ± 0.0001 | 0.9853 ± 0.0009 | 0.9909 ± 0.0009 | 0.2528 ± 0.0017 | 42, 67, 69 |

These canonical clustering metrics describe the supervised classifier's test-set assignments. Labels are used for training and validation, and the evaluated sample set differs from the full-dataset unsupervised results. This reference is not evidence for unsupervised GLoCA performance.

## CDC: five-seed CLS versus GLoCA

| Dataset | Embedding | ARI mean ± SD | NMI mean ± SD | ACC mean ± SD | Silhouette mean ± SD | Seeds |
|---|---|---|---|---|---|---|
| plantseg | CLS | 0.3383 ± 0.0116 | 0.6846 ± 0.0035 | 0.4365 ± 0.0095 | 0.0651 ± 0.0028 | 42, 67, 69, 99, 123 |
| plantseg | Gated GLoCA | 0.3944 ± 0.0236 | 0.7073 ± 0.0149 | 0.4810 ± 0.0201 | 0.1085 ± 0.0046 | 42, 67, 69, 99, 123 |
| plantvillage | CLS | 0.5996 ± 0.0172 | 0.8298 ± 0.0066 | 0.6260 ± 0.0129 | 0.0996 ± 0.0016 | 42, 67, 69, 99, 123 |
| plantvillage | Gated GLoCA | 0.6185 ± 0.0359 | 0.8671 ± 0.0139 | 0.6598 ± 0.0250 | 0.1956 ± 0.0087 | 42, 67, 69, 99, 123 |

### Paired uncertainty

Each delta is GLoCA minus CLS at the same seed. The following table alone expresses metric differences on a ×100 scale: for ACC/NMI, these are percentage-point differences; for ARI, they are ARI differences multiplied by 100. Confidence intervals are two-sided paired t intervals over five seeds (four degrees of freedom). P-values are unadjusted, two-sided paired t and exact Wilcoxon signed-rank tests, not population guarantees.

| Dataset | Metric | Mean Δ ×100 ± SD | 95% CI ×100 | Paired t p | Wilcoxon p | Positive/zero/negative seeds |
|---|---|---|---|---|---|---|
| plantseg | ARI | +5.61 ± 2.66 | [+2.31, +8.91] | 0.0092 | 0.0625 | 5/0/0 |
| plantseg | NMI | +2.27 ± 1.40 | [+0.53, +4.01] | 0.0223 | 0.0625 | 5/0/0 |
| plantseg | ACC | +4.45 ± 2.33 | [+1.55, +7.35] | 0.0130 | 0.0625 | 5/0/0 |
| plantvillage | ARI | +1.89 ± 2.91 | [-1.72, +5.51] | 0.2195 | 0.4375 | 3/0/2 |
| plantvillage | NMI | +3.73 ± 1.45 | [+1.93, +5.53] | 0.0045 | 0.0625 | 5/0/0 |
| plantvillage | ACC | +3.38 ± 1.41 | [+1.64, +5.13] | 0.0058 | 0.0625 | 5/0/0 |

PlantSeg improves on all five seeds for ARI, NMI, and ACC. PlantVillage improves on every seed for NMI and ACC, but not ARI: its ARI interval crosses zero. With five nonzero differences, even five concordant signs yield a minimum exact two-sided Wilcoxon p-value of 0.0625. Small sample size and multiple unadjusted comparisons limit inferential claims.

### Occupancy, calibration, and selection

All entries below are mean ± sample SD across five seeds. Cluster sizes refer to occupied clusters. Reliable ratio and selected count are final-training-epoch aggregates; they are not final-inference accuracy estimates.

| Dataset | Embedding | Nonempty clusters | Smallest cluster | Largest cluster | Cluster entropy | ECE | MCE | Reliable ratio | Selected count |
|---|---|---|---|---|---|---|---|---|---|
| plantseg | CLS | 95.6 ± 2.4 | 2.6 ± 2.1 | 535.4 ± 51.5 | 0.9093 ± 0.0066 | 0.0411 ± 0.0085 | 0.0932 ± 0.0367 | 0.3847 ± 0.0023 | 4408.4 ± 26.2 |
| plantseg | Gated GLoCA | 93.6 ± 3.2 | 1.2 ± 0.4 | 460.2 ± 61.4 | 0.9206 ± 0.0085 | 0.0711 ± 0.0140 | 0.1195 ± 0.0167 | 0.4597 ± 0.0053 | 5267.4 ± 60.9 |
| plantvillage | CLS | 36.6 ± 0.9 | 86.2 ± 190.0 | 3216.6 ± 276.0 | 0.9638 ± 0.0073 | 0.0788 ± 0.0101 | 0.2274 ± 0.0559 | 0.5979 ± 0.0078 | 32467.8 ± 425.2 |
| plantvillage | Gated GLoCA | 37.4 ± 0.5 | 605.2 ± 137.3 | 2991.6 ± 529.4 | 0.9805 ± 0.0062 | 0.1894 ± 0.0208 | 0.3195 ± 0.0364 | 0.7616 ± 0.0116 | 41359.2 ± 629.1 |

GLoCA raises mean reliable coverage from 0.3847 to 0.4597 on PlantSeg and from 0.5979 to 0.7616 on PlantVillage. Mean ECE also rises, from 0.0411 to 0.0711 and from 0.0788 to 0.1894, respectively. Increased selection coverage does not establish improved selection precision or uniformly improved calibration.

The historical artifacts did not save selected sample identities, their online pseudo-label IDs, or their selected confidence values. Selected-pseudo-label accuracy, selected-versus-unselected correctness, and selected-subset ECE are therefore unavailable. Final aggregate counts and full-dataset calibration remain directly observed. A fresh checkpoint inference pass would be a new proxy analysis, not recovery of the historical online selection.

## ProPos: seed-42 exploratory comparison

| Dataset | Embedding | Seed | ARI | NMI | ACC | Silhouette |
|---|---|---|---|---|---|---|
| plantseg | CLS | 42 | 0.3302 | 0.6685 | 0.4233 | 0.1181 |
| plantseg | Gated GLoCA | 42 | 0.3247 | 0.6630 | 0.4139 | 0.1361 |
| plantvillage | CLS | 42 | 0.5784 | 0.8191 | 0.6165 | 0.2215 |
| plantvillage | Gated GLoCA | 42 | 0.5059 | 0.7874 | 0.5604 | 0.2555 |

| Dataset | Seed | Δ ARI | Δ NMI | Δ ACC | Δ silhouette |
|---|---|---|---|---|---|
| plantseg | 42 | -0.0055 | -0.0055 | -0.0093 | +0.0180 |
| plantvillage | 42 | -0.0725 | -0.0318 | -0.0561 | +0.0340 |

GLoCA lowers label-aligned ARI, NMI, and ACC on both datasets while increasing silhouette on both. Better internal geometry therefore does not imply better alignment with disease labels. Three CLS seeds are available in the baseline table, but only one GLoCA seed per dataset is available; this is not a multi-seed GLoCA conclusion.

## PlantSeg adapter ablations

The ablations replace gated attention with mean pooling, remove the patch branch, or remove the residual CLS addition. The no-CLS-residual variant still takes CLS and patch information into its fusion MLP, but outputs the normalized correction rather than `z_cls + alpha * delta`.

| Dataset | Variant | ARI mean ± SD | NMI mean ± SD | ACC mean ± SD | Silhouette mean ± SD | Seeds |
|---|---|---|---|---|---|---|
| plantseg | CLS reference | 0.3441 ± 0.0103 | 0.6860 ± 0.0028 | 0.4394 ± 0.0086 | 0.0664 ± 0.0027 | 42, 67, 69 |
| plantseg | Gated GLoCA reference | 0.4032 ± 0.0284 | 0.7122 ± 0.0186 | 0.4887 ± 0.0231 | 0.1082 ± 0.0038 | 42, 67, 69 |
| plantseg | Mean-pool GLoCA | 0.3624 ± 0.0020 | 0.6926 ± 0.0008 | 0.4555 ± 0.0021 | 0.0997 ± 0.0028 | 42, 67, 69 |
| plantseg | No-CLS-residual GLoCA | 0.4178 ± 0.0224 | 0.7218 ± 0.0078 | 0.5034 ± 0.0189 | 0.1649 ± 0.0041 | 42, 67, 69 |
| plantseg | No-patches GLoCA | 0.3818 ± 0.0179 | 0.7017 ± 0.0047 | 0.4752 ± 0.0103 | 0.0974 ± 0.0020 | 42, 67, 69 |

### Capacity and head-size differences

Adapter counts below use the implemented 384-dimensional modules. Total trainable capacity adds both CDC heads at each exported output count, excluding the frozen DINOv2 backbone. Lists are ordered by seeds 42, 67, and 69. The no-CLS-residual module retains its inherited scalar parameter even though the forward output does not use that scalar.

| Variant | Adapter parameters | Total trainable parameters by seed |
|---|---| ---|
| CLS reference | 0 | 513,252, 513,252, 513,252 |
| Gated GLoCA reference | 1,477,634 | 1,990,886, 1,990,886, 1,990,886 |
| Mean-pool GLoCA | 1,181,569 | 1,694,821, 1,695,847, 1,695,847 |
| No-CLS-residual GLoCA | 1,477,634 | 1,990,886, 1,991,912, 1,991,912 |
| No-patches GLoCA | 738,817 | 1,252,069, 1,253,095, 1,253,095 |

Removing the patch branch retains much of the historical ACC gain over direct CLS. Across seeds 42, 67, and 69, No-patches minus CLS is about +3.58 percentage points and gated GLoCA minus No-patches is about +1.35 points. These arithmetic differences are descriptive: adapter capacity and output counts differ, so they cannot attribute that residual gain exclusively to patch locality. Mean pooling is worse than No-patches for ARI, NMI, and ACC on each of these seeds.

The no-CLS-residual variant has the highest mean ARI, NMI, ACC, and silhouette among these adapter variants, but its mean calibration error is also highest. Three seeds and the noted confounds do not establish a universal best adapter.

| Dataset | Variant | Nonempty clusters | Smallest cluster | Largest cluster | Cluster entropy | ECE | MCE | Reliable ratio | Selected count |
|---|---|---|---|---|---|---|---|---|---|
| plantseg | CLS reference | 96.3 ± 3.1 | 2.7 ± 2.9 | 530.3 ± 51.0 | 0.9117 ± 0.0081 | 0.0407 ± 0.0072 | 0.1056 ± 0.0452 | 0.3833 ± 0.0017 | 4392.0 ± 19.0 |
| plantseg | Gated GLoCA reference | 93.3 ± 4.0 | 1.3 ± 0.6 | 473.3 ± 72.5 | 0.9206 ± 0.0102 | 0.0667 ± 0.0178 | 0.1194 ± 0.0235 | 0.4612 ± 0.0049 | 5284.3 ± 55.7 |
| plantseg | Mean-pool GLoCA | 94.7 ± 1.5 | 1.0 ± 0.0 | 453.3 ± 70.6 | 0.9208 ± 0.0083 | 0.0750 ± 0.0029 | 0.1142 ± 0.0109 | 0.4439 ± 0.0021 | 5085.7 ± 24.1 |
| plantseg | No-CLS-residual GLoCA | 98.7 ± 4.7 | 10.3 ± 16.2 | 322.3 ± 57.1 | 0.9454 ± 0.0037 | 0.1648 ± 0.0167 | 0.2186 ± 0.0200 | 0.5559 ± 0.0022 | 6370.0 ± 25.1 |
| plantseg | No-patches GLoCA | 100.0 ± 1.0 | 1.0 ± 0.0 | 483.3 ± 53.6 | 0.9302 ± 0.0074 | 0.0573 ± 0.0105 | 0.1223 ± 0.0316 | 0.4446 ± 0.0029 | 5094.0 ± 32.9 |

## GLoCA learning and checkpoint behavior

The main CDC logs show nonzero alpha movement, changing correction norms, and reduced embedding-to-CLS cosine. First-to-last logged values are:

| Dataset | Seed | Alpha | CLS cosine | Correction norm | Attention entropy |
|---|---|---|---|---|---|
| plantseg | 42 | 0.001 → 0.243 | 1.000 → 0.677 | 8.465 → 146.503 | 5.526 → 4.270 |
| plantseg | 67 | 0.000 → -0.230 | 1.000 → 0.691 | 8.120 → 140.600 | 5.524 → 4.260 |
| plantseg | 69 | 0.000 → 0.249 | 1.000 → 0.706 | 8.379 → 123.670 | 5.529 → 4.417 |
| plantseg | 99 | -0.000 → -0.223 | 1.000 → 0.709 | 8.533 → 147.638 | 5.524 → 4.436 |
| plantseg | 123 | 0.001 → 0.225 | 1.000 → 0.715 | 8.039 → 135.141 | 5.525 → 4.507 |
| plantvillage | 42 | 0.030 → 0.206 | 0.953 → 0.589 | 192.897 → 147.739 | 3.971 → 3.890 |
| plantvillage | 67 | 0.030 → 0.238 | 0.946 → 0.609 | 244.368 → 139.962 | 3.675 → 3.709 |
| plantvillage | 69 | 0.031 → 0.226 | 0.955 → 0.611 | 194.306 → 133.708 | 3.942 → 3.940 |
| plantvillage | 99 | -0.031 → -0.235 | 0.965 → 0.617 | 152.131 → 123.498 | 3.921 → 4.175 |
| plantvillage | 123 | -0.028 → -0.217 | 0.940 → 0.579 | 277.527 → 134.083 | 4.314 → 4.111 |

These diagnostics show representation movement, not proof that the representation is better. Attention entropy describes concentration, not disease localization. Attention maps are supporting diagnostics, not final proof of cluster correctness. Inspect them alongside occupancy and label-aligned metrics.

The checkpoint table in Appendix D separates the best *recorded interval* ARI and ACC from the final export. The ARI epoch and maximum ACC may come from different checkpoints; neither maximum is a label-free model-selection rule. Loss improvement alone is not evidence of improved disease discovery.

## Limits of the evidence

- The strongest same-method evidence is the five-seed CDC comparison; PlantVillage ARI remains inconclusive under the paired uncertainty analysis.
- Different clustering methods, supervised test splits, and full-dataset transductive results are not interchangeable controls.
- Ablation capacity, some historical output counts, and seed coverage differ. The ablations do not isolate locality under a matched parameter budget.
- Historical PlantSeg evaluation label counts change between seed groups, even though CLS/GLoCA inputs and labels match within each reported main pair. The pooled seed statistics must be read with this qualification.
- Historical online selected-pseudo-label correctness cannot be recovered from the saved aggregates or final checkpoint alone.
- Results describe these dataset snapshots and settings. They do not establish generalization to unseen crops, field conditions, or disease diagnosis.
- Current strict execution records input/source/runtime identity and validates completion. The older runs predate that contract; their saved configs and outputs are not a complete historical source/runtime snapshot, and their numbers are not retroactively certified for exact replay.

## Artifact provenance

The tables use the following saved artifact roots, which are local and not distributed in Git. `metrics.csv` supplies final values, `assignments.json` supports independent ARI/NMI/ACC checks, and `logs.json` supplies training diagnostics. Interval histories come from `checkpoint_metrics.csv`. Summary CSVs are convenient indices but are not substituted for individual final metrics.

| Artifact root | Evidence |
|---|---|
| `outputs/baselines_full/` | Standard K-Means, spherical K-Means, untrained GLoCA, DEC, and IDEC references |
| `outputs/cached_cls_baselines/` | GMM, HDBSCAN, and supervised MLP references |
| `outputs/global_local_baseline/` | Fixed CLS + mean-patch reference |
| `outputs/main_experiments/{base,gloca_gated}/cdc/` | Five-seed matched CDC results |
| `outputs/main_experiments/{base,gloca_gated}/propos/` | Three-seed CLS baseline and seed-42 GLoCA comparison |
| `outputs/ablation_studies/` | PlantSeg adapter ablations |

## Appendix A: baseline per-seed results

Reported cluster counts are copied from canonical outputs. The supervised rows evaluate the test split; all other rows evaluate the full image collection.

| Dataset | Method | Seed | ARI | NMI | ACC | Silhouette |
|---|---|---|---|---|---|---|
| plantseg | DEC | 42 | 0.2699 | 0.6354 | 0.3710 | 0.1690 |
| plantseg | DEC | 67 | 0.2630 | 0.6331 | 0.3612 | 0.1693 |
| plantseg | DEC | 69 | 0.2561 | 0.6304 | 0.3666 | 0.1633 |
| plantseg | Fixed CLS + mean-patch K-Means | 42 | 0.2832 | 0.6528 | 0.3785 | 0.0570 |
| plantseg | Fixed CLS + mean-patch K-Means | 67 | 0.2792 | 0.6410 | 0.3792 | 0.0576 |
| plantseg | Fixed CLS + mean-patch K-Means | 69 | 0.2853 | 0.6534 | 0.3865 | 0.0557 |
| plantseg | GMM | 42 | 0.2766 | 0.6411 | 0.3760 | 0.0427 |
| plantseg | GMM | 67 | 0.2790 | 0.6448 | 0.3783 | 0.0476 |
| plantseg | GMM | 69 | 0.2639 | 0.6403 | 0.3660 | 0.0410 |
| plantseg | HDBSCAN | 42 | 115 | 96 | 0.0157 | 0.3254 | 0.1385 | -0.1442 |
| plantseg | HDBSCAN | 67 | 115 | 96 | 0.0157 | 0.3254 | 0.1385 | -0.1442 |
| plantseg | HDBSCAN | 69 | 115 | 96 | 0.0157 | 0.3254 | 0.1385 | -0.1442 |
| plantseg | IDEC | 42 | 0.2698 | 0.6354 | 0.3707 | 0.1687 |
| plantseg | IDEC | 67 | 0.2632 | 0.6333 | 0.3612 | 0.1692 |
| plantseg | IDEC | 69 | 0.2560 | 0.6304 | 0.3664 | 0.1629 |
| plantseg | K-Means | 42 | 0.2830 | 0.6503 | 0.3823 | 0.0571 |
| plantseg | K-Means | 67 | 0.2910 | 0.6519 | 0.3888 | 0.0632 |
| plantseg | K-Means | 69 | 0.2815 | 0.6515 | 0.3836 | 0.0570 |
| plantseg | ProPos (CLS) | 42 | 0.3302 | 0.6685 | 0.4233 | 0.1181 |
| plantseg | ProPos (CLS) | 67 | 0.3277 | 0.6659 | 0.4078 | 0.1168 |
| plantseg | ProPos (CLS) | 69 | 0.3091 | 0.6659 | 0.3969 | 0.1151 |
| plantseg | Spherical K-Means | 42 | 0.2967 | 0.6511 | 0.3995 | 0.0634 |
| plantseg | Spherical K-Means | 67 | 0.2928 | 0.6535 | 0.3906 | 0.0654 |
| plantseg | Spherical K-Means | 69 | 0.2868 | 0.6538 | 0.3912 | 0.0598 |
| plantseg | Spherical K-Means + untrained GLoCA | 42 | 0.2972 | 0.6510 | 0.3994 | 0.0633 |
| plantseg | Spherical K-Means + untrained GLoCA | 67 | 0.2934 | 0.6544 | 0.3913 | 0.0662 |
| plantseg | Spherical K-Means + untrained GLoCA | 69 | 0.2877 | 0.6546 | 0.3927 | 0.0597 |
| plantseg | Supervised MLP (test) | 42 | 0.6060 | 0.8152 | 0.7189 | 0.0453 |
| plantseg | Supervised MLP (test) | 67 | 0.6078 | 0.8201 | 0.7328 | 0.0499 |
| plantseg | Supervised MLP (test) | 69 | 0.6090 | 0.8164 | 0.7206 | 0.0412 |
| plantvillage | DEC | 42 | 0.4878 | 0.7670 | 0.5305 | 0.0871 |
| plantvillage | DEC | 67 | 0.5536 | 0.7940 | 0.5850 | 0.0976 |
| plantvillage | DEC | 69 | 0.5532 | 0.7982 | 0.5872 | 0.0971 |
| plantvillage | Fixed CLS + mean-patch K-Means | 42 | 0.5140 | 0.7832 | 0.5486 | 0.0947 |
| plantvillage | Fixed CLS + mean-patch K-Means | 67 | 0.5674 | 0.7951 | 0.5797 | 0.1044 |
| plantvillage | Fixed CLS + mean-patch K-Means | 69 | 0.5543 | 0.7988 | 0.6020 | 0.0968 |
| plantvillage | GMM | 42 | 0.4749 | 0.7788 | 0.5267 | 0.0869 |
| plantvillage | GMM | 67 | 0.5087 | 0.7831 | 0.5580 | 0.0930 |
| plantvillage | GMM | 69 | 0.5344 | 0.7934 | 0.6008 | 0.0982 |
| plantvillage | HDBSCAN | 42 | 38 | 6 | 0.0151 | 0.1433 | 0.1267 | 0.1541 |
| plantvillage | HDBSCAN | 67 | 38 | 6 | 0.0151 | 0.1433 | 0.1267 | 0.1541 |
| plantvillage | HDBSCAN | 69 | 38 | 6 | 0.0151 | 0.1433 | 0.1267 | 0.1541 |
| plantvillage | IDEC | 42 | 0.4878 | 0.7669 | 0.5304 | 0.0867 |
| plantvillage | IDEC | 67 | 0.5535 | 0.7940 | 0.5851 | 0.0969 |
| plantvillage | IDEC | 69 | 0.5530 | 0.7980 | 0.5870 | 0.0963 |
| plantvillage | K-Means | 42 | 0.5354 | 0.7856 | 0.5750 | 0.0903 |
| plantvillage | K-Means | 67 | 0.5166 | 0.7845 | 0.5555 | 0.1015 |
| plantvillage | K-Means | 69 | 0.5718 | 0.7877 | 0.5900 | 0.0990 |
| plantvillage | ProPos (CLS) | 42 | 0.5784 | 0.8191 | 0.6165 | 0.2215 |
| plantvillage | ProPos (CLS) | 67 | 0.5985 | 0.8331 | 0.6379 | 0.2260 |
| plantvillage | ProPos (CLS) | 69 | 0.5410 | 0.8092 | 0.6002 | 0.2027 |
| plantvillage | Spherical K-Means | 42 | 0.5338 | 0.7839 | 0.5709 | 0.0905 |
| plantvillage | Spherical K-Means | 67 | 0.5153 | 0.7836 | 0.5541 | 0.1019 |
| plantvillage | Spherical K-Means | 69 | 0.4994 | 0.7841 | 0.5636 | 0.0908 |
| plantvillage | Spherical K-Means + untrained GLoCA | 42 | 0.5337 | 0.7840 | 0.5709 | 0.0904 |
| plantvillage | Spherical K-Means + untrained GLoCA | 67 | 0.5152 | 0.7835 | 0.5540 | 0.1019 |
| plantvillage | Spherical K-Means + untrained GLoCA | 69 | 0.4986 | 0.7836 | 0.5635 | 0.0903 |
| plantvillage | Supervised MLP (test) | 42 | 0.9892 | 0.9855 | 0.9908 | 0.2543 |
| plantvillage | Supervised MLP (test) | 67 | 0.9892 | 0.9861 | 0.9918 | 0.2509 |
| plantvillage | Supervised MLP (test) | 69 | 0.9891 | 0.9844 | 0.9901 | 0.2533 |

## Appendix B: CDC per-seed results and paired deltas

| Dataset | Embedding | Seed | ARI | NMI | ACC | Silhouette |
|---|---|---|---|---|---|---|
| plantseg | CLS | 42 | 0.3385 | 0.6838 | 0.4303 | 0.0667 |
| plantseg | CLS | 67 | 0.3560 | 0.6851 | 0.4473 | 0.0636 |
| plantseg | CLS | 69 | 0.3379 | 0.6891 | 0.4406 | 0.0689 |
| plantseg | CLS | 99 | 0.3235 | 0.6794 | 0.4235 | 0.0615 |
| plantseg | CLS | 123 | 0.3357 | 0.6856 | 0.4408 | 0.0648 |
| plantseg | Gated GLoCA | 42 | 0.4333 | 0.7266 | 0.5124 | 0.1122 |
| plantseg | Gated GLoCA | 67 | 0.3769 | 0.6912 | 0.4663 | 0.1048 |
| plantseg | Gated GLoCA | 69 | 0.3994 | 0.7188 | 0.4873 | 0.1076 |
| plantseg | Gated GLoCA | 99 | 0.3775 | 0.6976 | 0.4623 | 0.1036 |
| plantseg | Gated GLoCA | 123 | 0.3848 | 0.7022 | 0.4765 | 0.1141 |
| plantvillage | CLS | 42 | 0.5889 | 0.8213 | 0.6243 | 0.0983 |
| plantvillage | CLS | 67 | 0.5910 | 0.8260 | 0.6182 | 0.0976 |
| plantvillage | CLS | 69 | 0.6176 | 0.8369 | 0.6386 | 0.1003 |
| plantvillage | CLS | 99 | 0.6185 | 0.8357 | 0.6392 | 0.1006 |
| plantvillage | CLS | 123 | 0.5820 | 0.8291 | 0.6097 | 0.1012 |
| plantvillage | Gated GLoCA | 42 | 0.6488 | 0.8752 | 0.6674 | 0.2089 |
| plantvillage | Gated GLoCA | 67 | 0.5878 | 0.8605 | 0.6470 | 0.1862 |
| plantvillage | Gated GLoCA | 69 | 0.6214 | 0.8632 | 0.6691 | 0.1988 |
| plantvillage | Gated GLoCA | 99 | 0.6578 | 0.8863 | 0.6909 | 0.1928 |
| plantvillage | Gated GLoCA | 123 | 0.5768 | 0.8503 | 0.6248 | 0.1910 |

All deltas below use native metric units.

| Dataset | Seed | Δ ARI | Δ NMI | Δ ACC | Δ silhouette |
|---|---|---|---|---|---|
| plantseg | 42 | +0.0949 | +0.0428 | +0.0821 | +0.0456 |
| plantseg | 67 | +0.0209 | +0.0061 | +0.0190 | +0.0412 |
| plantseg | 69 | +0.0615 | +0.0297 | +0.0468 | +0.0387 |
| plantseg | 99 | +0.0541 | +0.0182 | +0.0388 | +0.0421 |
| plantseg | 123 | +0.0492 | +0.0166 | +0.0357 | +0.0493 |
| plantvillage | 42 | +0.0599 | +0.0539 | +0.0431 | +0.1107 |
| plantvillage | 67 | -0.0032 | +0.0345 | +0.0288 | +0.0886 |
| plantvillage | 69 | +0.0038 | +0.0262 | +0.0305 | +0.0985 |
| plantvillage | 99 | +0.0394 | +0.0506 | +0.0517 | +0.0922 |
| plantvillage | 123 | -0.0051 | +0.0213 | +0.0151 | +0.0899 |

### Per-seed calibration and selection

| Dataset | Embedding | Seed | Nonempty clusters | ECE | MCE | Reliable ratio | Selected count |
|---|---|---|---|---|---|---|---|
| plantseg | CLS | 42 | 93 | 0.0483 | 0.1534 | 0.3815 | 4371 |
| plantseg | CLS | 67 | 99 | 0.0397 | 0.1001 | 0.3837 | 4397 |
| plantseg | CLS | 69 | 97 | 0.0340 | 0.0635 | 0.3847 | 4408 |
| plantseg | CLS | 99 | 94 | 0.0513 | 0.0834 | 0.3870 | 4434 |
| plantseg | CLS | 123 | 95 | 0.0323 | 0.0659 | 0.3868 | 4432 |
| plantseg | Gated GLoCA | 42 | 89 | 0.0465 | 0.0968 | 0.4640 | 5316 |
| plantseg | Gated GLoCA | 67 | 97 | 0.0802 | 0.1437 | 0.4556 | 5220 |
| plantseg | Gated GLoCA | 69 | 94 | 0.0735 | 0.1177 | 0.4640 | 5317 |
| plantseg | Gated GLoCA | 99 | 96 | 0.0791 | 0.1217 | 0.4525 | 5185 |
| plantseg | Gated GLoCA | 123 | 92 | 0.0764 | 0.1175 | 0.4625 | 5299 |
| plantvillage | CLS | 42 | 35 | 0.0741 | 0.3048 | 0.5841 | 31720 |
| plantvillage | CLS | 67 | 37 | 0.0842 | 0.1754 | 0.6004 | 32603 |
| plantvillage | CLS | 69 | 37 | 0.0773 | 0.2631 | 0.6038 | 32789 |
| plantvillage | CLS | 99 | 37 | 0.0657 | 0.2152 | 0.6007 | 32623 |
| plantvillage | CLS | 123 | 37 | 0.0924 | 0.1785 | 0.6004 | 32604 |
| plantvillage | Gated GLoCA | 42 | 37 | 0.1787 | 0.3829 | 0.7512 | 40796 |
| plantvillage | Gated GLoCA | 67 | 38 | 0.1963 | 0.2892 | 0.7471 | 40569 |
| plantvillage | Gated GLoCA | 69 | 37 | 0.1871 | 0.3074 | 0.7721 | 41931 |
| plantvillage | Gated GLoCA | 99 | 37 | 0.1648 | 0.3084 | 0.7676 | 41682 |
| plantvillage | Gated GLoCA | 123 | 38 | 0.2203 | 0.3096 | 0.7701 | 41818 |

## Appendix C: ablation per-seed results

The CLS and gated references reuse the corresponding main CDC runs.

| Dataset | Variant | Seed | ARI | NMI | ACC | Silhouette |
|---|---|---|---|---|---|---|
| plantseg | CLS reference | 42 | 0.3385 | 0.6838 | 0.4303 | 0.0667 |
| plantseg | CLS reference | 67 | 0.3560 | 0.6851 | 0.4473 | 0.0636 |
| plantseg | CLS reference | 69 | 0.3379 | 0.6891 | 0.4406 | 0.0689 |
| plantseg | Gated GLoCA reference | 42 | 0.4333 | 0.7266 | 0.5124 | 0.1122 |
| plantseg | Gated GLoCA reference | 67 | 0.3769 | 0.6912 | 0.4663 | 0.1048 |
| plantseg | Gated GLoCA reference | 69 | 0.3994 | 0.7188 | 0.4873 | 0.1076 |
| plantseg | Mean-pool GLoCA | 42 | 0.3621 | 0.6917 | 0.4530 | 0.0964 |
| plantseg | Mean-pool GLoCA | 67 | 0.3605 | 0.6932 | 0.4566 | 0.1009 |
| plantseg | Mean-pool GLoCA | 69 | 0.3644 | 0.6928 | 0.4568 | 0.1016 |
| plantseg | No-CLS-residual GLoCA | 42 | 0.4416 | 0.7308 | 0.5227 | 0.1664 |
| plantseg | No-CLS-residual GLoCA | 67 | 0.3971 | 0.7172 | 0.4849 | 0.1603 |
| plantseg | No-CLS-residual GLoCA | 69 | 0.4147 | 0.7174 | 0.5025 | 0.1679 |
| plantseg | No-patches GLoCA | 42 | 0.4006 | 0.7019 | 0.4842 | 0.0966 |
| plantseg | No-patches GLoCA | 67 | 0.3800 | 0.7063 | 0.4773 | 0.0959 |
| plantseg | No-patches GLoCA | 69 | 0.3648 | 0.6969 | 0.4640 | 0.0997 |

| Variant | Seed | Nonempty clusters | ECE | MCE | Reliable ratio | Selected count |
|---|---|---|---|---|---|---|
| CLS reference | 42 | 93 | 0.0483 | 0.1534 | 0.3815 | 4371 |
| CLS reference | 67 | 99 | 0.0397 | 0.1001 | 0.3837 | 4397 |
| CLS reference | 69 | 97 | 0.0340 | 0.0635 | 0.3847 | 4408 |
| Gated GLoCA reference | 42 | 89 | 0.0465 | 0.0968 | 0.4640 | 5316 |
| Gated GLoCA reference | 67 | 97 | 0.0802 | 0.1437 | 0.4556 | 5220 |
| Gated GLoCA reference | 69 | 94 | 0.0735 | 0.1177 | 0.4640 | 5317 |
| Mean-pool GLoCA | 42 | 95 | 0.0716 | 0.1039 | 0.4415 | 5059 |
| Mean-pool GLoCA | 67 | 96 | 0.0770 | 0.1256 | 0.4456 | 5106 |
| Mean-pool GLoCA | 69 | 93 | 0.0763 | 0.1130 | 0.4444 | 5092 |
| No-CLS-residual GLoCA | 42 | 95 | 0.1470 | 0.1989 | 0.5546 | 6355 |
| No-CLS-residual GLoCA | 67 | 104 | 0.1801 | 0.2388 | 0.5585 | 6399 |
| No-CLS-residual GLoCA | 69 | 97 | 0.1673 | 0.2182 | 0.5547 | 6356 |
| No-patches GLoCA | 42 | 100 | 0.0460 | 0.0934 | 0.4430 | 5076 |
| No-patches GLoCA | 67 | 101 | 0.0590 | 0.1175 | 0.4479 | 5132 |
| No-patches GLoCA | 69 | 99 | 0.0668 | 0.1561 | 0.4428 | 5074 |

## Appendix D: recorded checkpoint peaks versus final scores

These are maxima over saved interval evaluations, not every training epoch. ProPos rows are restricted to the seed-42 CLS/GLoCA comparison. Final scores remain the results used throughout this report.

| Method | Dataset | Embedding | Seed | Peak ARI epoch (1-based) | Peak recorded ARI | Final ARI | Peak recorded ACC | Final ACC |
|---|---|---|---|---|---|---|---|---|
| cdc | plantseg | CLS | 42 | 190 | 0.3421 | 0.3385 | 0.4324 | 0.4303 |
| cdc | plantseg | CLS | 67 | 130 | 0.3594 | 0.3560 | 0.4493 | 0.4473 |
| cdc | plantseg | CLS | 69 | 150 | 0.3385 | 0.3379 | 0.4424 | 0.4406 |
| cdc | plantseg | CLS | 99 | 150 | 0.3300 | 0.3235 | 0.4287 | 0.4235 |
| cdc | plantseg | CLS | 123 | 140 | 0.3361 | 0.3357 | 0.4414 | 0.4408 |
| cdc | plantseg | Gated GLoCA | 42 | 200 | 0.4333 | 0.4333 | 0.5147 | 0.5124 |
| cdc | plantseg | Gated GLoCA | 67 | 200 | 0.3769 | 0.3769 | 0.4663 | 0.4663 |
| cdc | plantseg | Gated GLoCA | 69 | 200 | 0.3994 | 0.3994 | 0.4909 | 0.4873 |
| cdc | plantseg | Gated GLoCA | 99 | 200 | 0.3775 | 0.3775 | 0.4642 | 0.4623 |
| cdc | plantseg | Gated GLoCA | 123 | 190 | 0.3862 | 0.3848 | 0.4780 | 0.4765 |
| cdc | plantvillage | CLS | 42 | 15 | 0.6039 | 0.5889 | 0.6243 | 0.6243 |
| cdc | plantvillage | CLS | 67 | 50 | 0.5910 | 0.5910 | 0.6182 | 0.6182 |
| cdc | plantvillage | CLS | 69 | 50 | 0.6176 | 0.6176 | 0.6417 | 0.6386 |
| cdc | plantvillage | CLS | 99 | 25 | 0.6206 | 0.6185 | 0.6494 | 0.6392 |
| cdc | plantvillage | CLS | 123 | 40 | 0.5845 | 0.5820 | 0.6166 | 0.6097 |
| cdc | plantvillage | Gated GLoCA | 42 | 45 | 0.6500 | 0.6488 | 0.6681 | 0.6674 |
| cdc | plantvillage | Gated GLoCA | 67 | 10 | 0.6240 | 0.5878 | 0.6609 | 0.6470 |
| cdc | plantvillage | Gated GLoCA | 69 | 5 | 0.6405 | 0.6214 | 0.6777 | 0.6691 |
| cdc | plantvillage | Gated GLoCA | 99 | 35 | 0.6584 | 0.6578 | 0.6917 | 0.6909 |
| cdc | plantvillage | Gated GLoCA | 123 | 25 | 0.5901 | 0.5768 | 0.6350 | 0.6248 |
| propos | plantseg | CLS | 42 | 100 | 0.3385 | 0.3302 | 0.4181 | 0.4233 |
| propos | plantseg | Gated GLoCA | 42 | 70 | 0.3292 | 0.3247 | 0.4139 | 0.4139 |
| propos | plantvillage | CLS | 42 | 27 | 0.5664 | 0.5784 | 0.6224 | 0.6165 |
| propos | plantvillage | Gated GLoCA | 42 | 18 | 0.5518 | 0.5059 | 0.5787 | 0.5604 |
