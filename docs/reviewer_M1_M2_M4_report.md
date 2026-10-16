# Reviewer Analysis: M1, M2, and M4

All ACC, ARI, NMI, ECE, reliability, confidence, and difference values in this report use the manuscript's 0–100 scale. “pp” means percentage points. Sample standard deviations use `ddof=1`. Analyses were executed on 27 August 2026 without retraining or modifying any original experiment artifact.

## A. Executive Summary

- **M1 is complete.** All 20 main runs were recovered directly and their ACC, ARI, and NMI values were independently reproduced from `assignments.json`. GLoCA improved PlantSeg ACC by **4.45 pp** (95% CI 1.55 to 7.35, paired *p*=0.013), ARI by **5.61 pp** (2.31 to 8.91, *p*=0.009), and NMI by **2.27 pp** (0.53 to 4.01, *p*=0.022); every seed improved. On PlantVillage, ACC improved by **3.38 pp** (1.64 to 5.13, *p*=0.0058) and NMI by **3.73 pp** (1.93 to 5.53, *p*=0.0045), also in all five seeds. PlantVillage ARI increased by **1.89 pp**, but its CI crossed zero (-1.72 to 5.51), *p*=0.220, with three positive and two negative seeds. That comparison is inconclusive with five seeds.
- **M2 is numerically complete.** The exact three-seed CDC row is ACC **43.94 ± 0.86**, ARI **34.41 ± 1.03**, NMI **68.60 ± 0.28**, ECE **4.07 ± 0.72**. Historical arithmetic attributes 3.58 of the 4.93 pp mean ACC gain to No Patch versus CDC and 1.35 pp to Full versus No Patch. The table is not capacity-controlled: CDC, No Patch, Mean Pool, and Full have 513,252; 1,252,069; 1,694,821; and 1,990,886 trainable parameters.
- **M4 selected-pseudo-label correctness is unavailable for the historical runs.** All 20 runs are `APPROXIMATE` under the planning taxonomy because a fresh proxy pass is possible, but no proxy was executed. The final-epoch aggregate counts/ratios and full-dataset ECE/ACC are direct. Historical selected indices, pseudo-label IDs, and selected confidences are not saved, and the stochastic final epoch cannot be replayed exactly from the checkpoint. Consequently, selected accuracy, correct/incorrect selected counts, selected confidence, selected-vs-unselected accuracy, confidence gap, and selected-subset ECE are not reported.
- **The recoverable M4 evidence shows coverage increased while full-dataset calibration worsened.** PlantVillage reliability rose by **16.37 pp** and selected count by **8,891** on average, while ECE rose by **11.07 pp**. This does not establish a coverage–precision tradeoff or a selected confidence–accuracy mismatch because selected correctness is missing.

## B. Data Provenance and Recovery

The analysis used the canonical runs identified in `reviewer_analysis_M1_M2_M4_plan.md`:

- Main CDC: `outputs/main_experiments/base/cdc/{plantseg,plantvillage}/seed_<seed>/`.
- Main CDC+GLoCA: `outputs/main_experiments/gloca_gated/cdc/{plantseg,plantvillage}/seed_<seed>/`.
- Mean Patch Pooling: `outputs/ablation_studies/gloca_mean_pool/cdc/plantseg/seed_<seed>/`.
- No Patch Branch: `outputs/ablation_studies/gloca_no_patches/cdc/plantseg/seed_<seed>/`.

For each run, `metrics.csv` was the direct source for final metrics and final-epoch aggregate selection fields; `assignments.json` was the independent evaluation check; `checkpoint.ckpt` supplied checkpoint-internal configuration; and the original YAML supplied experiment intent. SHA-256 hashes, exact paths, environment versions, commit, and commands are recorded in `artifact_inventory.csv` and `analysis_manifest.json`.

All 26 analyzed assignment files reproduced stored ACC, ARI, and NMI with maximum absolute error **5.55×10^-17**. ACC was recomputed using the repository convention: build a matrix with predicted cluster as row and ground-truth label as column, solve `linear_sum_assignment(matrix.max() - matrix)`, and divide matched mass by N. ARI and NMI were recomputed with scikit-learn.

The recovered five-seed aggregates reproduce every manuscript sanity check within rounding. Run selection was based on path/configuration/checkpoint provenance, not closeness to these aggregates.

## C. M1 — Paired CDC vs GLoCA Statistics

### Complete per-seed data

| Dataset | Seed | CDC ACC | GLoCA ACC | Δ ACC | CDC ARI | GLoCA ARI | Δ ARI | CDC NMI | GLoCA NMI | Δ NMI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PlantSeg | 42 | 43.0267 | 51.2393 | +8.2126 | 33.8451 | 43.3318 | +9.4867 | 68.3840 | 72.6608 | +4.2768 |
| PlantSeg | 67 | 44.7286 | 46.6312 | +1.9026 | 35.6030 | 37.6939 | +2.0909 | 68.5075 | 69.1218 | +0.6143 |
| PlantSeg | 69 | 44.0566 | 48.7345 | +4.6780 | 33.7864 | 39.9399 | +6.1535 | 68.9108 | 71.8805 | +2.9697 |
| PlantSeg | 99 | 42.3460 | 46.2297 | +3.8837 | 32.3467 | 37.7535 | +5.4068 | 67.9420 | 69.7604 | +1.8184 |
| PlantSeg | 123 | 44.0827 | 47.6523 | +3.5696 | 33.5675 | 38.4835 | +4.9161 | 68.5611 | 70.2179 | +1.6567 |
| PlantVillage | 42 | 62.4344 | 66.7397 | +4.3053 | 58.8896 | 64.8776 | +5.9881 | 82.1312 | 87.5245 | +5.3933 |
| PlantVillage | 67 | 61.8194 | 64.6994 | +2.8800 | 59.0966 | 58.7767 | -0.3199 | 82.5995 | 86.0475 | +3.4479 |
| PlantVillage | 69 | 63.8578 | 66.9110 | +3.0531 | 61.7607 | 62.1377 | +0.3771 | 83.6941 | 86.3157 | +2.6216 |
| PlantVillage | 99 | 63.9204 | 69.0876 | +5.1671 | 61.8467 | 65.7838 | +3.9371 | 83.5745 | 88.6335 | +5.0590 |
| PlantVillage | 123 | 60.9723 | 62.4841 | +1.5118 | 58.1967 | 57.6850 | -0.5118 | 82.9086 | 85.0345 | +2.1259 |

### Paired statistics and interpretation

| Dataset | Metric | Mean Δ | SD Δ | SE Δ | 95% CI | *t*(4) | Paired *p* | Wilcoxon W | Wilcoxon *p* | +/0/- | Evidence |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| PlantSeg | ACC | 4.4493 | 2.3342 | 1.0439 | [1.5510, 7.3476] | 4.2622 | 0.013031 | 0 | 0.0625 | 5/0/0 | Clear positive paired evidence |
| PlantSeg | ARI | 5.6108 | 2.6567 | 1.1881 | [2.3121, 8.9095] | 4.7225 | 0.009154 | 0 | 0.0625 | 5/0/0 | Clear positive paired evidence |
| PlantSeg | NMI | 2.2672 | 1.3998 | 0.6260 | [0.5291, 4.0053] | 3.6216 | 0.022326 | 0 | 0.0625 | 5/0/0 | Positive evidence; wider relative uncertainty |
| PlantVillage | ACC | 3.3835 | 1.4051 | 0.6284 | [1.6389, 5.1281] | 5.3846 | 0.005751 | 0 | 0.0625 | 5/0/0 | Clear positive paired evidence |
| PlantVillage | ARI | 1.8941 | 2.9123 | 1.3024 | [-1.7220, 5.5102] | 1.4543 | 0.219543 | 4 | 0.4375 | 3/0/2 | Inconclusive with five seeds |
| PlantVillage | NMI | 3.7296 | 1.4503 | 0.6486 | [1.9287, 5.5304] | 5.7501 | 0.004535 | 0 | 0.0625 | 5/0/0 | Clear positive paired evidence |

The reviewer's PlantVillage concern separates cleanly by metric. ACC improved at every seed, with a positive CI and paired *p*=0.0058. ARI did not improve consistently: seeds 67 and 123 decreased, and the CI includes effects from -1.72 to +5.51 pp. The marginal means alone obscure this distinction.

The Wilcoxon tests are secondary. With five nonzero differences, the smallest attainable exact two-sided value when all signs agree is 0.0625, so failure to cross 0.05 is expected and does not contradict the effect directions or paired *t* results.

## D. M2 — Updated PlantSeg Ablation

### Exact historical per-seed rows

| Variant | Seed | ACC | ARI | NMI | ECE | Trainable params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CDC | 42  | 43.0267 | 33.8451 | 68.3840 | 4.8294 | 513,252 |
| CDC | 67  | 44.7286 | 35.6030 | 68.5075 | 3.9720 | 513,252 |
| CDC | 69  | 44.0566 | 33.7864 | 68.9108 | 3.3976 | 513,252 |
| No Patch Branch | 42  | 48.4203 | 40.0585 | 70.1922 | 4.5961 | 1,252,069 |
| No Patch Branch | 67  | 47.7308 | 38.0042 | 70.6331 | 5.9023 | 1,253,095 |
| No Patch Branch | 69  | 46.3955 | 36.4818 | 69.6917 | 6.6769 | 1,253,095 |
| Mean Patch Pooling | 42  | 45.3046 | 36.2117 | 69.1668 | 7.1580 | 1,694,821 |
| Mean Patch Pooling | 67  | 45.6624 | 36.0538 | 69.3223 | 7.6970 | 1,695,847 |
| Mean Patch Pooling | 69  | 45.6799 | 36.4431 | 69.2769 | 7.6346 | 1,695,847 |
| Full GLoCA | 42  | 51.2393 | 43.3318 | 72.6608 | 4.6523 | 1,990,886 |
| Full GLoCA | 67  | 46.6312 | 37.6939 | 69.1218 | 8.0152 | 1,990,886 |
| Full GLoCA | 69  | 48.7345 | 39.9399 | 71.8805 | 7.3522 | 1,990,886 |

### Trainable capacity

Counts were obtained from the actual module definitions and independently checked by instantiation in the repository's `XDC` environment. The frozen DINOv2 backbone is excluded.

| Variant | Adapter params | CDC clustering head | Calibration head | Total | Added vs CDC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CDC | 0 | 257,139 | 257,139 | 514,278 | 0 |
| No Patch Branch | 738,817 | 257,139 | 257,139 | 1,253,095 | 738,817 |
| Mean Patch Pooling | 1,181,569 | 257,139 | 257,139 | 1,695,847 | 1,181,569 |
| Full GLoCA | 1,477,634 | 257,139 | 257,139 | 1,991,912 | 1,477,634 |

No Patch is 738,817 parameters smaller than Full. Mean Pool is 296,065 smaller than Full. Mean Pool is 442,752 larger than No Patch. Therefore Table 4 cannot be called capacity-controlled. CDC trains its clustering and calibration heads; each GLoCA variant trains those heads plus its adapter, including the learnable residual scale. The backbone remains frozen.

### Paired gain decomposition

| Comparison | Metric | Seed 42 Δ | Seed 67 Δ | Seed 69 Δ | Mean Δ ± SD |
| --- | --- | ---: | ---: | ---: | --- |
| No Patch - CDC | ACC | +5.3936 | +3.0023 | +2.3390 | +3.5783 ± 1.6067 |
| No Patch - CDC | ARI | +6.2134 | +2.4012 | +2.6954 | +3.7700 ± 2.1211 |
| No Patch - CDC | NMI | +1.8082 | +2.1256 | +0.7809 | +1.5716 ± 0.7029 |
| No Patch - CDC | ECE | -0.2334 | +1.9303 | +3.2792 | +1.6587 ± 1.7720 |
| Full - No Patch | ACC | +2.8190 | -1.0997 | +2.3390 | +1.3528 ± 2.1374 |
| Full - No Patch | ARI | +3.2734 | -0.3103 | +3.4581 | +2.1404 ± 2.1244 |
| Full - No Patch | NMI | +2.4686 | -1.5113 | +2.1888 | +1.0487 ± 2.2214 |
| Full - No Patch | ECE | +0.0562 | +2.1129 | +0.6753 | +0.9481 ± 1.0551 |
| Full - CDC | ACC | +8.2126 | +1.9026 | +4.6780 | +4.9311 ± 3.1626 | 
| Full - CDC | ARI | +9.4867 | +2.0909 | +6.1535 | +5.9104 ± 3.7039 | 
| Full - CDC | NMI | +4.2768 | +0.6143 | +2.9697 | +2.6203 ± 1.8561 | 
| Full - CDC | ECE | -0.1772 | +4.0433 | +3.9545 | +2.6069 ± 2.4115 | 
| Mean Pool - No Patch | ACC | -3.1157 | -2.0684 | -0.7157 | -1.9666 ± 1.2033 | 
| Mean Pool - No Patch | ARI | -3.8468 | -1.9504 | -0.0387 | -1.9453 ± 1.9040 | 
| Mean Pool - No Patch | NMI | -1.0253 | -1.3108 | -0.4148 | -0.9170 ± 0.4577 | 
| Mean Pool - No Patch | ECE | +2.5620 | +1.7947 | +0.9577 | +1.7715 ± 0.8024 | 

Historically, the No-Patch arithmetic is 72.6% of the Full-minus-CDC mean ACC gain, 63.8% of ARI, and 60.0% of NMI. These ratios are descriptive, not formal causal decompositions.

Mean pooling hurts all three clustering metrics relative to No Patch at every seed and raises ECE. This supports the narrower observation that naive mean-patch aggregation is not sufficient in these runs. It does not isolate locality because Mean Pool also adds 442,752 parameters over No Patch, and Full adds 738,817 over No Patch.

With n=3, paired SDs are useful descriptions of seed variation. Student-*t* intervals would have df=2 and be extremely unstable, so they are not emphasized.

## E. M4 — Selected Pseudo-Label Quality

### Exactness status and per-run recoverable fields

`metrics.csv` directly stores final-epoch selected count and reliable ratio. It does not store the identities, online pseudo-label IDs, or confidence values that generated those aggregates. Final checkpoints save model weights and some global RNG state, but not persistent-worker augmentation RNG, exact shuffled meta-batch order, or the per-sample online predictions from the changing model. A final checkpoint alone is therefore insufficient to replay the historical final epoch.

All 20 rows below retain the plan's `APPROXIMATE` recoverability label. This means a newly defined checkpoint proxy could be computed, not that the historical result was approximated here. `Approximation_Executed=False` in the CSV; selected accuracy and related fields are intentionally unavailable.

| Dataset | Method | Seed | Total N | Selected N | Reliable % | Correct / incorrect selected | Selected PL accuracy | Mean selected confidence | ECE % | Final ACC % | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| PlantSeg | CDC | 42 | 11,458 | 4,371 | 38.148 | — | — | — | 4.829 | 43.027 | APPROXIMATE; quality not computed |
| PlantSeg | CDC | 67 | 11,458 | 4,397 | 38.375 | — | — | — | 3.972 | 44.729 | APPROXIMATE; quality not computed |
| PlantSeg | CDC | 69 | 11,458 | 4,408 | 38.471 | — | — | — | 3.398 | 44.057 | APPROXIMATE; quality not computed |
| PlantSeg | CDC | 99 | 11,458 | 4,434 | 38.698 | — | — | — | 5.134 | 42.346 | APPROXIMATE; quality not computed |
| PlantSeg | CDC | 123 | 11,458 | 4,432 | 38.680 | — | — | — | 3.229 | 44.083 | APPROXIMATE; quality not computed |
| PlantSeg | CDC+GLoCA | 42 | 11,458 | 5,316 | 46.396 | — | — | — | 4.652 | 51.239 | APPROXIMATE; quality not computed |
| PlantSeg | CDC+GLoCA | 67 | 11,458 | 5,220 | 45.558 | — | — | — | 8.015 | 46.631 | APPROXIMATE; quality not computed |
| PlantSeg | CDC+GLoCA | 69 | 11,458 | 5,317 | 46.404 | — | — | — | 7.352 | 48.735 | APPROXIMATE; quality not computed |
| PlantSeg | CDC+GLoCA | 99 | 11,458 | 5,185 | 45.252 | — | — | — | 7.907 | 46.230 | APPROXIMATE; quality not computed |
| PlantSeg | CDC+GLoCA | 123 | 11,458 | 5,299 | 46.247 | — | — | — | 7.639 | 47.652 | APPROXIMATE; quality not computed |
| PlantVillage | CDC | 42 | 54,305 | 31,720 | 58.411 | — | — | — | 7.407 | 62.434 | APPROXIMATE; quality not computed |
| PlantVillage | CDC | 67 | 54,305 | 32,603 | 60.037 | — | — | — | 8.423 | 61.819 | APPROXIMATE; quality not computed |
| PlantVillage | CDC | 69 | 54,305 | 32,789 | 60.379 | — | — | — | 7.735 | 63.858 | APPROXIMATE; quality not computed |
| PlantVillage | CDC | 99 | 54,305 | 32,623 | 60.074 | — | — | — | 6.570 | 63.920 | APPROXIMATE; quality not computed |
| PlantVillage | CDC | 123 | 54,305 | 32,604 | 60.039 | — | — | — | 9.242 | 60.972 | APPROXIMATE; quality not computed |
| PlantVillage | CDC+GLoCA | 42 | 54,305 | 40,796 | 75.124 | — | — | — | 17.866 | 66.740 | APPROXIMATE; quality not computed |
| PlantVillage | CDC+GLoCA | 67 | 54,305 | 40,569 | 74.706 | — | — | — | 19.631 | 64.699 | APPROXIMATE; quality not computed |
| PlantVillage | CDC+GLoCA | 69 | 54,305 | 41,931 | 77.214 | — | — | — | 18.712 | 66.911 | APPROXIMATE; quality not computed |
| PlantVillage | CDC+GLoCA | 99 | 54,305 | 41,682 | 76.755 | — | — | — | 16.476 | 69.088 | APPROXIMATE; quality not computed |
| PlantVillage | CDC+GLoCA | 123 | 54,305 | 41,818 | 77.006 | — | — | — | 22.031 | 62.484 | APPROXIMATE; quality not computed |

### Five-seed direct summaries

| Dataset | Method | Reliable % | Selected N | Selected PL accuracy | Correct / incorrect selected | Mean selected confidence | ECE % | Final ACC % |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| PlantSeg | CDC | 38.47 ± 0.23 | 4,408.4 ± 26.2 | Unavailable | Unavailable | Unavailable | 4.11 ± 0.85 | 43.65 ± 0.95 |
| PlantSeg | CDC+GLoCA | 45.97 ± 0.53 | 5,267.4 ± 60.9 | Unavailable | Unavailable | Unavailable | 7.11 ± 1.40 | 48.10 ± 2.01 |
| PlantVillage | CDC | 59.79 ± 0.78 | 32,467.8 ± 425.2 | Unavailable | Unavailable | Unavailable | 7.88 ± 1.01 | 62.60 ± 1.29 |
| PlantVillage | CDC+GLoCA | 76.16 ± 1.16 | 41,359.2 ± 629.1 | Unavailable | Unavailable | Unavailable | 18.94 ± 2.08 | 65.98 ± 2.50 |

The available paired differences are:

- PlantSeg: reliability **+7.497 ± 0.661 pp** (95% CI 6.677 to 8.317), selected count **+859.0 ± 75.7**, and ECE **+3.001 ± 1.880 pp** (0.667 to 5.334).
- PlantVillage: reliability **+16.373 ± 0.959 pp** (15.182 to 17.564), selected count **+8,891.4 ± 520.9**, and ECE **+11.068 ± 1.085 pp** (9.721 to 12.415).

The required paired selected-accuracy analysis cannot be calculated. Neither can correct/incorrect selected counts, selected-vs-unselected accuracy, mean selected confidence, selected confidence-minus-accuracy, or selected-subset ECE. These are unavailable, not zero.

If a future instrumented rerun saves historical selected identities and pseudo-label IDs, the approved post-hoc procedure is: derive one Hungarian predicted-cluster-to-label map from the full evaluation collection, freeze it, and apply it to the selected pseudo-label IDs. A mapping must not be refit on the selected subset.

For PlantVillage, the present artifacts establish higher coverage and worse full-dataset ECE. They establish **neither** a coverage–precision tradeoff **nor** a selected confidence–accuracy mismatch, because the precision/correctness denominator and selected confidences are missing. The observed ECE increase is a calibration concern, but it is not a substitute for selected-subset correctness.

## F. Integrated Interpretation

The headline GLoCA improvements are strongest for PlantSeg ACC/ARI/NMI and PlantVillage ACC/NMI: all five paired differences are positive and their paired confidence intervals exclude zero. PlantVillage ARI is not robust at n=5 and should not be presented as a consistent improvement.

The ablation shows that a patch-free trainable adapter coincides with a substantial part of the historical gain and that Full GLoCA retains an additional numerical advantage. However, the stored table cannot tell how much gain is caused specifically by patch locality because adapter capacity changes. The fully matched Full-minus-CDC comparison remains favorable across the three PlantSeg seeds. 

Reliable-sample coverage increases materially with GLoCA, especially on PlantVillage, while full-dataset ECE worsens. Because historical selected-label correctness is unavailable, the paper should describe this as a coverage/calibration pattern rather than improved reliable-label quality or a demonstrated coverage–precision tradeoff.

## G. Paper-Ready Tables

### Table M1

| Dataset | Metric | CDC mean±SD | GLoCA mean±SD | Paired Δ mean±SD | 95% CI | *p* | +/0/- |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| PlantSeg | ACC | 43.65 ± 0.95 | 48.10 ± 2.01 | +4.45 ± 2.33 | [1.55, 7.35] | 0.0130 | 5/0/0 |
| PlantSeg | ARI | 33.83 ± 1.16 | 39.44 ± 2.36 | +5.61 ± 2.66 | [2.31, 8.91] | 0.0092 | 5/0/0 |
| PlantSeg | NMI | 68.46 ± 0.35 | 70.73 ± 1.49 | +2.27 ± 1.40 | [0.53, 4.01] | 0.0223 | 5/0/0 |
| PlantVillage | ACC | 62.60 ± 1.29 | 65.98 ± 2.50 | +3.38 ± 1.41 | [1.64, 5.13] | 0.0058 | 5/0/0 |
| PlantVillage | ARI | 59.96 ± 1.72 | 61.85 ± 3.59 | +1.89 ± 2.91 | [-1.72, 5.51] | 0.2195 | 3/0/2 |
| PlantVillage | NMI | 82.98 ± 0.66 | 86.71 ± 1.39 | +3.73 ± 1.45 | [1.93, 5.53] | 0.0045 | 5/0/0 |

### Updated Table 4

| Variant | Trainable Parameters | ACC | ARI | NMI | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| CDC | 513,252 | 43.94 ± 0.86 | 34.41 ± 1.03 | 68.60 ± 0.28 | 4.07 ± 0.72 |
| No Patch Branch | 1,253,095 | 47.52 ± 1.03 | 38.18 ± 1.79 | 70.17 ± 0.47 | 5.73 ± 1.05 |
| Mean Patch Pooling | 1,695,847 | 45.55 ± 0.21 | 36.24 ± 0.20 | 69.26 ± 0.08 | 7.50 ± 0.29 |
| Full GLoCA | 1,990,886 | 48.87 ± 2.31 | 40.32 ± 2.84 | 71.22 ± 1.86 | 6.67 ± 1.78 |


### Expanded Table 3

| Dataset | Method | Reliable ratio | Selected pseudo-label accuracy | Pseudo-label count | ECE |
| --- | --- | ---: | --- | ---: | ---: |
| PlantSeg | CDC | 38.47 ± 0.23 | Not recoverable | 4,408 ± 26 | 4.11 ± 0.85 |
| PlantSeg | CDC+GLoCA | 45.97 ± 0.53 | Not recoverable | 5,267 ± 61 | 7.11 ± 1.40 |
| PlantVillage | CDC | 59.79 ± 0.78 | Not recoverable | 32,468 ± 425 | 7.88 ± 1.01 |
| PlantVillage | CDC+GLoCA | 76.16 ± 1.16 | Not recoverable | 41,359 ± 629 | 18.94 ± 2.08 |

## H. Reviewer-Response Evidence

### M1

The 20 canonical runs yield matched five-seed improvements in all six mean comparisons. Five comparisons have positive 95% paired CIs; PlantVillage ARI is the exception (mean +1.89 pp, CI -1.72 to 5.51, *p*=0.220, 3 positive/2 negative seeds). ACC improves in every seed on both datasets.

### M2

The exact three-seed CDC row is 43.94 ± 0.86 ACC, 34.41 ± 1.03 ARI, 68.60 ± 0.28 NMI, and 4.07 ± 0.72 ECE. Historical arithmetic places +3.58 pp ACC before patch information and +1.35 pp afterward, but Full adds 738,817 parameters over No Patch. Mean pooling is worse than No Patch on ACC/ARI/NMI at all three seeds and raises ECE.

### M4

GLoCA raises average reliable coverage from 38.47% to 45.97% on PlantSeg and from 59.79% to 76.16% on PlantVillage, while ECE rises from 4.11% to 7.11% and from 7.88% to 18.94%, respectively. Exact selected-pseudo-label correctness cannot be recovered because selected identities and pseudo-label IDs were not saved and the stochastic final epoch cannot be replayed exactly.

## J. Limitations

- Five matched seeds provide limited power and imprecise effect intervals; the Wilcoxon signed-rank test has very coarse resolution at n=5.
- The ablation has only three seeds. Its paired SDs are descriptive, and inferential intervals would be unstable with df=2.
- Parameter counts differ substantially. The ablation is a design-component comparison, not a capacity-controlled estimate of locality.
- Historical M4 selected identities, pseudo-label IDs, and selected confidence values were not saved. Global checkpoint RNG state does not include all dataloader-worker, sampler-order, and augmentation state needed for exact replay.
- A fresh checkpoint proxy was deliberately not executed. Any such future proxy must be separately named, deterministic by definition, and not represented as the historical final epoch.
- The repository worktree was already dirty. All original artifacts were treated read-only; generated files are isolated under `reviewer_analysis_M1_M2_M4/`.
