# Implementation Details

This document records the implementation settings used for the reportable PlantSeg and PlantVillage experiments. It describes the current local data snapshot, configs, and code paths. PlantWild is retained for smoke tests and historical artifacts, but is not part of the main reported experiment scope.

## Dataset and preprocessing

The repository consumes folder-per-class image datasets through `FolderImageDataset`. The main configs leave `limit_per_class` and `include_classes` unset, so all image files in the processed class directories are used.

| Dataset | Local source snapshot | Final images | Categories | Category definition |
|---|---|---:|---:|---|
| PlantSeg | `data/raw/plantsegv3/images`, converted with `Metadatav2.csv` into `data/raw/plantseg_folder` | 11,458 | 115 | Disease-field categories; current sanitized folder names are crop/disease strings such as `apple_black_rot`, but the class is derived from the metadata `Disease` field rather than from a mask. |
| PlantVillage | `data/raw/plantvillage` | 54,305 | 38 | Crop-condition categories, including healthy classes, such as `Apple___healthy` and `Tomato___Late_blight`. |

The repository records the local PlantSeg source directory and metadata filename, but does not pin an external PlantSeg release identifier. It also does not contain an external PlantVillage version or download manifest; the counts above are from the local folder snapshot used for the report.

PlantSeg masks are not used. The conversion script excludes mask-, annotation-, label-, and segmentation-like paths or filenames, and the training dataset reads only RGB image files from the resulting class folders. The source annotations and metadata `Label file` values are not passed to the model. No reportable category whitelist or per-class cap is applied. The final counts above therefore include all processed image files and exclude non-image annotations and any files rejected by the converter's mask-like filter.

All inputs are converted to RGB and processed at 224 x 224 pixels. ImageNet normalization is applied with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

Deterministic evaluation and prediction preprocessing is:

```text
Resize(256) -> CenterCrop(224) -> ToTensor -> ImageNet Normalize
```

CDC uses three views of each image:

| View | Main operations |
|---|---|
| Weak | `RandomResizedCrop(224)`, random horizontal flip, `ColorJitter(0.4, 0.4, 0.2, 0.1)`, `RandomGrayscale(p=0.2)`, tensor conversion, ImageNet normalization. |
| Strong | Random horizontal flip, `RandomResizedCrop(224)`, four randomly sampled operations from the CDC RandAugment-like list, tensor conversion, ImageNet normalization, then one Cutout hole with configured maximum length 56 and random length enabled. |
| Calibration | The deterministic evaluation transform above. |

The CDC RandAugment-like list contains identity, autocontrast, equalize, rotation, solarize, color, contrast, brightness, sharpness, shear, translation, and posterize operations. Strong-view augmentation is generated once by the dataloader and stored in the CDC virtual meta-batch.

## DINOv2 backbone

The exact configured checkpoint is `facebook/dinov2-small`, i.e. DINOv2 ViT-S/14. The backbone hidden dimension is 384, so the CLS vector is 384-dimensional and each patch token is 384-dimensional. At 224 x 224 with patch size 14, the patch grid is 16 x 16 and contains 256 patch tokens, in addition to the CLS token.

The direct-CLS path uses L2-normalized DINOv2 CLS embeddings. The GLoCA path receives the raw DINOv2 CLS and patch-token outputs and L2-normalizes the final adapter embedding. DINOv2 parameters have `requires_grad=False`; the backbone forward pass is under `torch.no_grad()`, and the backbone is forced to evaluation mode even when the surrounding model is put into training mode.

## GLoCA configuration

The main adapter is `gloca_gated` with:

| Setting | Value |
|---|---:|
| Output embedding dimension | 384 |
| Gated-attention hidden dimension | 384 |
| Fusion-MLP hidden dimension | 768 |
| CLS projection | 384 -> 384 |
| Patch projection | 384 -> 384 |
| Residual scale initialization | `alpha_init: 0.0` |
| Output normalization | L2 normalization enabled |
| GLoCA learning-rate multiplier | 1.0 |
| Alpha learning-rate multiplier | 1.0 |
| Trainable GLoCA parameter count | 1,477,634 |

The gated adapter computes `z_cls`, gated-attention pooled patch features `z_patch`, a fusion correction `delta`, and `normalize(z_cls + alpha * delta)`. The square CLS projection is identity-initialized. The gated-attention linear layers, patch projection, and fusion MLP retain their standard PyTorch linear-layer initialization; there is no additional zero initialization of the gated fusion MLP. `alpha` is the only scalar adapter parameter and starts at zero.

For CDC, GLoCA body and alpha parameters are placed in the clustering optimizer when GLoCA is not frozen. The strong-view clustering cross-entropy loss updates the clustering head and GLoCA. Calibration embeddings and calibration targets are detached before calibration-head training, so calibration loss updates only the calibration head and does not update GLoCA or the clustering head. This preserves the controlled distinction between representation learning and calibration-head fitting.

## CDC configuration

### Head architecture and initialization

CDC has two independent `CDCMLP` heads: a clustering head and a calibration head. Each head is:

```text
Linear(384, 512) -> SafeBatchNorm1d(512) -> ReLU -> Linear(512, C)
```

Here `C` is the dataset category count: 115 for PlantSeg and 38 for PlantVillage. The head hidden dimension is 512.

Prototype initialization is enabled in the main configs. It runs without gradients on deterministic embeddings:

1. Row-zscore the embeddings and run local Torch spherical K-Means with 512 clusters, `kmeans++` initialization, `n_init=10`, `max_iter=100`, and tolerance `1e-4` to initialize the first-layer rows.
2. Apply the first-layer transform, batch normalization, and ReLU, then run a second local Torch spherical K-Means with `C` clusters to initialize the second-layer rows.
3. Copy the resulting prototype weights into both CDC heads.

The external CDC repository is not imported directly. The main configs set `orthogonalize_init: true`, so the repository's CUDA-optional local port of CDC's orthogonal prototype-row refinement is run after the two K-Means stages with the default `orthogonalize_epochs=2000` and `orthogonalize_scale=5.0`. This is a local port of the original `orth_train`-style operation, not execution of the external implementation itself.

### Training schedule

The main CDC schedule is:

| Setting | PlantSeg | PlantVillage |
|---|---:|---:|
| CDC calibration mini-cluster number `K` | 160 | 128 |
| Physical dataloader batch size | 256 | 256 |
| Virtual meta-batch size | 2,048 | 2,048 |
| Optimization sub-batch size | 256 | 256 |
| `per_class_selected_num` | `auto` | `auto` |
| `meta_batch_drop_last` | `false` | `false` |
| Entropy-loss coefficient `w_en` | 1.0 | 1.0 |
| Total training epochs | 200 | 200 |

The mini-cluster number `K` above is the calibration-target K-Means count; it is distinct from the CDC output category count `C`. There is no separate CDC pretraining phase in the main configs. The optimizer is AdamW with learning rate `1e-4` and weight decay `0.0`. There is no learning-rate scheduler. The clustering optimizer owns the clustering head and trainable GLoCA parameters; the calibration optimizer owns only the calibration head.

CDC uses reliable weak-view confidence and pseudo-label selection over each virtual meta-batch. The strong view is used for clustering-head cross-entropy on selected reliable samples. The calibration view is used to construct detached mini-cluster targets and to train the calibration head. Final prediction assignments and confidence values come from the calibration head.

## Model selection and inference

The main CDC result table uses the final trained model state after the configured 200 epochs and the final exported `checkpoint.ckpt`; it does not substitute the best checkpoint from `checkpoint_metrics.csv`. Checkpoint metrics are monitoring records only. Ground-truth labels are not used to select an unsupervised CDC checkpoint or to tune CDC hyperparameters.

Inference uses the deterministic single-view preprocessing described above. CDC, K-Means, spherical K-Means, GMM, HDBSCAN, DEC, IDEC, and ProPos are transductive: the same unlabeled image collection is used to fit the embedding head or clustering model and to produce the final assignments, with labels used only afterward for evaluation. The supervised MLP is the explicit exception: it uses a stratified 70/15/15 train/validation/test split from the same collection and selects its checkpoint by validation accuracy, breaking ties with validation loss.

## Baselines

All unsupervised baselines use normalized DINOv2 CLS features unless stated otherwise. For the main datasets, `K` or the number of mixture components is the dataset category count `C`.

| Baseline | Important settings |
|---|---|
| K-Means | Local Torch implementation; `K=C`; direct CLS features are already L2-normalized; `kmeans++`, `n_init=20`, `max_iter=300`, tolerance `1e-4`. |
| Spherical K-Means | The same local Torch implementation with spherical/cosine updates; features are L2-normalized inside the fit path; `K=C`, `kmeans++`, `n_init=20`, `max_iter=300`, tolerance `1e-4`. |
| GMM | scikit-learn `GaussianMixture`; `n_components=C`, diagonal covariance, `init_params=k-means++`, `n_init=3`, `max_iter=300`, tolerance `1e-3`, regularization `1e-6`. |
| HDBSCAN | scikit-learn HDBSCAN; fixed `min_cluster_size=5`, `min_samples=None`, `cluster_selection_epsilon=0.0`, Euclidean metric, `algorithm=auto`, `leaf_size=40`, `cluster_selection_method=eom`, `allow_single_cluster=false`, `n_jobs=-1`; no hyperparameter search. On the cached report artifacts it returns 95 non-noise clusters plus one noise assignment on PlantSeg, and 5 non-noise clusters plus one noise assignment on PlantVillage. |
| DEC | Standalone DINO-CLS autoencoder, `384 -> 512 -> 512 -> 2048 -> 64` encoder with mirrored decoder; latent dimension 64; Student-t parameter `alpha=1.0`; 20 pretraining epochs and 10 refinement epochs; Adam with pretraining LR `1e-3` and refinement LR `1e-4`; reconstruction weight 0.0. |
| IDEC | Same architecture and schedule as DEC, but reconstruction weight `lambda_recon=0.1`; Adam with pretraining LR `1e-3` and refinement LR `1e-4`. |
| ProPos | 100 training epochs; AdamW, LR `5e-4`, predictor LR multiplier 10, weight decay `1e-4`; temperature `0.5`, latent-noise scale `sigma=0.001`, `lambda_psl=0.1`, symmetric PSA/PSL loss, EMA momentum starting at `0.996` and increasing to `1.0`; configured `warmup_epochs=10` (the implementation applies warmup while `epoch <= warmup_epochs`), K-Means E-step every epoch, `n_init=10`. The final state is used and final target-encoder embeddings are clustered with final spherical K-Means; no label-selected checkpoint is used. |
| Supervised MLP | Cached CLS classifier `384 -> 768 -> GELU -> 384 -> GELU -> C`; AdamW, LR `1e-3`, weight decay `1e-4`, class-weighted cross-entropy, 100 epochs; stratified per-class 70/15/15 split; best validation accuracy, then validation loss, selects the checkpoint, and canonical classification outputs are test-split only. |

HDBSCAN raw label `-1` is retained in diagnostic logs and remapped to one explicit nonnegative assignment cluster for canonical metrics. PlantSeg has 95 discovered non-noise clusters and 7,128 noise points out of 11,458 images; PlantVillage has 5 discovered non-noise clusters and 693 noise points out of 54,305 images. Noise is not dropped or downweighted.

## Evaluation

The canonical metric implementation is `src/evaluation/clustering_metrics.py`:

- **ARI:** `sklearn.metrics.adjusted_rand_score(labels, assignments)`.
- **NMI:** `sklearn.metrics.normalized_mutual_info_score(labels, assignments)` with the library default averaging behavior.
- **ACC:** Hungarian-aligned clustering accuracy. The contingency matrix is `M[predicted_cluster, true_class]`; the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) selects the one-to-one matching maximizing the matched counts, and ACC is the matched-count sum divided by the number of samples. The matrix is padded to a square size sufficient for all label and assignment IDs, so the predicted cluster count need not equal `C`.

HDBSCAN noise participates in ARI, NMI, ACC, and silhouette through its explicit remapped assignment. It is not removed before evaluation. The raw HDBSCAN cluster count and noise count are reported separately.

CDC calibration uses `calibration_error_metrics` with 15 equal-width confidence bins on `[0, 1]`. Confidence is the maximum softmax probability from the final calibration head. For each bin, correctness is computed after applying the Hungarian predicted-cluster-to-true-class mapping; ECE is the sample-fraction-weighted mean absolute accuracy-confidence gap, and MCE is the maximum bin gap.

`reliable_sample_ratio` is the final training epoch's total selected reliable samples divided by the total samples seen across that epoch's virtual meta-batches. `pseudo_label_count` is the final epoch's total number of selected pseudo-labels, not a training-history average and not a final inference count. The report aggregates these final-run values across seeds.

## Seeds and aggregation

| Experiment family | Seeds |
|---|---|
| Standard and cached-CLS baselines | 42, 67, 69 |
| Main CDC matched CLS/GLoCA results | 42, 67, 69, 99, 123 |
| ProPos direct-CLS baseline | 42, 67, 69 |
| ProPos + GLoCA exploratory comparison | 42 |
| PlantSeg GLoCA ablations | 42, 67, 69 |

Mean plus/minus standard deviation in the report uses the sample standard deviation with `ddof=1`; a single-run group is displayed with standard deviation zero.

## Compute and reproducibility

Experiments were run with an NVIDIA GeForce RTX 4070 Laptop GPU, Python 3.12.13, PyTorch 2.12.0+cu132, scikit-learn 1.4.2, and repository `https://github.com/group6ai2012/GLoCA.git` at commit `1716027bf76023316d2c721b4eff0f68669542e5`.

The working tree used for this report also contained uncommitted configuration, source, and documentation changes, so the commit hash alone is not a complete snapshot of the exact experiment state; the per-run `config.yaml` and `logs.json` files are the authoritative artifact-level records.

## Implementation references

Key implementation files are:

- Dataset registry, folder loading, and CDC transforms: `src/data/registry.py`, `src/data/folder_dataset.py`, `src/data/transforms.py`.
- PlantSeg conversion and mask filtering: `scripts/prepare_plantseg_folder_dataset.py`.
- DINOv2 wrapper: `src/features/dinov2_backbone.py`.
- GLoCA adapter: `src/models/gloca.py`.
- CDC head and training loop: `src/models/clustering/cdc.py`, `src/training/cdc_trainer.py`.
- Baseline implementations: `src/models/clustering/kmeans.py`, `src/models/baselines/dec_idec.py`, `src/models/baselines/cached_cls.py`, `src/training/propos_trainer.py`.
- Metric definitions and output aggregation: `src/evaluation/clustering_metrics.py`, `scripts/generate_full_result_report.py`.
