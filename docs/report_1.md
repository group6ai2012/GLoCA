# Report 1: ProPos 200-Epoch PlantWild Analysis

Date: 2026-06-13

This report analyzes the two latest completed ProPos PlantWild runs:

- `outputs/propos_full/propos_cls_full_200ep/propos/plantwild/seed_42`
- `outputs/propos_full/propos_gloca_gated_full_200ep/propos/plantwild/seed_42`

It also compares those runs against the current PlantWild baseline sweep in:

- `outputs/baselines_full/baseline_summary.csv`
- `outputs/baselines_full/baseline_summary_agg.csv`

Important caveat: these two completed runs use the same dataset, seed, backbone, head family, evaluation code, and output schema, but they are not perfectly paper-grade matched because `kmeans_interval` differs. The CLS run uses `kmeans_interval=1`; the GLoCA-gated run uses `kmeans_interval=2`. The findings below should be treated as strong implementation and diagnostic evidence. A final paper table should rerun the matched pair with the same ProPos schedule.

## Executive Summary

ProPos is now validated as a working implemented method in this repository. The implementation trains for 200 epochs, writes the expected output artifacts, keeps DINOv2 frozen, uses live two-view training, exports deterministic prediction embeddings, and produces competitive PlantWild clustering metrics.

The matched ProPos comparison does not show that GLoCA-gated is broken. On the contrary, GLoCA-gated reaches essentially the same final ARI as CLS-ProPos, improves final ACC and silhouette, and achieves similar peak ARI/ACC in the epoch history. It also changes the embedding substantially and optimizes the ProPos loss more strongly than CLS in final loss terms.

The main bottleneck is not alpha starvation or absent GLoCA learning. GLoCA learns strongly: final alpha moves from `0.0` to `-8.4046`, embedding-to-CLS cosine falls to about `0.537`, and attention becomes more concentrated than the initial near-uniform state. The bottleneck is objective alignment: patch-token information helps ProPos optimize its own objective, but it does not reliably improve alignment with ground-truth disease labels.

Subsequent qualitative attention-map inspection refines the interpretation of the high attention entropy values. High entropy should not be read as direct evidence that GLoCA is failing to attend to disease evidence. In some classes, GLoCA attends to the visible disease regions very well; in other classes it attends to background, borders, artifacts, or non-lesion object regions. More importantly, attention localization and final cluster alignment are not equivalent: disease-localizing attention can still produce fragmented clusters, while weakly localized attention can still coincide with a cohesive cluster.

A preliminary preprocessing ablation also suggests that full-frame preservation alone is not the solution. Resize-pad and resize-stretch variants did not improve ProPos-CLS behavior relative to the existing preprocessing pipeline. The current crop/resize pipeline should remain the practical default for now, with preprocessing retained as a qualitative caveat rather than the next main optimization axis.

More ProPos epochs are not the next priority. The most useful metric window appears around roughly epochs `70-120`, while later loss improvement does not consistently translate to better ARI or ACC. The next research direction should freeze ProPos as an implemented baseline/diagnostic method, report its findings, and move to CDC as the next main method.

## Final Metric Comparison

These are final `metrics.csv` values from the two 200-epoch seed-42 ProPos runs.

| Run | GLoCA | ARI | NMI | ACC | Silhouette | Final total loss | Final PSA | Final PSL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ProPos CLS, 200 ep | disabled | 0.326506 | 0.666694 | 0.400853 | 0.114619 | -1.523476 | -1.764363 | 2.408874 |
| ProPos GLoCA-gated, 200 ep | gloca_gated | 0.326628 | 0.654514 | 0.417653 | 0.132384 | -1.580791 | -1.823843 | 2.430524 |

Interpretation:

- Final ARI is effectively tied: GLoCA is higher by only `0.000122`.
- Final ACC is better for GLoCA by `0.016800`.
- Final silhouette is better for GLoCA by `0.017765`.
- Final NMI is lower for GLoCA by `0.012179`.
- GLoCA reaches a better final ProPos total loss and PSA loss.
- PSL is slightly higher for GLoCA, so the loss advantage is mainly driven by stronger PSA alignment.

This is a non-catastrophic result. GLoCA-gated is not causing collapse under ProPos.

## Epoch-History Peaks

The final `metrics.csv` row is not the whole story because the logs contain per-epoch clustering diagnostics. The best observed epochs were:

| Run | Best ARI epoch | Best ARI | Best ACC epoch | Best ACC | Best NMI epoch | Best NMI |
|---|---:|---:|---:|---:|---:|---:|
| ProPos CLS | 58 | 0.350289 | 70 | 0.433670 | 155 | 0.674839 |
| ProPos GLoCA-gated | 66 | 0.352669 | 118 | 0.443332 | 66 | 0.674425 |

GLoCA-gated reaches the higher observed peak ARI and ACC:

- Peak ARI: `0.352669` for GLoCA vs. `0.350289` for CLS.
- Peak ACC: `0.443332` for GLoCA vs. `0.433670` for CLS.
- Peak NMI is nearly tied: `0.674425` for GLoCA vs. `0.674839` for CLS.

This supports the conclusion that GLoCA-gated + ProPos is functioning and can reach comparable peak performance to CLS-ProPos.

## Epoch Window and Late-Training Behavior

Selected epoch-history values:

| Run | Epoch | ARI | NMI | ACC | Loss |
|---|---:|---:|---:|---:|---:|
| CLS | 50 | 0.335280 | 0.666220 | 0.419133 | -1.456139 |
| CLS | 70 | 0.345184 | 0.671783 | 0.433670 | -1.456834 |
| CLS | 100 | 0.332816 | 0.667175 | 0.424095 | -1.465138 |
| CLS | 120 | 0.332462 | 0.669276 | 0.413301 | -1.469289 |
| CLS | 199 | 0.345719 | 0.672082 | 0.429405 | -1.478349 |
| GLoCA | 50 | 0.331337 | 0.671892 | 0.422528 | -1.506834 |
| GLoCA | 70 | 0.328745 | 0.662788 | 0.415303 | -1.519011 |
| GLoCA | 100 | 0.330432 | 0.657423 | 0.421919 | -1.531602 |
| GLoCA | 120 | 0.334381 | 0.659979 | 0.413910 | -1.533936 |
| GLoCA | 199 | 0.322895 | 0.651259 | 0.402942 | -1.552862 |

The useful window appears roughly around epochs `70-120`, with important peaks slightly before or inside that region:

- CLS best ACC occurs at epoch `70`.
- GLoCA best ACC occurs at epoch `118`.
- GLoCA best ARI/NMI occurs at epoch `66`.

After that, loss keeps improving, especially for GLoCA, but disease-label clustering metrics do not improve in lockstep. This is evidence against simply running ProPos longer as the next priority.

## GLoCA Learning Diagnostics

GLoCA-gated is not alpha-starved and is not failing to learn.

Key diagnostics from the GLoCA run:

| Epoch | Alpha | CLS cosine | Delta norm | Attention entropy | Attention max |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0895 | 0.9990 | 9.3945 | 5.4781 | 0.0084 |
| 50 | -3.1198 | 0.7840 | 11.3577 | 4.7642 | 0.0440 |
| 66 | -4.0455 | 0.7169 | 11.0438 | 4.7794 | 0.0443 |
| 100 | -5.6346 | 0.6276 | 10.9091 | 4.7921 | 0.0450 |
| 118 | -6.3960 | 0.5964 | 10.8081 | 4.8149 | 0.0433 |
| 199 | -8.4046 | 0.5372 | 10.7832 | 4.8229 | 0.0432 |

Final `metrics.csv` attention statistics:

- `attention_entropy = 4.868949`
- `attention_max = 0.039434`
- `attention_top5_mass = 0.150596`
- `attention_variance = 0.00003585`

The correction is substantial:

- alpha moves far from the zero initialization,
- the final embedding is no longer close to CLS,
- the patch correction has a large norm,
- attention moves away from the near-uniform start.

Therefore, the current bottleneck is not lack of GLoCA movement. The bottleneck is that the learned patch correction is not clearly better aligned with disease-label clusters.

## Attention Heatmap Analysis

Qualitative inspection of GLoCA attention maps after the 200-epoch ProPos run shows that the scalar entropy diagnostics should be interpreted with caution. The final attention entropy is still relatively high compared with a sharply peaked one-patch distribution, but this does not mean that GLoCA is never paying attention to meaningful disease evidence. The heatmaps show heterogeneous behavior across classes and images.

Observed regimes include:

| Regime | Example class(es) | Observation | Interpretation |
|---|---|---|---|
| Disease-localizing attention with cohesive clustering | Apple mosaic virus | Attention often overlaps visible mosaic/chlorotic disease patterns, and inspected samples are assigned to the same cluster. | This is the best-case behavior: GLoCA can attend to disease evidence and the final embedding can preserve that disease-centered grouping. |
| Disease-localizing attention with fragmented clustering | Apple black rot, blueberry rust | Attention can highlight plausible lesion or rust regions, but inspected samples are split across multiple clusters. | Local disease evidence is detected, but ProPos/GLoCA embeddings are still organized by additional factors such as host appearance, object layout, background, source domain, disease severity, or global visual style. |
| Weakly localized or mislocalized attention with cohesive clustering | Carrot cavity spot | Attention often falls outside the visible lesion region, yet inspected samples are assigned to the same cluster. | Cohesive clusters can arise from global object, crop, source, or host cues even when the explicit attention map is not lesion-centered. |
| Artifact/background attention with mostly cohesive clustering | Banana anthracnose | Attention often focuses on external elements such as borders, backgrounds, slide/text artifacts, or non-disease regions; most inspected samples still fall into the dominant class cluster with at least one exception. | Some PlantWild subsets may be clusterable from non-disease cues. This is useful clustering signal for the unsupervised objective but weak evidence of disease-centered reasoning. |

The main conclusion is that attention localization is neither necessary nor sufficient for disease-label cluster alignment. It is not necessary because some samples cluster together even when attention is not lesion-localizing. It is not sufficient because some samples with visually convincing disease attention still fragment across clusters.

This strengthens, rather than weakens, the objective-alignment diagnosis. GLoCA can sometimes identify local disease evidence, but the ProPos objective does not reliably force that local evidence to dominate the final clustering space. The final embedding is likely influenced by a mixture of disease pattern, host species, object type, image source, background, acquisition style, and global layout. Therefore, high attention entropy should be reported as a coarse concentration statistic, not as proof that GLoCA is ignoring points of interest.

## Preprocessing Ablation

The attention-map inspection raised a plausible preprocessing concern: PlantWild images are heterogeneous, many are non-square, and some contain only a small plant, leaf, or fruit region. Aggressive cropping can potentially remove disease evidence or bias attention toward whichever region survives the transform.

A preliminary preprocessing ablation tested full-frame-preserving alternatives, especially resize-pad and resize-stretch variants. The initial resize-pad ProPos-CLS run did not show improvement over the existing preprocessing pipeline. Around epoch 21 of a 75-epoch resize-pad run, the observed metrics were approximately:

```text
NMI 0.6199
ARI 0.2871
ACC 0.3729
```

This is below the useful early-to-mid training behavior of the existing ProPos-CLS preprocessing path. The resize-stretch variant also did not appear better. Since most PlantWild aspect ratios inspected so far remain within a relatively safe range rather than being extreme, the existing crop/resize pipeline appears to act as a useful zoom prior: it may occasionally discard off-center evidence, but it also increases the effective scale of the plant and disease regions within DINOv2's 224x224 input.

The preprocessing conclusion is therefore not that cropping is always correct, nor that padding is always wrong. The better framing is a scale-versus-field-of-view tradeoff:

```text
Current crop/resize pipeline:
  + increases effective plant/lesion scale
  + avoids large padded regions
  - may discard off-center disease evidence in some images

Resize-pad / full-frame preservation:
  + preserves the whole original image
  - shrinks the plant/lesion region at 224x224
  - may introduce border/padding cues
  - did not improve early ProPos-CLS metrics in the preliminary ablation

Resize-stretch:
  + preserves full-frame inclusion without padding
  - distorts geometry/aspect ratio
  - did not show clear improvement
```

For the current project stage, the existing preprocessing pipeline should remain the default. Preprocessing should be kept as a caveat for qualitative attention interpretation, but it should not displace the next method direction. More advanced alternatives, such as conservative random-resized crop schedules, crop ensembling, foreground-aware cropping, or higher-resolution DINOv2 inputs, can be deferred until there is stronger evidence that preprocessing is the dominant bottleneck.

## PlantWild Baseline Comparison

This comparison is useful for method validation, but it is not the primary GLoCA evidence because it compares different clustering methods. The primary GLoCA evidence remains CLS-ProPos vs. GLoCA-ProPos under the same seed and method.

Current PlantWild baseline aggregate results across seeds `42 69 67`:

| Method | ARI mean | NMI mean | ACC mean | Silhouette mean |
|---|---:|---:|---:|---:|
| K-Means CLS | 0.282549 | 0.650265 | 0.378366 | 0.059500 |
| Spherical K-Means CLS | 0.282549 | 0.650265 | 0.378366 | 0.059500 |
| K-Means GLoCA-gated untrained | 0.275219 | 0.642475 | 0.379265 | 0.057952 |
| IDEC CLS | 0.260701 | 0.632378 | 0.360260 | 0.169330 |
| DEC CLS | 0.260629 | 0.632366 | 0.360260 | 0.169511 |

Seed-42 ProPos final values are above the current aggregate ARI/ACC baseline frontier:

| Run | Seed(s) | ARI | NMI | ACC | Silhouette |
|---|---:|---:|---:|---:|---:|
| ProPos GLoCA-gated, 200 ep | 42 | 0.326628 | 0.654514 | 0.417653 | 0.132384 |
| ProPos CLS, 200 ep | 42 | 0.326506 | 0.666694 | 0.400853 | 0.114619 |
| Best aggregate baseline by ARI: K-Means CLS | 42,69,67 | 0.282549 | 0.650265 | 0.378366 | 0.059500 |
| Best aggregate baseline by ACC: K-Means GLoCA untrained | 42,69,67 | 0.275219 | 0.642475 | 0.379265 | 0.057952 |

This validates ProPos as a meaningful implemented method on PlantWild. However, because the ProPos rows are seed-42 only and the baseline rows are three-seed aggregates, the next reporting step should avoid overstating cross-method differences until ProPos has matching multi-seed runs.

## Main Interpretation

ProPos is now an implemented and working method. It should be treated as a validated baseline/diagnostic method rather than as an unfinished port.

GLoCA-gated + ProPos is not broken:

- It reaches comparable or slightly better peak ARI/ACC than CLS-ProPos.
- It does not catastrophically degrade final metrics.
- It changes the embedding substantially.
- It optimizes the ProPos objective better in final total-loss and PSA-loss terms.
- Its learned patch correction is not clearly better aligned with ground-truth disease labels.

The central bottleneck is objective alignment. Patch-token information appears useful for the ProPos optimization objective, but that objective does not reliably convert the extra local information into disease-label-aligned clusters.

The attention-map inspection makes this interpretation more precise. GLoCA is not merely attending randomly or failing to focus. It sometimes attends to disease evidence convincingly, but those cases are not guaranteed to form disease-consistent clusters. Conversely, some cohesive clusters appear even when attention is not lesion-centered. This means the attention map should be treated as an interpretability diagnostic, not as a direct proxy for clustering correctness.

The preprocessing ablation also reduces the likelihood that the immediate next step should be a broad preprocessing rewrite. Full-frame-preserving resize-pad and resize-stretch variants did not improve the ProPos-CLS trajectory in the preliminary test. The existing preprocessing pipeline remains a defensible default, although cropping artifacts should still be considered when interpreting individual attention maps.

## Recommended Next Direction

The next direction should be:

1. Freeze ProPos as an implemented baseline/diagnostic method.
2. Report ProPos findings as evidence that the repository can run a live two-view trainable method and that GLoCA can learn under such a method.
3. Do not prioritize more ProPos epochs.
4. Keep the current preprocessing pipeline as the default for now; document resize-pad and resize-stretch as non-improving preliminary ablations rather than promoting them.
5. Move to CDC as the next main method.
6. Use CDC confidence calibration and reliable sample selection to test whether better pseudo-label reliability improves disease-label alignment.
7. Keep attention-map inspection as a qualitative diagnostic, with the explicit caveat that attention localization and cluster alignment are related but not equivalent.

The core hypothesis for CDC is now sharper:

```text
GLoCA can inject patch-token information.
ProPos can optimize that information, but not reliably align it to disease labels.
CDC should test whether calibrated reliable-sample selection can make the training signal more disease-aligned.
```
