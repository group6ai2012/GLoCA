"""Generate the consolidated result report from completed local artifacts.

Run with the project environment, for example:

    conda run -n XDC python scripts/generate_full_result_report.py
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "full_result_report.md"
REPORTABLE_DATASETS = ("plantseg", "plantvillage")
BASELINE_SEEDS = (42, 67, 69)
CDC_SEEDS = (42, 67, 69, 99, 123)
ABLATION_SEEDS = (42, 67, 69)
METRICS = ("ari", "nmi", "acc", "silhouette")
CDC_DIAGNOSTIC_COLUMNS = (
    "n_nonempty_clusters",
    "cluster_size_min",
    "cluster_size_max",
    "cluster_size_entropy",
    "calibration_ece",
    "calibration_mce",
    "reliable_sample_ratio",
    "pseudo_label_count",
    "pseudo_label_entropy",
    "clustering_confidence_mean",
    "calibrated_confidence_mean",
)


def load_metric_rows(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(root.rglob("metrics.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        row["_path"] = path.relative_to(ROOT).as_posix()
        rows.append(row)
    return pd.DataFrame(rows)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def clean_seed(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["seed"] = numeric(frame, "seed").astype(int)
    frame["dataset"] = frame["dataset"].astype(str)
    return frame


def baseline_method(path: str) -> str:
    if "spherical_kmeans_cls" in path:
        return "Spherical K-Means"
    if "kmeans_gloca_gated_untrained" in path:
        return "Spherical K-Means + untrained GLoCA"
    if "kmeans_cls" in path:
        return "K-Means"
    if "idec_cls" in path:
        return "IDEC"
    if "dec_cls" in path:
        return "DEC"
    return "Baseline"


def compact(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def signed(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    number = float(value)
    return f"{number:+.{digits}f}"


def mean_sd(values: Iterable[object], digits: int = 3) -> str:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return "—"
    std = series.std(ddof=1) if len(series) > 1 else 0.0
    return f"{series.mean():.{digits}f} ± {std:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def metric_table(frame: pd.DataFrame, include_gloca: bool = False) -> str:
    rows: list[list[object]] = []
    for _, row in frame.sort_values(["dataset", "seed"]).iterrows():
        cells: list[object] = [row["dataset"], int(row["seed"])]
        if include_gloca:
            cells.append(row["embedding"])
        cells.extend(compact(row.get(metric)) for metric in METRICS)
        rows.append(cells)
    headers = ["Dataset", "Seed"] + (["Embedding"] if include_gloca else []) + [m.upper() for m in METRICS]
    return markdown_table(headers, rows)


def diagnostic_seed_table(frame: pd.DataFrame, label_column: str) -> tuple[str, str]:
    occupancy_rows: list[list[object]] = []
    calibration_rows: list[list[object]] = []
    for _, row in frame.sort_values(["dataset", "seed", label_column]).iterrows():
        prefix = [row["dataset"], int(row["seed"]), row[label_column]]
        occupancy_rows.append(
            prefix
            + [
                compact(row.get("n_nonempty_clusters"), 0),
                compact(row.get("cluster_size_min"), 0),
                compact(row.get("cluster_size_max"), 0),
                compact(row.get("cluster_size_entropy")),
            ]
        )
        calibration_rows.append(
            prefix
            + [
                compact(row.get("calibration_ece")),
                compact(row.get("calibration_mce")),
                compact(row.get("reliable_sample_ratio")),
                compact(row.get("pseudo_label_count"), 0),
                compact(row.get("pseudo_label_entropy")),
                compact(row.get("clustering_confidence_mean")),
                compact(row.get("calibrated_confidence_mean")),
            ]
        )
    label = label_column.title()
    occupancy = markdown_table(
        ["Dataset", "Seed", label, "Nonempty clusters", "Min cluster", "Max cluster", "Cluster entropy"],
        occupancy_rows,
    )
    calibration = markdown_table(
        [
            "Dataset",
            "Seed",
            label,
            "ECE",
            "MCE",
            "Reliable ratio",
            "Pseudo labels",
            "Pseudo-label entropy",
            "Clustering confidence",
            "Calibrated confidence",
        ],
        calibration_rows,
    )
    return occupancy, calibration


def diagnostic_aggregate_tables(frame: pd.DataFrame, group_columns: list[str], label_column: str) -> tuple[str, str]:
    occupancy_rows: list[list[object]] = []
    calibration_rows: list[list[object]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        prefix = list(keys)
        seeds = ", ".join(str(seed) for seed in sorted(group["seed"].astype(int).unique()))
        occupancy_rows.append(
            prefix
            + [
                mean_sd(group["n_nonempty_clusters"], 1),
                mean_sd(group["cluster_size_min"], 1),
                mean_sd(group["cluster_size_max"], 1),
                mean_sd(group["cluster_size_entropy"]),
                seeds,
            ]
        )
        calibration_rows.append(
            prefix
            + [
                mean_sd(group["calibration_ece"]),
                mean_sd(group["calibration_mce"]),
                mean_sd(group["reliable_sample_ratio"]),
                mean_sd(group["pseudo_label_count"], 1),
                mean_sd(group["pseudo_label_entropy"]),
                mean_sd(group["clustering_confidence_mean"]),
                mean_sd(group["calibrated_confidence_mean"]),
                seeds,
            ]
        )
    occupancy = markdown_table(
        [
            "Dataset",
            label_column.title(),
            "Nonempty clusters mean ± SD",
            "Min cluster mean ± SD",
            "Max cluster mean ± SD",
            "Cluster entropy mean ± SD",
            "Seeds",
        ],
        occupancy_rows,
    )
    calibration = markdown_table(
        [
            "Dataset",
            label_column.title(),
            "ECE mean ± SD",
            "MCE mean ± SD",
            "Reliable ratio mean ± SD",
            "Pseudo labels mean ± SD",
            "Pseudo-label entropy mean ± SD",
            "Clustering confidence mean ± SD",
            "Calibrated confidence mean ± SD",
            "Seeds",
        ],
        calibration_rows,
    )
    return occupancy, calibration


def aggregate_table(frame: pd.DataFrame, group_columns: list[str], label_column: str | None = None) -> str:
    rows: list[list[object]] = []
    sort_columns = group_columns
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        cells: list[object] = list(keys)
        if label_column:
            cells.append(group[label_column].iloc[0])
        cells.extend(mean_sd(group[metric]) for metric in METRICS)
        cells.append(", ".join(str(seed) for seed in sorted(group["seed"].astype(int).unique())))
        rows.append(cells)
    labels = [column.title() for column in sort_columns]
    if label_column:
        labels.append(label_column.title())
    labels += [m.upper() + " mean ± SD" for m in METRICS] + ["Seeds"]
    return markdown_table(labels, rows)


def matched_delta_table(frame: pd.DataFrame, key_columns: list[str]) -> str:
    pivot = frame.pivot_table(index=key_columns, columns="embedding", values=list(METRICS), aggfunc="first")
    rows: list[list[object]] = []
    for keys, values in pivot.sort_index().iterrows():
        if not isinstance(keys, tuple):
            keys = (keys,)
        cells: list[object] = list(keys)
        for metric in METRICS:
            try:
                cells.append(signed(values[(metric, "GLoCA")].item() - values[(metric, "CLS")].item(), 3))
            except KeyError:
                cells.append("—")
        rows.append(cells)
    return markdown_table([column.title() for column in key_columns] + ["Δ " + m.upper() for m in METRICS], rows)


def reference_delta_table(frame: pd.DataFrame, reference: pd.DataFrame, key_columns: list[str]) -> str:
    merged = frame.merge(
        reference[key_columns + list(METRICS)],
        on=key_columns,
        how="inner",
        suffixes=("", "_reference"),
    )
    rows: list[list[object]] = []
    for _, row in merged.sort_values(key_columns + ["variant"]).iterrows():
        rows.append(
            [row["variant"]] + [int(row[column]) for column in key_columns if column == "seed"] + [
                signed(float(row[metric]) - float(row[f"{metric}_reference"]), 3) for metric in METRICS
            ]
        )
    return markdown_table(["Variant"] + [column.title() for column in key_columns if column == "seed"] + ["Δ " + m.upper() for m in METRICS], rows)


def diagnostics_rows(paths: list[Path]) -> list[list[object]]:
    rows: list[list[object]] = []
    for path in paths:
        log_path = path.parent / "logs.json"
        if not log_path.exists():
            continue
        data = json.loads(log_path.read_text(encoding="utf-8"))
        history = data.get("gloca_diagnostics_history") or []
        if not history:
            continue
        first, last = history[0], history[-1]
        parts = path.relative_to(ROOT).parts
        dataset = next((part for part in REPORTABLE_DATASETS if part in parts), "unknown")
        seed = next((part.removeprefix("seed_") for part in parts if part.startswith("seed_")), "?")
        rows.append(
            [
                dataset,
                seed,
                f"{compact(first.get('gloca_alpha_value'))} → {compact(last.get('gloca_alpha_value'))}",
                f"{compact(first.get('gloca_embedding_cls_cosine_mean'))} → {compact(last.get('gloca_embedding_cls_cosine_mean'))}",
                f"{compact(first.get('gloca_delta_norm_mean'), 1)} → {compact(last.get('gloca_delta_norm_mean'), 1)}",
                f"{compact(first.get('gloca_attention_entropy_mean'))} → {compact(last.get('gloca_attention_entropy_mean'))}",
            ]
        )
    return sorted(rows, key=lambda row: (row[0], int(row[1])))


def validate_counts(
    baselines: pd.DataFrame,
    cdc: pd.DataFrame,
    propos_cls: pd.DataFrame,
    propos_gloca: pd.DataFrame,
    ablations: pd.DataFrame,
) -> None:
    assert len(baselines) == 30, f"Expected 30 reportable baseline rows, found {len(baselines)}"
    assert set(baselines["seed"]) == set(BASELINE_SEEDS)
    assert len(cdc) == 20, f"Expected 20 CDC rows, found {len(cdc)}"
    assert set(cdc["seed"]) == set(CDC_SEEDS)
    assert len(propos_cls) == 6, f"Expected 6 ProPos CLS rows, found {len(propos_cls)}"
    assert set(propos_cls["seed"]) == set(BASELINE_SEEDS)
    assert len(propos_gloca) == 2, f"Expected 2 ProPos+GLoCA rows, found {len(propos_gloca)}"
    assert set(propos_gloca["seed"]) == {42}
    assert len(ablations) == 9, f"Expected 9 ablation rows, found {len(ablations)}"
    assert set(ablations["seed"]) == set(ABLATION_SEEDS)


def validate_cached_count(cached: pd.DataFrame) -> None:
    assert len(cached) == 18, f"Expected 18 cached-CLS baseline rows, found {len(cached)}"
    assert set(cached["seed"]) == set(BASELINE_SEEDS)


def build_report() -> str:
    baseline_raw = clean_seed(load_metric_rows(ROOT / "outputs" / "baselines_full"))
    baseline_raw = baseline_raw[baseline_raw["dataset"].isin(REPORTABLE_DATASETS)].copy()
    baseline_raw["method"] = baseline_raw["_path"].map(baseline_method)

    cached = clean_seed(load_metric_rows(ROOT / "outputs" / "cached_cls_baselines"))
    cached = cached[cached["dataset"].isin(REPORTABLE_DATASETS)].copy()
    cached["method"] = cached["head"].map(
        {"gmm": "GMM", "hdbscan": "HDBSCAN", "supervised_mlp": "Supervised MLP"}
    ).fillna(cached["head"])

    cdc = clean_seed(load_metric_rows(ROOT / "outputs" / "main_experiments"))
    cdc = cdc[(cdc["dataset"].isin(REPORTABLE_DATASETS)) & cdc["_path"].str.contains("/cdc/")].copy()
    cdc["embedding"] = cdc["gloca"].map(lambda value: "CLS" if str(value) == "disabled" else "GLoCA")

    propos = clean_seed(load_metric_rows(ROOT / "outputs" / "main_experiments"))
    propos = propos[(propos["dataset"].isin(REPORTABLE_DATASETS)) & propos["_path"].str.contains("/propos/")].copy()
    propos_cls = propos[propos["gloca"].astype(str) == "disabled"].copy()
    propos_cls["method"] = "ProPos (DINO CLS)"
    propos_gloca = propos[propos["gloca"].astype(str) != "disabled"].copy()
    propos_gloca["embedding"] = "GLoCA"
    propos_exploratory = pd.concat(
        [propos_cls[propos_cls["seed"] == 42].assign(embedding="CLS"), propos_gloca],
        ignore_index=True,
    )

    ablations = clean_seed(load_metric_rows(ROOT / "outputs" / "ablation_studies"))
    ablations = ablations[ablations["dataset"].isin(REPORTABLE_DATASETS)].copy()
    ablations["variant"] = ablations["_path"].str.split("/").str[2]
    variant_labels = {
        "gloca_mean_pool": "Mean-pool GLoCA",
        "gloca_no_patches": "No-patches GLoCA",
        "gloca_no_cls_residual": "No-CLS-residual GLoCA",
    }
    ablations["variant"] = ablations["variant"].map(variant_labels).fillna(ablations["variant"])

    validate_counts(baseline_raw, cdc, propos_cls, propos_gloca, ablations)
    validate_cached_count(cached)

    baseline_agg = aggregate_table(baseline_raw, ["dataset", "method"])
    baseline_per_seed = metric_table(baseline_raw.rename(columns={"method": "embedding"}), include_gloca=True)

    cdc_agg = aggregate_table(cdc, ["dataset", "embedding"])
    cdc_per_seed = metric_table(cdc, include_gloca=True)
    cdc_deltas = matched_delta_table(cdc, ["dataset", "seed"])
    cdc_occupancy, cdc_calibration = diagnostic_seed_table(cdc, "embedding")
    cdc_occupancy_agg, cdc_calibration_agg = diagnostic_aggregate_tables(cdc, ["dataset", "embedding"], "embedding")

    propos_cls_agg = aggregate_table(propos_cls, ["dataset", "method"])
    propos_cls_per_seed = metric_table(propos_cls.rename(columns={"method": "embedding"}), include_gloca=True)
    propos_exploratory_per_seed = metric_table(propos_exploratory, include_gloca=True)
    propos_exploratory_deltas = matched_delta_table(propos_exploratory, ["dataset", "seed"])

    ablation_reference = cdc[
        (cdc["dataset"] == "plantseg")
        & (cdc["embedding"] == "GLoCA")
        & cdc["seed"].isin(ABLATION_SEEDS)
    ].copy()
    ablation_reference["variant"] = "Gated GLoCA (reference)"
    comparison_columns = ["dataset", "seed", "variant"] + list(METRICS) + list(CDC_DIAGNOSTIC_COLUMNS)
    ablation_comparison = pd.concat(
        [ablation_reference[comparison_columns], ablations[comparison_columns]],
        ignore_index=True,
    )
    ablation_agg = aggregate_table(ablation_comparison, ["dataset", "variant"])
    ablation_per_seed = metric_table(ablation_comparison.rename(columns={"variant": "embedding"}), include_gloca=True)
    ablation_deltas = reference_delta_table(ablations, ablation_reference, ["dataset", "seed"])
    ablation_occupancy, ablation_calibration = diagnostic_seed_table(ablation_comparison, "variant")
    ablation_occupancy_agg, ablation_calibration_agg = diagnostic_aggregate_tables(
        ablation_comparison,
        ["dataset", "variant"],
        "variant",
    )

    baseline_untrained = baseline_raw[baseline_raw["method"].isin(["Spherical K-Means", "Spherical K-Means + untrained GLoCA"])]
    baseline_untrained = baseline_untrained.copy()
    baseline_untrained["embedding"] = baseline_untrained["method"].map(
        {"Spherical K-Means": "CLS", "Spherical K-Means + untrained GLoCA": "GLoCA"}
    )
    baseline_untrained_deltas = matched_delta_table(baseline_untrained, ["dataset", "seed"])

    cached_agg = aggregate_table(cached, ["dataset", "method"])
    cached_per_seed = metric_table(cached.rename(columns={"method": "embedding"}), include_gloca=True)

    diagnostic_paths = [
        ROOT / path
        for path in cdc[cdc["embedding"] == "GLoCA"]["_path"].tolist()
    ]
    diagnostic_table = markdown_table(
        ["Dataset", "Seed", "Alpha first → last", "CLS cosine first → last", "Delta norm first → last", "Attention entropy first → last"],
        diagnostics_rows(diagnostic_paths),
    )

    lines = [
        "# Full Result Report",
        "",
        f"Generated from completed local artifacts on {date.today().isoformat()} using the `XDC` conda environment.",
        "",
        "## Executive summary",
        "",
        "This report consolidates the complete reportable experiment set currently present in the repository:",
        "",
        markdown_table(
            ["Experiment family", "Datasets", "Seeds", "Completed result files", "Interpretation status"],
            [
                ["Baselines (standard + cached-CLS + ProPos CLS)", "PlantSeg, PlantVillage", "42, 67, 69", "54", "Full three-seed baseline sweep; 9 baseline heads"],
                ["CDC", "PlantSeg, PlantVillage", "42, 67, 69, 99, 123", "20", "Full five-seed matched CLS/GLoCA comparison"],
                ["ProPos + GLoCA", "PlantSeg, PlantVillage", "42", "2", "Exploratory matched result; not primary-paper evidence"],
                ["GLoCA ablations", "PlantSeg", "42, 67, 69", "9", "Full three-seed ablation comparison"],
            ],
        ),
        "",
        "The controlled comparison is direct DINO CLS versus a GLoCA embedding under the same downstream clustering method. DINOv2 is frozen. PlantWild artifacts are excluded from the primary report because the project current-state document treats PlantWild as smoke/debug context and notes its near-duplicate relationship with PlantSeg.",
        "",
        "The strongest result is CDC: GLoCA improves the five-seed mean for all four metrics on both PlantSeg and PlantVillage. ProPos with direct DINO CLS is now a completed three-seed baseline; the seed-42 ProPos + GLoCA comparison remains exploratory and is not primary-paper evidence. The ablations test which parts of the gated adapter matter on PlantSeg.",
        "",
        "## Evaluation and reporting conventions",
        "",
        "Metrics are read from each run's canonical `metrics.csv`: adjusted Rand index (ARI), normalized mutual information (NMI), Hungarian-aligned clustering accuracy (ACC), and silhouette. Aggregate values are mean ± sample standard deviation across the listed seeds. `metrics.csv` is treated as the final exported result; checkpoint histories are not substituted for final metrics.",
        "",
        "Attention maps and scalar diagnostics are supporting diagnostics, not proof of disease localization or clustering correctness. In particular, lower attention entropy means more concentrated attention, not necessarily better disease evidence.",
        "",
        "## 1. Three-seed baselines",
        "",
        "The three-seed baseline group contains DEC, IDEC, K-Means, spherical K-Means, spherical K-Means with untrained `gloca_gated`, GMM, HDBSCAN, a supervised MLP reference, and ProPos with direct DINO CLS. All are evaluated on PlantSeg and PlantVillage with seeds 42, 67, and 69.",
        "",
        "### Aggregate baseline results",
        "",
        baseline_agg,
        "",
        "### Complete per-seed baseline results",
        "",
        baseline_per_seed,
        "",
        "### CLS versus untrained GLoCA spherical K-Means",
        "",
        "The untrained adapter is intended to preserve the normalized CLS path at initialization. The matched deltas below are correspondingly small.",
        "",
        baseline_untrained_deltas,
        "",
        "### Cached-CLS baseline extensions",
        "",
        "The repository also contains three cached-CLS baselines evaluated with the same three seeds on PlantSeg and PlantVillage. GMM and HDBSCAN are unsupervised references. The supervised MLP uses labels and is therefore reported only as a separate supervised reference, not as primary evidence for unsupervised clustering.",
        "",
        "### Aggregate cached-CLS results",
        "",
        cached_agg,
        "",
        "### Complete per-seed cached-CLS results",
        "",
        cached_per_seed,
        "",
        "### ProPos with direct DINO CLS",
        "",
        "ProPos with `gloca: disabled` is included as a three-seed trainable clustering baseline. Its results are distinct from the seed-42 ProPos + GLoCA exploration reported later; only the direct-CLS ProPos configuration is designated as baseline evidence.",
        "",
        "#### Aggregate ProPos CLS baseline results",
        "",
        propos_cls_agg,
        "",
        "#### Complete per-seed ProPos CLS baseline results",
        "",
        propos_cls_per_seed,
        "",
        "## 2. Five-seed CDC results",
        "",
        "CDC is the primary multi-seed trainable-method result. Each reportable dataset has matched CLS and GLoCA runs for seeds 42, 67, 69, 99, and 123.",
        "",
        "### Aggregate CDC results",
        "",
        cdc_agg,
        "",
        "### Complete per-seed CDC final metrics",
        "",
        cdc_per_seed,
        "",
        "### Matched GLoCA minus CLS deltas",
        "",
        cdc_deltas,
        "",
        "On PlantSeg, the GLoCA delta is positive for every metric on every seed. On PlantVillage, NMI, ACC, and silhouette are positive on every seed; ARI is positive on four of five seeds and slightly negative on seed 67. This supports a cautious positive CDC claim without implying that every metric improves on every seed.",
        "",
        "### CDC occupancy",
        "",
        "Cluster occupancy is reported from the final exported assignments. Higher normalized cluster-size entropy indicates a more even cluster-size distribution; it is not a ground-truth clustering metric on its own.",
        "",
        "#### Aggregate occupancy",
        "",
        cdc_occupancy_agg,
        "",
        "#### Per-seed occupancy",
        "",
        cdc_occupancy,
        "",
        "Across both datasets, GLoCA generally maintains or increases normalized cluster-size entropy. On PlantVillage, the minimum occupied-cluster size rises sharply under GLoCA, indicating that the final assignments avoid the singleton or near-singleton occupied clusters seen in several CLS runs. These occupancy changes are descriptive and must be read with ARI, NMI, and ACC rather than as evidence of correctness by themselves.",
        "",
        "### CDC calibration and reliable-sample coverage",
        "",
        "CDC's calibration fields are evaluated from the final exported calibration head. Reliable ratio and pseudo-label count quantify coverage of samples selected as reliable during training; ECE and MCE quantify calibration error, where lower is better.",
        "",
        "#### Aggregate calibration and coverage",
        "",
        cdc_calibration_agg,
        "",
        "#### Per-seed calibration and coverage",
        "",
        cdc_calibration,
        "",
        "GLoCA increases reliable-sample coverage and pseudo-label count on every matched main CDC seed. Calibration error is not uniformly improved: PlantSeg has mixed ECE/MCE outcomes, while PlantVillage ECE and MCE are consistently higher with GLoCA. The supported claim is therefore improved CDC clustering with greater reliable-sample coverage, not universally better calibration.",
        "",
        "## 3. ProPos + GLoCA: seed-42 exploratory comparison",
        "",
        "The three-seed ProPos CLS baseline is reported in Section 1. This section retains the two matched seed-42 ProPos + GLoCA artifacts for transparency, but they are exploratory diagnostics rather than primary final-paper evidence because no matched GLoCA seed-67 or seed-69 runs are available.",
        "",
        "### Seed-42 matched final metrics",
        "",
        propos_exploratory_per_seed,
        "",
        "### Matched GLoCA minus CLS deltas",
        "",
        propos_exploratory_deltas,
        "",
        "For the available seed, GLoCA lowers ARI, NMI, and ACC on both datasets while changing silhouette in opposite directions: it decreases on PlantSeg and increases slightly on PlantVillage. This remains useful method-development context, but is not carried forward as a primary final-paper result.",
        "",
        "## 4. Three-seed GLoCA ablations",
        "",
        "The ablation study uses CDC on PlantSeg with seeds 42, 67, and 69. Each variant changes one structural part of `gloca_gated` while keeping the dataset, seeds, clustering method, and evaluation metrics fixed:",
        "",
        "- Mean-pool GLoCA replaces gated attention pooling with mean pooling.",
        "- No-patches GLoCA removes the patch branch.",
        "- No-CLS-residual GLoCA removes the residual CLS output path.",
        "",
        "### Aggregate ablation results",
        "",
        ablation_agg,
        "",
        "### Complete per-seed ablation final metrics",
        "",
        ablation_per_seed,
        "",
        "### Ablation minus matched gated-GLoCA reference",
        "",
        ablation_deltas,
        "",
        "The ablation aggregates and deltas are controlled against the gated-GLoCA reference on the same three seeds. They are not independent clustering methods; they isolate adapter design choices within the same CDC composition.",
        "",
        "### Ablation occupancy",
        "",
        "The tables below compare final CDC assignment occupancy for the three adapter variants against the matched gated-GLoCA reference on PlantSeg seeds 42, 67, and 69.",
        "",
        "#### Aggregate occupancy",
        "",
        ablation_occupancy_agg,
        "",
        "#### Per-seed occupancy",
        "",
        ablation_occupancy,
        "",
        "No-CLS-residual GLoCA has the highest occupancy entropy and avoids singleton clusters on seed 42, consistent with its stronger ARI/ACC and silhouette in this three-seed ablation. Mean-pool and no-patches variants have lower entropy or more small occupied clusters than the gated reference on several seeds; those patterns are descriptive rather than standalone quality rankings.",
        "",
        "### Ablation calibration and reliable-sample coverage",
        "",
        "These calibration fields use the same CDC head and final-output convention as the main CDC section, so they can be compared directly within the matched three-seed PlantSeg ablation subset.",
        "",
        "#### Aggregate calibration and coverage",
        "",
        ablation_calibration_agg,
        "",
        "#### Per-seed calibration and coverage",
        "",
        ablation_calibration,
        "",
        "No-CLS-residual GLoCA selects the most reliable samples and has the largest pseudo-label count, but it also has the highest ECE and MCE. Mean-pool and no-patches variants retain lower calibration error than the gated reference on some seeds, while generally selecting fewer reliable samples. The ablations reinforce the main result: reliable-sample coverage, calibration error, and disease-label clustering are related but not interchangeable objectives.",
        "",
        "## 5. GLoCA learning diagnostics",
        "",
        "The final CDC GLoCA logs include nontrivial parameter movement across all ten runs. The table reports first-to-last values from `gloca_diagnostics_history`.",
        "",
        diagnostic_table,
        "",
        "Alpha movement, nonzero delta norms, and falling CLS cosine show that GLoCA changes the representation during CDC training. The positive CDC results therefore should not be attributed to a permanently inactive adapter. Conversely, representation movement alone is not sufficient: the seed-42 ProPos results show that movement can fail to improve disease-label alignment.",
        "",
        "## 6. Overall interpretation",
        "",
        "The complete artifact set supports the following bounded conclusions:",
        "",
        "1. Untrained `gloca_gated` is nearly equivalent to the direct CLS spherical K-Means baseline, supporting the conservative initialization contract.",
        "2. With CDC held fixed, trained GLoCA improves the five-seed average of ARI, NMI, ACC, and silhouette on PlantSeg, and improves the five-seed average of all four metrics on PlantVillage.",
        "3. ProPos with direct DINO CLS is a completed three-seed baseline. The available seed-42 ProPos + GLoCA comparison is negative for label-aligned metrics on both datasets and remains exploratory rather than primary-paper evidence.",
        "4. The three-seed PlantSeg ablations provide structural evidence about the gated adapter, but they do not replace the matched CDC CLS baseline or establish a universal best adapter without considering variance and seed coverage.",
        "5. Attention concentration and GLoCA scalar movement are diagnostics. They should be interpreted jointly with ARI, NMI, ACC, cluster occupancy, and the matched baseline.",
        "",
        "## Limitations and next steps",
        "",
        "- ProPos CLS is available across three seeds, but ProPos + GLoCA remains seed-42-only; seed 67 and seed 69 GLoCA runs would be required for a multi-seed ProPos + GLoCA claim.",
        "- The main CDC comparison uses five seeds, while the baseline and ablation families use three seeds; standard deviations are therefore not directly comparable across families as uncertainty estimates.",
        "- PlantWild is omitted from primary evidence by project policy.",
        "- CDC calibration-quality fields should not be interpreted as uniformly improved merely because clustering metrics improve.",
        "- Class-level confusion, cluster composition, and paired qualitative attention review remain useful follow-up analyses.",
        "",
        "## Artifact sources",
        "",
        markdown_table(
            ["Family", "Artifact root", "Canonical file"],
            [
                ["Baselines", "`outputs/baselines_full/`", "`metrics.csv` plus `baseline_summary*.csv`"],
                ["Cached-CLS baselines", "`outputs/cached_cls_baselines/`", "`metrics.csv` plus `cached_cls_baseline_summary*.csv`"],
                ["CDC", "`outputs/main_experiments/{base,gloca_gated}/cdc/`", "`metrics.csv`, `logs.json`, `assignments.json`"],
                ["ProPos CLS baseline / ProPos + GLoCA exploratory", "`outputs/main_experiments/{base,gloca_gated}/propos/`", "`metrics.csv`, `logs.json`, `checkpoint.ckpt`"],
                ["Ablations", "`outputs/ablation_studies/`", "`metrics.csv`, `logs.json`, `assignments.json`"],
            ],
        ),
        "",
        "The report is reproducible with `conda run -n XDC python scripts/generate_full_result_report.py`.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
