from __future__ import annotations

import time
import warnings
from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F

from src.data import ClusteringDataModule, FolderImageDataset, get_dataset_spec
from src.data.transforms import build_eval_transform
from src.evaluation import compute_clustering_metrics
from src.experiments.config import save_experiment_config
from src.experiments.outputs import write_outputs
from src.features import DINOv2Backbone
from src.models import ClusteringBaseModel, build_adapter
from src.models.clustering.kmeans import fit_kmeans
from src.models.clustering import StudentTHead
from src.runners.common import apply_runtime_settings
from src.runners.diagnostics import attention_diagnostics, cluster_diagnostics, embedding_diagnostics
from src.utils import ExperimentResult, resolve_device, resolve_gloca_name, resolve_output_dir, seed_everything


def run_student_t(config: dict[str, Any]) -> ExperimentResult:
    config = deepcopy(config)
    runtime_logs = apply_runtime_settings(config)
    if config["head"]["name"] != "student_t":
        raise ValueError(f"run_student_t only supports head.name='student_t', got {config['head']['name']!r}")

    spec = get_dataset_spec(config["dataset"])
    eval_dataset = FolderImageDataset(
        spec,
        transform=build_eval_transform(int(config["backbone"]["image_size"])),
        training_views=1,
    )
    if config["head"]["n_clusters"] == "auto":
        config["head"]["n_clusters"] = eval_dataset.n_classes
    if config["dataset"]["training_views"] == "auto":
        config["dataset"]["training_views"] = 1
    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, output_dir / "config.yaml")

    seed_everything(int(config["experiment"]["seed"]))
    datamodule = ClusteringDataModule(
        spec=spec,
        image_size=int(config["backbone"]["image_size"]),
        batch_size=int(config["trainer"]["batch_size"]),
        num_workers=int(config["trainer"]["num_workers"]),
        training_mode="single_view",
        training_views=config["dataset"]["training_views"],
    )

    backbone = DINOv2Backbone(
        variant=config["backbone"]["variant"],
        image_size=int(config["backbone"]["image_size"]),
        freeze=bool(config["backbone"].get("freeze", True)),
    )
    if any(param.requires_grad for param in backbone.parameters()):
        raise RuntimeError("All DINOv2 parameters must have requires_grad=False")

    adapter = build_adapter(config, input_dim=backbone.output_dim)
    gloca_name = resolve_gloca_name(config)
    embedding_dim = backbone.output_dim
    if adapter is not None:
        embedding_dim = int(config["gloca"]["embedding_dim"])

    head = StudentTHead(
        n_clusters=int(config["head"]["n_clusters"]),
        embedding_dim=embedding_dim,
        seed=int(config["experiment"]["seed"]),
    )
    model = ClusteringBaseModel(
        backbone=backbone,
        adapter=adapter,
        head=head,
        normalize_cls=bool(config["gloca"].get("normalize_output", True)),
    )
    train_start = time.perf_counter()
    device = resolve_device(config)
    model.to(device)

    cache_start = time.perf_counter()
    cache = _precompute_backbone_cache(model, datamodule, device, needs_patches=model.adapter is not None)
    backbone_cache_time_s = time.perf_counter() - cache_start

    head_train_start = time.perf_counter()
    _initialize_student_t_centers(model, cache, device, seed=int(config["experiment"]["seed"]))
    _train_student_t_from_cache(model, cache, config, device)
    head_train_time_s = time.perf_counter() - head_train_start
    total_train_time_s = time.perf_counter() - train_start
    torch.save({"state_dict": model.state_dict(), "config": config}, output_dir / "checkpoint.ckpt")

    infer_start = time.perf_counter()
    assembled = _predict_from_cache(model, cache, device)
    inference_time_s = time.perf_counter() - infer_start

    metrics = compute_clustering_metrics(
        assembled["labels"].numpy(),
        assembled["assignments"].numpy(),
        assembled["embeddings"].numpy(),
    )
    cluster_stats = cluster_diagnostics(assembled["assignments"].numpy(), int(config["head"]["n_clusters"]))
    embedding_stats = embedding_diagnostics(assembled["embeddings"])
    attention_stats = attention_diagnostics(assembled["attention"])
    peak_gpu_mb = 0.0
    if torch.cuda.is_available():
        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    assignments_payload = {
        "head": "student_t",
        "backbone": config["backbone"]["variant"],
        "gloca": gloca_name,
        "dataset": spec.name,
        "seed": int(config["experiment"]["seed"]),
        "n_clusters": int(config["head"]["n_clusters"]),
        "image_ids": assembled["image_ids"],
        "labels": assembled["labels"].tolist(),
        "assignments": assembled["assignments"].tolist(),
        "patch_grid": list(assembled["patch_grid"]),
    }
    metrics_row = {
        "experiment": config["experiment"]["name"],
        "head": "student_t",
        "backbone": config["backbone"]["variant"],
        "dataset": spec.name,
        "seed": int(config["experiment"]["seed"]),
        "gloca": gloca_name,
        "n_clusters": int(config["head"]["n_clusters"]),
        "n_images": int(assembled["embeddings"].shape[0]),
        "ari": metrics["ari"],
        "nmi": metrics["nmi"],
        "acc": metrics["acc"],
        "silhouette": metrics["silhouette"],
        **cluster_stats,
        **embedding_stats,
        **attention_stats,
        "total_time_s": total_train_time_s + inference_time_s,
        "backbone_cache_time_s": backbone_cache_time_s,
        "head_train_time_s": head_train_time_s,
        "inference_time_s": inference_time_s,
        "peak_gpu_mb": peak_gpu_mb,
        "uses_cached_backbone_features": True,
    }
    logs = {
        "uses_cached_backbone_features": True,
        "backbone_requires_grad_false": not any(param.requires_grad for param in backbone.parameters()),
        "backbone_cache_time_s": backbone_cache_time_s,
        "head_train_time_s": head_train_time_s,
        "total_train_time_s": total_train_time_s,
        "inference_time_s": inference_time_s,
        "head": "student_t",
        "backbone": config["backbone"]["variant"],
        "gloca": gloca_name,
        "embedding_shape": list(assembled["embeddings"].shape),
        "attention_shape": None if assembled["attention"] is None else list(assembled["attention"].shape),
        "num_workers": int(config["trainer"]["num_workers"]),
        **runtime_logs,
        **cluster_stats,
        **embedding_stats,
        **attention_stats,
    }
    write_outputs(
        output_dir=output_dir,
        config=config,
        assignments_payload=assignments_payload,
        metrics_row=metrics_row,
        embeddings=assembled["embeddings"],
        attention=assembled["attention"],
        logs=logs,
    )
    return ExperimentResult(output_dir=str(output_dir), metrics=metrics_row)


def run_dec(config: dict[str, Any]) -> ExperimentResult:
    warnings.warn(
        "run_dec is deprecated for the centroid-only smoke head; use run_student_t instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    config = deepcopy(config)
    if config.get("head", {}).get("name") == "dec":
        config["head"]["name"] = "student_t"
    return run_student_t(config)

def _precompute_backbone_cache(
    model: ClusteringBaseModel,
    datamodule: ClusteringDataModule,
    device: torch.device,
    needs_patches: bool,
) -> dict[str, Any]:
    model.backbone.to(device)
    model.backbone.eval()
    cls_parts: list[torch.Tensor] = []
    patch_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []
    image_ids: list[str] = []
    patch_grid = None
    with torch.no_grad():
        for batch in datamodule.predict_dataloader():
            out = model.backbone(batch["image"].to(device))
            cls_parts.append(out["cls"].detach().cpu())
            if needs_patches:
                patch_parts.append(out["patch_tokens"].detach().cpu().half())
            label_parts.append(batch["label"].detach().cpu())
            index_parts.append(batch["index"].detach().cpu())
            image_ids.extend(batch["image_id"])
            patch_grid = out["patch_grid"]

    indices = torch.cat(index_parts, dim=0).long()
    order = torch.argsort(indices)
    ordered_ids = [image_ids[i] for i in order.tolist()]
    cache = {
        "cls": torch.cat(cls_parts, dim=0)[order].contiguous(),
        "patch_tokens": torch.cat(patch_parts, dim=0)[order].contiguous() if patch_parts else None,
        "labels": torch.cat(label_parts, dim=0)[order].long().contiguous(),
        "indices": indices[order].contiguous(),
        "image_ids": ordered_ids,
        "patch_grid": patch_grid,
    }
    return cache


def _cached_embeddings(
    model: ClusteringBaseModel,
    cache: dict[str, Any],
    rows: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    cls = cache["cls"][rows].to(device)
    if model.adapter is None:
        embedding = F.normalize(cls, dim=-1) if model.normalize_cls else cls
        return {"embedding": embedding, "attention": None}
    patch_tokens = cache["patch_tokens"][rows].to(device).float()
    out = model.adapter(cls=cls, patch_tokens=patch_tokens, patch_grid=cache["patch_grid"])
    return {"embedding": out["embedding"], "attention": out["attention"]}


def _all_cached_embeddings(
    model: ClusteringBaseModel,
    cache: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    embeddings: list[torch.Tensor] = []
    attentions: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, cache["cls"].shape[0], batch_size):
            rows = torch.arange(start, min(start + batch_size, cache["cls"].shape[0]))
            encoded = _cached_embeddings(model, cache, rows, device)
            embeddings.append(encoded["embedding"].detach().cpu())
            if encoded["attention"] is not None:
                attentions.append(encoded["attention"].detach().cpu())
    attention = torch.cat(attentions, dim=0) if attentions else None
    return torch.cat(embeddings, dim=0), attention


def _initialize_student_t_centers(
    model: ClusteringBaseModel,
    cache: dict[str, Any],
    device: torch.device,
    seed: int,
) -> None:
    embeddings, _ = _all_cached_embeddings(model, cache, device, batch_size=128)
    result = fit_kmeans(
        embeddings,
        int(model.head.n_clusters),
        spherical=False,
        init="kmeans++",
        n_init=10,
        max_iter=300,
        tol=1.0e-4,
        seed=seed,
        device=device,
    )
    model.head.cluster_centers.data.copy_(result["centers"].to(device))


def _refresh_cached_target(
    model: ClusteringBaseModel,
    cache: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    embeddings, _ = _all_cached_embeddings(model, cache, device, batch_size=batch_size)
    q_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], batch_size):
            q = model.head(embeddings[start : start + batch_size].to(device))["q"]
            q_parts.append(q.detach().cpu())
    return model.head.target_from_q(torch.cat(q_parts, dim=0)).to(device)


def _train_student_t_from_cache(
    model: ClusteringBaseModel,
    cache: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> None:
    model.train()
    model.backbone.eval()
    batch_size = int(config["trainer"]["batch_size"])
    max_epochs = int(config["trainer"]["max_epochs"])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=float(config["trainer"]["lr"]))
    n_samples = cache["cls"].shape[0]
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    for epoch in range(max_epochs):
        target = _refresh_cached_target(model, cache, device, batch_size=128)
        permutation = torch.randperm(n_samples, generator=generator)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            rows = permutation[start : start + batch_size]
            encoded = _cached_embeddings(model, cache, rows, device)
            out = model.head(encoded["embedding"])
            p = target[rows.to(device)]
            loss = F.kl_div(out["q"].clamp_min(1e-12).log(), p.detach(), reduction="batchmean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        print(f"epoch={epoch + 1} loss={epoch_loss / max(1, n_batches):.6f}", flush=True)


def _predict_from_cache(
    model: ClusteringBaseModel,
    cache: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    embeddings, attention = _all_cached_embeddings(model, cache, device, batch_size=128)
    assignments: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], 128):
            assignments.append(model.head.predict(embeddings[start : start + 128].to(device)).detach().cpu())
    return {
        "embeddings": embeddings,
        "labels": cache["labels"],
        "assignments": torch.cat(assignments, dim=0),
        "image_ids": cache["image_ids"],
        "attention": attention,
        "patch_grid": cache["patch_grid"],
    }
