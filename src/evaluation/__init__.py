__all__ = ["compute_clustering_metrics"]


def __getattr__(name: str):
    if name == "compute_clustering_metrics":
        from src.evaluation.clustering_metrics import compute_clustering_metrics

        return compute_clustering_metrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
