__all__ = ["calibration_error_metrics", "compute_clustering_metrics"]


def __getattr__(name: str):
    if name == "compute_clustering_metrics":
        from src.evaluation.clustering_metrics import compute_clustering_metrics

        return compute_clustering_metrics
    if name == "calibration_error_metrics":
        from src.evaluation.clustering_metrics import calibration_error_metrics

        return calibration_error_metrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
