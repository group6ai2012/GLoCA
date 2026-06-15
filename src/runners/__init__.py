__all__ = ["run_cdc", "run_dec_idec", "run_kmeans", "run_propos", "run_student_t", "run_dec"]


def __getattr__(name: str):
    if name == "run_cdc":
        from src.runners.cdc import run_cdc

        return run_cdc
    if name == "run_dec_idec":
        from src.runners.dec_idec import run_dec_idec

        return run_dec_idec
    if name == "run_kmeans":
        from src.runners.kmeans import run_kmeans

        return run_kmeans
    if name == "run_propos":
        from src.runners.propos import run_propos

        return run_propos
    if name in {"run_dec", "run_student_t"}:
        from src.runners.student_t import run_dec, run_student_t

        return {"run_dec": run_dec, "run_student_t": run_student_t}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
