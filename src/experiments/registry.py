from src.models.clustering import ProPosHead, StudentTHead
from src.models.gloca import ADAPTERS


HEADS = {"student_t": StudentTHead, "propos": ProPosHead, "kmeans": None, "dec": None, "idec": None}
GLOCA_REGISTRY = {name: adapter_cls for name, adapter_cls in ADAPTERS.items()}
