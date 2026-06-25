from src.models.clustering import ProPosHead
from src.models.gloca import ADAPTERS


HEADS = {"propos": ProPosHead, "kmeans": None, "dec": None, "idec": None}
GLOCA_REGISTRY = {name: adapter_cls for name, adapter_cls in ADAPTERS.items()}
