from .vit import ViT
from .aggregation import aggregate_to_scan_level

MODEL_REGISTRY = {
    'vit': ViT,
}

def get_model(name):
    return MODEL_REGISTRY[name]
