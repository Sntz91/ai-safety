from .vit import ViT

MODEL_REGISTRY = {
    'vit': ViT,
}

def get_model(name):
    return MODEL_REGISTRY[name]
