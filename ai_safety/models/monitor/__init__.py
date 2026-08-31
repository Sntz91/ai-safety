from .densenet import DenseNet

MODEL_REGISTRY = {
    'densenet': DenseNet,
}

def get_model(name):
    return MODEL_REGISTRY[name]
