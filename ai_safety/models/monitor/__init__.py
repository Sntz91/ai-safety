from .densenet import DenseNet
from .aggregation import aggregate_dual_pooling_to_scan_level, aggregate_topk_saliency_to_scan_level
from .threshold_distance import compute_s_dist, compute_decision_distance
from .fusion import fuse_risk

MODEL_REGISTRY = {
    'densenet': DenseNet,
}

def get_model(name):
    return MODEL_REGISTRY[name]
