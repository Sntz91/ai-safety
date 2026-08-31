import numpy as np

METADATA = {
    "type": "discrete",
    "category": "operating_point",
    "name": "precision"
}

def evaluate(preds, targets):
    """Compute Precision / Positive Predictive Value (PPV)."""
    preds = np.asarray(preds, dtype=np.int64).flatten()
    targets = np.asarray(targets, dtype=np.int64).flatten()
    tp = int(np.sum((preds == 1) & (targets == 1)))
    fp = int(np.sum((preds == 1) & (targets == 0)))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
