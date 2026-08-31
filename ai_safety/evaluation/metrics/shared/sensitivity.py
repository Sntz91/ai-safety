import numpy as np

METADATA = {
    "type": "discrete",
    "category": "operating_point",
    "name": "sensitivity"
}

def evaluate(preds, targets):
    """Compute Sensitivity / True Positive Rate / Recall."""
    preds = np.asarray(preds, dtype=np.int64).flatten()
    targets = np.asarray(targets, dtype=np.int64).flatten()
    tp = int(np.sum((preds == 1) & (targets == 1)))
    fn = int(np.sum((preds == 0) & (targets == 1)))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
