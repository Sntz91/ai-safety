import numpy as np

METADATA = {
    "type": "discrete",
    "category": "operating_point",
    "name": "f1"
}

def evaluate(preds, targets):
    """Compute F1 Score (harmonic mean of precision and sensitivity)."""
    preds = np.asarray(preds, dtype=np.int64).flatten()
    targets = np.asarray(targets, dtype=np.int64).flatten()
    tp = int(np.sum((preds == 1) & (targets == 1)))
    fp = int(np.sum((preds == 1) & (targets == 0)))
    fn = int(np.sum((preds == 0) & (targets == 1)))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else np.nan
