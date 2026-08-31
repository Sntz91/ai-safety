import numpy as np

METADATA = {
    "type": "continuous",
    "category": "calibration",
    "name": "brier"
}

def evaluate(preds, targets):
    """Compute Brier score (mean squared error between predicted prob and binary GT)."""
    preds = np.asarray(preds, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    return float(np.mean((preds - targets) ** 2))
