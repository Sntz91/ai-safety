import numpy as np

METADATA = {
    "type": "discrete",
    "category": "operating_point",
    "name": "specificity"
}

def evaluate(preds, targets):
    """Compute Specificity / True Negative Rate."""
    preds = np.asarray(preds, dtype=np.int64).flatten()
    targets = np.asarray(targets, dtype=np.int64).flatten()
    tn = int(np.sum((preds == 0) & (targets == 0)))
    fp = int(np.sum((preds == 1) & (targets == 0)))
    return float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
