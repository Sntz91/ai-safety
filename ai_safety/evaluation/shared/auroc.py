import numpy as np
from sklearn.metrics import roc_auc_score
import math

METADATA = {
    "category": "discrimination",
    "name": "auroc"
}

def evaluate(preds, targets):
    """Computes Area Under the Receiver Operating Characteristic Curve."""
    if len(np.unique(targets)) < 2:
        return math.nan
    return float(roc_auc_score(targets, preds))
