import numpy as np
from sklearn.metrics import average_precision_score
import math

METADATA = {
    "type": "continuous",
    "category": "discrimination",
    "name": "auprc"
}

def evaluate(preds, targets):
    """Computes Area Under the Precision-Recall Curve."""
    if len(np.unique(targets)) < 2:
        return math.nan
    return float(average_precision_score(targets, preds))
