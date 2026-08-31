import numpy as np

METADATA = {
    "type": "continuous",
    "category": "clinical_utility",
    "name": "budgets"
}

def evaluate(preds, targets, budget_fractions=(0.05, 0.10, 0.20)):
    """
    Calculates the percentage of errors intercepted (Recall) and False Reassurance Rate (FRR)
    at specific clinical review budgets. Returns flattened dict for JSON logging.
    """
    preds = np.asarray(preds, dtype=np.float32).flatten()
    targets = np.asarray(targets, dtype=np.int64).flatten()
    n_samples = len(targets)

    results = {}
    if n_samples == 0 or np.sum(targets) == 0:
        for frac in budget_fractions:
            results[f"recall_at_{frac}"] = 0.0
            results[f"frr_at_{frac}"] = 0.0
        return results

    order = np.argsort(-preds)
    sorted_errors = targets[order]
    total_errors = np.sum(targets)

    for frac in budget_fractions:
        k = max(1, int(round(frac * n_samples)))
        
        flagged_errors = np.sum(sorted_errors[:k])
        unflagged_errors = np.sum(sorted_errors[k:])
        unflagged_total = max(1, n_samples - k)

        recall = float(flagged_errors / max(1, total_errors))
        frr = float(unflagged_errors / unflagged_total)

        results[f"recall_at_{frac}"] = recall
        results[f"frr_at_{frac}"] = frr

    return results
