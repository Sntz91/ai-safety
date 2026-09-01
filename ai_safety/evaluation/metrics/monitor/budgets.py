import numpy as np

METADATA = {
    "type": "continuous",
    "category": "clinical_utility",
    "name": "budgets"
}

def evaluate(preds, targets, mask=None, budget_fractions=(0.05, 0.10, 0.20)):
    """Calculates global audit budget recall and FRR across all incoming samples."""
    preds = np.asarray(preds, dtype=np.float32).flatten()
    targets = np.asarray(targets, dtype=np.int64).flatten()
    n_total = len(preds)

    if mask is None:
        mask = np.ones(n_total, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool).flatten()

    total_errors = np.sum(targets[mask])
    if n_total == 0 or total_errors == 0:
        return {f"recall_at_{f}": 0.0 for f in budget_fractions} | {f"frr_at_{f}": 0.0 for f in budget_fractions}

    order = np.argsort(-preds)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(n_total)

    results = {}
    for frac in budget_fractions:
        k = max(1, int(round(frac * n_total)))
        flagged_errors = np.sum(targets[mask & (ranks < k)])
        unflagged_errors = total_errors - flagged_errors
        unflagged_total = max(1, n_total - k)

        results[f"recall_at_{frac}"] = float(flagged_errors / total_errors)
        results[f"frr_at_{frac}"] = float(unflagged_errors / unflagged_total)

    return results


def evaluate_curve(preds, targets, mask=None, budget_fractions=tuple(i / 100 for i in range(0, 55, 5))):
    """Calculates budget vs recall curve points by delegating to evaluate()."""
    raw = evaluate(preds, targets, mask=mask, budget_fractions=budget_fractions)
    return {
        "budgets": [float(f) for f in budget_fractions],
        "recalls": [raw[f"recall_at_{f}"] for f in budget_fractions],
        "frrs": [raw[f"frr_at_{f}"] for f in budget_fractions],
    }
