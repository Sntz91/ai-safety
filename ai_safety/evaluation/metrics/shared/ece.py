import numpy as np

METADATA = {
    "type": "continuous",
    "category": "calibration",
    "name": "ece"
}

def evaluate(preds, targets, n_bins=10):
    """Compute standard Expected Calibration Error (ECE) with equal-width bins."""
    preds = np.asarray(preds, dtype=np.float32).flatten()
    targets = np.asarray(targets, dtype=np.float32).flatten()
    n_samples = len(preds)
    if n_samples == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (preds >= bin_lower) & (preds <= bin_upper)
        else:
            in_bin = (preds >= bin_lower) & (preds < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            avg_acc = np.mean(targets[in_bin])
            avg_conf = np.mean(preds[in_bin])
            ece += (bin_count / n_samples) * np.abs(avg_acc - avg_conf)

    return float(ece)
