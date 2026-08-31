import numpy as np

METADATA = {
    "category": "calibration",
    "name": "ada_ece"
}

def evaluate(preds, targets, n_bins=10):
    """Compute Adaptive Expected Calibration Error (AdaECE) using equal-mass (quantile) bins."""
    preds = np.asarray(preds, dtype=np.float32).flatten()
    targets = np.asarray(targets, dtype=np.float32).flatten()
    n_samples = len(preds)
    if n_samples == 0:
        return 0.0

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_boundaries = np.percentile(preds, quantiles)
    bin_boundaries = np.maximum.accumulate(bin_boundaries)

    ada_ece = 0.0
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
            ada_ece += (bin_count / n_samples) * np.abs(avg_acc - avg_conf)

    return float(ada_ece)
