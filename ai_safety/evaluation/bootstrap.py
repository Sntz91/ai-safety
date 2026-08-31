import numpy as np


def bootstrap_metric(y_true, y_prob, metric_fn, n_bootstraps=1000, random_state=42):
    """Compute mean and 95% bootstrap confidence intervals for a metric function."""
    rng = np.random.RandomState(random_state)
    values = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        val = metric_fn(y_true[indices], y_prob[indices])
        if not np.isnan(val):
            values.append(val)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
