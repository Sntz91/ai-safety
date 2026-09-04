import numpy as np

from ai_safety.constants import DEFAULT_TARGET_SENSITIVITY
from ai_safety.evaluation.bootstrap import bootstrap_metric
from ai_safety.utils.curves import get_roc_curve, get_pr_curve


def evaluate_binary(y_true, y_prob, locked_threshold=0.5, bootstraps=100, metric_funcs=None):
    """Domain-agnostic binary evaluation engine.

    Computes continuous calibration/discrimination metrics, ROC and PR curves,
    and discrete performance at locked and optimal operating points.

    Args:
        y_true: 1D array of binary ground truths (0 or 1).
        y_prob: 1D array of predicted probabilities / risk scores.
        locked_threshold: Fixed operating threshold (e.g., from validation set).
        bootstraps: Number of bootstrap iterations for 95% confidence intervals.
        metric_funcs: Dictionary of metric modules discovered via registry.

    Returns:
        metrics: Dictionary containing categorized continuous metrics and operating points.
        curves: Dictionary containing ROC and PR curve points.
    """
    y_true = np.asarray(y_true, dtype=np.int64).flatten()
    y_prob = np.asarray(y_prob, dtype=np.float64).flatten()

    if metric_funcs is None:
        metric_funcs = {}

    continuous_metrics = {
        k: v for k, v in metric_funcs.items() if getattr(v, "METADATA", {}).get("type") == "continuous"
    }
    discrete_metrics = {
        k: v for k, v in metric_funcs.items() if getattr(v, "METADATA", {}).get("type") == "discrete"
    }

    metrics = {}

    # 1. Continuous Metrics (Calibration, Discrimination, Clinical Utility)
    for m_name, mod in continuous_metrics.items():
        cat = mod.METADATA.get("category", "other")
        test_res = mod.evaluate(y_prob, y_true)
        if isinstance(test_res, dict):
            metrics.setdefault(cat, {}).update(test_res)
        else:
            _, ci_l, ci_u = bootstrap_metric(y_true, y_prob, lambda t, p: mod.evaluate(p, t), bootstraps)
            metrics.setdefault(cat, {})[m_name] = {
                "value": float(test_res) if test_res is not None and not np.isnan(test_res) else None,
                "ci_lower": float(ci_l) if ci_l is not None and not np.isnan(ci_l) else None,
                "ci_upper": float(ci_u) if ci_u is not None and not np.isnan(ci_u) else None,
            }

    # 2. Performance Curves
    fpr, tpr, roc_t = get_roc_curve(y_prob, y_true)
    prec, rec, pr_t = get_pr_curve(y_prob, y_true)
    curves = {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_t.tolist()},
        "pr": {"precision": prec.tolist(), "recall": rec.tolist(), "thresholds": pr_t.tolist()},
    }

    # 3. Operating Point Thresholds
    # Optimal Sensitivity (>= target)
    valid_idx = np.where(tpr >= DEFAULT_TARGET_SENSITIVITY)[0]
    t_opt_sens = float(np.clip(roc_t[valid_idx[0]], 0.0, 1.0)) if len(valid_idx) > 0 else 0.5

    # Optimal F1 Score
    f1_arr = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) != 0)
    idx_best = np.argmax(f1_arr)
    t_opt_f1 = float(np.clip(pr_t[idx_best], 0.0, 1.0)) if idx_best < len(pr_t) else 0.5

    op_points = {
        "LOCKED (85% Sens)": float(locked_threshold),
        "OPTIMAL (85% Sens)": t_opt_sens,
        "OPTIMAL (Max F1)": t_opt_f1,
    }

    # 4. Discrete Operating Point Metrics
    metrics["operating_point"] = {}
    for op_name, t_val in op_points.items():
        y_pred = (y_prob >= t_val).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        op_entry = {
            "threshold": float(t_val),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        }

        for m_name, mod in discrete_metrics.items():
            res = mod.evaluate(y_pred, y_true)
            _, ci_l, ci_u = bootstrap_metric(
                y_true, y_prob, lambda t, p: mod.evaluate((p >= t_val).astype(int), t), bootstraps
            )
            op_entry[m_name] = {
                "value": float(res) if res is not None and not np.isnan(res) else None,
                "ci_lower": float(ci_l) if ci_l is not None and not np.isnan(ci_l) else None,
                "ci_upper": float(ci_u) if ci_u is not None and not np.isnan(ci_u) else None,
            }

        metrics["operating_point"][op_name] = op_entry

    return metrics, curves
