import importlib
import pkgutil

import numpy as np
from sklearn.metrics import confusion_matrix

from ai_safety.utils.curves import get_roc_curve, get_pr_curve


def bootstrap_metric(y_true, y_prob, metric_fn, n_bootstraps=1000, random_state=42):
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
    return np.mean(values), np.percentile(values, 2.5), np.percentile(values, 97.5)


def compute_discrete_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 and not np.isnan(ppv) and not np.isnan(sens) else np.nan
    return sens, spec, ppv, f1, tp, fp, tn, fn


def discover_metrics(*packages):
    """Automatically discover metric modules exposing an evaluate function.

    Scans the given packages (e.g. evaluation.shared, evaluation.monitor,
    evaluation.diagnostic). Any module with an ``evaluate`` function is registered
    under its module name.
    """
    metrics = {}
    for package in packages:
        prefix = package.__name__ + "."
        for _, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
            if ispkg:
                continue
            mod = importlib.import_module(modname)
            if hasattr(mod, "evaluate"):
                metrics[modname.split(".")[-1]] = mod.evaluate
    return metrics


def run_evaluation(y_true, y_prob, t_val, t_name, bootstraps, metric_funcs):
    metrics = {"continuous": {}, "discrete": {}}

    for m_name, m_fn in metric_funcs.items():
        test_res = m_fn(y_prob, y_true)
        if isinstance(test_res, dict):
            metrics["continuous"].update(test_res)
        else:
            _, ci_l, ci_u = bootstrap_metric(y_true, y_prob, lambda t, p: m_fn(p, t), bootstraps)
            metrics["continuous"][m_name] = {
                "value": float(test_res) if test_res is not None else None,
                "ci_lower": float(ci_l) if ci_l is not None and not np.isnan(ci_l) else None,
                "ci_upper": float(ci_u) if ci_u is not None and not np.isnan(ci_u) else None
            }

    y_pred = (y_prob >= t_val).astype(int)
    sens, spec, ppv, f1, tp, fp, tn, fn = compute_discrete_metrics(y_true, y_pred)

    metrics["discrete"][t_name] = {
        "threshold": float(t_val),
        "sensitivity": {"value": float(sens)},
        "specificity": {"value": float(spec)},
        "ppv": {"value": float(ppv)},
        "f1": {"value": float(f1)},
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    }

    def _sens(t, p): return compute_discrete_metrics(t, p >= t_val)[0]
    def _spec(t, p): return compute_discrete_metrics(t, p >= t_val)[1]
    def _ppv(t, p): return compute_discrete_metrics(t, p >= t_val)[2]
    def _f1(t, p): return compute_discrete_metrics(t, p >= t_val)[3]

    for m_key, _fn in [("sensitivity", _sens), ("specificity", _spec), ("ppv", _ppv), ("f1", _f1)]:
        _, ci_l, ci_u = bootstrap_metric(y_true, y_prob, _fn, bootstraps)
        metrics["discrete"][t_name][m_key]["ci_lower"] = float(ci_l)
        metrics["discrete"][t_name][m_key]["ci_upper"] = float(ci_u)

    fpr, tpr, roc_t = get_roc_curve(y_prob, y_true)
    prec, rec, pr_t = get_pr_curve(y_prob, y_true)
    curves = {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_t.tolist()},
        "pr": {"precision": prec.tolist(), "recall": rec.tolist(), "thresholds": pr_t.tolist()}
    }
    return metrics, curves
