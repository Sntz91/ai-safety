from ai_safety.evaluation.evaluator import evaluate_binary
from ai_safety.models.diagnostic.aggregation import aggregate_to_scan_level
from ai_safety.utils.helpers import extract_subtypes


def evaluate_diagnostic_dataset(df, slice_thresholds, scan_thresholds, bootstraps=100, metric_funcs=None, k=3):
    """Evaluate diagnostic model predictions for all subtypes in a dataset at slice and scan levels.

    Args:
        df: Pandas DataFrame containing `label_<subtype>` and `prob_<subtype>` columns.
        slice_thresholds: List/array of locked slice-level thresholds per subtype.
        scan_thresholds: List/array of locked scan-level thresholds per subtype.
        bootstraps: Number of bootstrap iterations.
        metric_funcs: Discovered metric dictionary.
        k: Number of top slices to average for scan-level aggregation.

    Returns:
        ds_metrics: Nested dict of metrics per subtype and aggregation level.
        ds_curves: Nested dict of curves per subtype and aggregation level.
    """
    subtypes = extract_subtypes(df.columns, suffix="")
    ds_metrics = {}
    ds_curves = {}

    for idx, subtype in enumerate(subtypes):
        y_true = df[f"label_{subtype}"].values
        y_prob = df[f"prob_{subtype}"].values
        t_slice = slice_thresholds[idx] if idx < len(slice_thresholds) else 0.5

        # 1. Slice-Level Evaluation
        sl_metrics, sl_curves = evaluate_binary(y_true, y_prob, t_slice, bootstraps, metric_funcs)

        # 2. Scan-Level Evaluation
        if "series_id" in df.columns:
            t_scan = scan_thresholds[idx] if idx < len(scan_thresholds) else 0.5
            _, scan_prob, scan_true = aggregate_to_scan_level(df["series_id"].values, y_prob, y_true, k=k)
            sc_metrics, sc_curves = evaluate_binary(scan_true, scan_prob, t_scan, bootstraps, metric_funcs)
        else:
            sc_metrics, sc_curves = {}, {}

        ds_metrics[subtype] = {"slice_level": sl_metrics, "scan_level": sc_metrics}
        ds_curves[subtype] = {"slice_level": sl_curves, "scan_level": sc_curves}

    return ds_metrics, ds_curves
