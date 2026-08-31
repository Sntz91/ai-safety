from ai_safety.evaluation.evaluator import evaluate_binary
from ai_safety.utils.aggregation import aggregate_to_scan_level, aggregate_dual_pooling_to_scan_level


def evaluate_monitor_risk(
    df, subtype, risk, diag_t_slice, diag_t_scan, slice_locked, scan_locked, bootstraps=100, metric_funcs=None, k=3
):
    """Evaluate a single risk-score vector (higher = riskier) at slice and scan level.

    Derives slice-level error ground truth `e = label_<subtype>_mon` and true scan-level
    errors from the diagnostic model predictions vs diagnostic ground truth.
    """
    y_true = df[f"label_{subtype}_mon"].values

    # 1. Slice-Level Evaluation
    sl_metrics, sl_curves = evaluate_binary(y_true, risk, slice_locked, bootstraps, metric_funcs)

    # 2. Scan-Level Evaluation (Dual Pooling)
    sc_metrics, sc_curves = {}, {}
    if "series_id" in df.columns:
        series_ids = df["series_id"].values
        orig_diag_gts = df[f"label_{subtype}_diag"].values
        orig_diag_probs = df[f"prob_{subtype}_diag"].values

        # Determine true scan-level diagnostic errors
        _, scan_diag_probs, scan_diag_gts = aggregate_to_scan_level(series_ids, orig_diag_probs, orig_diag_gts, k=k)
        scan_diag_preds = (scan_diag_probs >= diag_t_scan).astype(int)
        scan_true = (scan_diag_gts != scan_diag_preds).astype(int)

        # Aggregate monitor risk using dual pooling
        _, scan_risk = aggregate_dual_pooling_to_scan_level(
            series_ids, diag_probs=orig_diag_probs, mon_probs=risk, diag_threshold=diag_t_slice, k=k
        )

        sc_metrics, sc_curves = evaluate_binary(scan_true, scan_risk, scan_locked, bootstraps, metric_funcs)

    return sl_metrics, sl_curves, sc_metrics, sc_curves
