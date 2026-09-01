import pandas as pd
from ai_safety.evaluation.evaluator import evaluate_binary
from ai_safety.evaluation.metrics.monitor.budgets import evaluate as evaluate_budgets, evaluate_curve as evaluate_budgets_curve
from ai_safety.models.monitor.threshold_distance import compute_s_dist
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


def evaluate_monitor_dataset(
    df,
    diag_slice_thresholds,
    diag_scan_thresholds,
    mon_slice_thresholds,
    mon_scan_thresholds,
    bootstraps=100,
    metric_funcs=None,
    k=3,
):
    """Evaluate monitor model and threshold-distance baseline across confidence cohorts.

    Args:
        df: Merged DataFrame containing monitor and diagnostic columns.
        diag_slice_thresholds: List of locked diagnostic slice thresholds.
        diag_scan_thresholds: List of locked diagnostic scan thresholds.
        mon_slice_thresholds: List of locked monitor slice thresholds.
        mon_scan_thresholds: List of locked monitor scan thresholds.
        bootstraps: Number of bootstrap iterations.
        metric_funcs: Discovered metric dictionary.
        k: Number of top slices to average for scan-level aggregation.

    Returns:
        ds_metrics: Nested dict of metrics per subtype, cohort, level, and model.
        ds_curves: Nested dict of curves per subtype, cohort, level, and model.
    """
    subtypes = [c.replace("label_", "").replace("_mon", "") for c in df.columns if c.startswith("label_") and c.endswith("_mon")]
    if not subtypes:
        subtypes = [c.replace("label_", "") for c in df.columns if c.startswith("label_")]

    ds_metrics = {}
    ds_curves = {}

    for idx, subtype in enumerate(subtypes):
        diag_t_slice = diag_slice_thresholds[idx] if idx < len(diag_slice_thresholds) else 0.5
        diag_t_scan = diag_scan_thresholds[idx] if idx < len(diag_scan_thresholds) else 0.5
        t_slice = mon_slice_thresholds[idx] if idx < len(mon_slice_thresholds) else 0.5
        t_scan = mon_scan_thresholds[idx] if idx < len(mon_scan_thresholds) else 0.5

        p_diag = df[f"prob_{subtype}_diag"].values
        y_diag = df[f"label_{subtype}_diag"].values
        y_true_slice = df[f"label_{subtype}_mon"].values
        mon_risk_slice = df[f"prob_{subtype}_mon"].values
        sdist_risk_slice = compute_s_dist(p_diag, diag_t_slice)

        slice_error_masks = {
            "all": pd.Series(True, index=df.index),
            "high_conf": ((p_diag <= 0.10) & (y_diag == 1)) | ((p_diag >= 0.80) & (y_diag == 0)),
            "high_conf_fn": (p_diag <= 0.10) & (y_diag == 1),
            "high_conf_fp": (p_diag >= 0.80) & (y_diag == 0),
        }

        scan_error_masks = {}
        if "series_id" in df.columns:
            series_ids = df["series_id"].values
            orig_diag_gts = df[f"label_{subtype}_diag"].values
            orig_diag_probs = df[f"prob_{subtype}_diag"].values
            _, scan_diag_probs, scan_diag_gts = aggregate_to_scan_level(series_ids, orig_diag_probs, orig_diag_gts, k=k)
            scan_diag_preds = (scan_diag_probs >= diag_t_scan).astype(int)
            scan_true = (scan_diag_gts != scan_diag_preds).astype(int)
            _, scan_mon_risk = aggregate_dual_pooling_to_scan_level(
                series_ids, diag_probs=orig_diag_probs, mon_probs=mon_risk_slice, diag_threshold=diag_t_slice, k=k
            )
            _, scan_sdist_risk = aggregate_dual_pooling_to_scan_level(
                series_ids, diag_probs=orig_diag_probs, mon_probs=sdist_risk_slice, diag_threshold=diag_t_slice, k=k
            )
            scan_error_masks = {
                "all": pd.Series(True, index=range(len(scan_diag_probs))),
                "high_conf": ((scan_diag_probs <= 0.10) & (scan_diag_gts == 1)) | ((scan_diag_probs >= 0.80) & (scan_diag_gts == 0)),
                "high_conf_fn": (scan_diag_probs <= 0.10) & (scan_diag_gts == 1),
                "high_conf_fp": (scan_diag_probs >= 0.80) & (scan_diag_gts == 0),
            }

        cohorts = {
            "all": df,
            "high_conf": df[(p_diag <= 0.10) | (p_diag >= 0.80)].reset_index(drop=True),
            "high_conf_fn": df[p_diag <= 0.10].reset_index(drop=True),
            "high_conf_fp": df[p_diag >= 0.80].reset_index(drop=True),
        }

        sub_metrics = {}
        sub_curves = {}

        for c_name, c_df in cohorts.items():
            if len(c_df) == 0:
                continue

            # Trained monitor risk
            mon_risk = c_df[f"prob_{subtype}_mon"].values
            m_sl, m_slc, m_sc, m_scc = evaluate_monitor_risk(
                c_df, subtype, mon_risk, diag_t_slice, diag_t_scan, t_slice, t_scan, bootstraps, metric_funcs, k=k
            )

            # Threshold-distance baseline: s_dist(diag_prob, tau_diag)
            sdist_risk = compute_s_dist(c_df[f"prob_{subtype}_diag"].values, diag_t_slice)
            t_sl, t_slc, t_sc, t_scc = evaluate_monitor_risk(
                c_df, subtype, sdist_risk, diag_t_slice, diag_t_scan, t_slice, t_scan, bootstraps, metric_funcs, k=k
            )

            # Global audit budget evaluation
            m_sl["clinical_utility"] = evaluate_budgets(mon_risk_slice, y_true_slice, mask=slice_error_masks[c_name])
            t_sl["clinical_utility"] = evaluate_budgets(sdist_risk_slice, y_true_slice, mask=slice_error_masks[c_name])
            m_slc["budget_curve"] = evaluate_budgets_curve(mon_risk_slice, y_true_slice, mask=slice_error_masks[c_name])
            t_slc["budget_curve"] = evaluate_budgets_curve(sdist_risk_slice, y_true_slice, mask=slice_error_masks[c_name])

            if scan_error_masks:
                m_sc["clinical_utility"] = evaluate_budgets(scan_mon_risk, scan_true, mask=scan_error_masks[c_name])
                t_sc["clinical_utility"] = evaluate_budgets(scan_sdist_risk, scan_true, mask=scan_error_masks[c_name])
                m_scc["budget_curve"] = evaluate_budgets_curve(scan_mon_risk, scan_true, mask=scan_error_masks[c_name])
                t_scc["budget_curve"] = evaluate_budgets_curve(scan_sdist_risk, scan_true, mask=scan_error_masks[c_name])

            sub_metrics[c_name] = {
                "slice_level": {"monitor": m_sl, "threshold_distance": t_sl},
                "scan_level": {"monitor": m_sc, "threshold_distance": t_sc},
            }
            sub_curves[c_name] = {
                "slice_level": {"monitor": m_slc, "threshold_distance": t_slc},
                "scan_level": {"monitor": m_scc, "threshold_distance": t_scc},
            }

        ds_metrics[subtype] = sub_metrics
        ds_curves[subtype] = sub_curves

    return ds_metrics, ds_curves
