import numpy as np

from ai_safety.utils.aggregation import aggregate_to_scan_level, aggregate_dual_pooling_to_scan_level
from ai_safety.utils.curves import get_roc_curve, get_pr_curve
from ai_safety.evaluation.stats import run_evaluation


def evaluate_risk(df, subtype, risk, diag_t_slice, diag_t_scan, slice_locked, scan_locked, bootstraps, metric_funcs):
    """Evaluate a single risk-score vector (higher = riskier) at slice and scan level.

    Dictates the failure ground truth `e = label_<subtype>_mon` (shared by all
    risk models) and the true scan-level errors derived from the diagnostic model.
    """
    y_true = df[f'label_{subtype}_mon'].values

    # --- Slice-Level ---
    sl_metrics, sl_curves = run_evaluation(y_true, risk, slice_locked, "LOCKED (85% Sens)", bootstraps, metric_funcs)

    fpr_arr, tpr_arr, th_arr_roc = get_roc_curve(risk, y_true)
    valid_idx = np.where(tpr_arr >= 0.85)[0]
    dyn_t_slice_sens = float(np.clip(th_arr_roc[valid_idx[0]], 0.0, 1.0)) if len(valid_idx) > 0 else 0.5
    sl_opt_sens, _ = run_evaluation(y_true, risk, dyn_t_slice_sens, "OPTIMAL (85% Sens)", bootstraps, metric_funcs)
    sl_metrics['discrete'].update(sl_opt_sens['discrete'])

    prec_arr, rec_arr, th_arr = get_pr_curve(risk, y_true)
    f1_arr = np.divide(2 * prec_arr * rec_arr, prec_arr + rec_arr, out=np.zeros_like(prec_arr), where=(prec_arr + rec_arr) != 0)
    idx_best = np.argmax(f1_arr)
    dyn_t_slice_f1 = float(np.clip(th_arr[idx_best], 0.0, 1.0)) if idx_best < len(th_arr) else 0.5
    sl_opt_f1, _ = run_evaluation(y_true, risk, dyn_t_slice_f1, "OPTIMAL (Max F1)", bootstraps, metric_funcs)
    sl_metrics['discrete'].update(sl_opt_f1['discrete'])

    # --- Scan-Level ---
    sc_metrics, sc_curves = {}, {}
    if 'series_id' in df.columns:
        series_ids = df['series_id'].values
        orig_diag_gts = df[f'label_{subtype}_diag'].values
        orig_diag_probs = df[f'prob_{subtype}_diag'].values

        _, scan_diag_probs, scan_diag_gts = aggregate_to_scan_level(series_ids, orig_diag_probs, orig_diag_gts, k=3)
        scan_diag_preds = (scan_diag_probs >= diag_t_scan).astype(int)
        scan_true = (scan_diag_gts != scan_diag_preds).astype(int)

        _, scan_risk = aggregate_dual_pooling_to_scan_level(
            series_ids, diag_probs=orig_diag_probs, mon_probs=risk, diag_threshold=diag_t_slice, k=3
        )

        sc_metrics, sc_curves = run_evaluation(scan_true, scan_risk, scan_locked, "LOCKED (85% Sens)", bootstraps, metric_funcs)

        fpr_arr, tpr_arr, th_arr_roc = get_roc_curve(scan_risk, scan_true)
        valid_idx = np.where(tpr_arr >= 0.85)[0]
        dyn_t_scan_sens = float(np.clip(th_arr_roc[valid_idx[0]], 0.0, 1.0)) if len(valid_idx) > 0 else 0.5
        sc_opt_sens, _ = run_evaluation(scan_true, scan_risk, dyn_t_scan_sens, "OPTIMAL (85% Sens)", bootstraps, metric_funcs)
        sc_metrics['discrete'].update(sc_opt_sens['discrete'])

        prec_arr, rec_arr, th_arr = get_pr_curve(scan_risk, scan_true)
        f1_arr = np.divide(2 * prec_arr * rec_arr, prec_arr + rec_arr, out=np.zeros_like(prec_arr), where=(prec_arr + rec_arr) != 0)
        idx_best = np.argmax(f1_arr)
        dyn_t_scan_f1 = float(np.clip(th_arr[idx_best], 0.0, 1.0)) if idx_best < len(th_arr) else 0.5
        sc_opt_f1, _ = run_evaluation(scan_true, scan_risk, dyn_t_scan_f1, "OPTIMAL (Max F1)", bootstraps, metric_funcs)
        sc_metrics['discrete'].update(sc_opt_f1['discrete'])

    return sl_metrics, sl_curves, sc_metrics, sc_curves
