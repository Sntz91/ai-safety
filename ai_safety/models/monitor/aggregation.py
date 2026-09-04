import numpy as np
from collections import defaultdict
from ai_safety.models.diagnostic.aggregation import aggregate_to_scan_level
from ai_safety.utils.helpers import is_valid_scan_id


def aggregate_dual_pooling_to_scan_level(scan_ids, diag_probs, mon_probs, diag_threshold=0.5, k=3, slice_gts=None):
    """
    Aggregate monitor probabilities using a dual-pooling strategy based on the diagnostic model's prediction.

    If the diagnostic model predicts POSITIVE for the scan (scan_prob >= diag_threshold),
    any error must be a False Positive (often a highly localized single-slice artifact).
    We use MAX (Top-1) pooling for the monitor.

    If the diagnostic model predicts NEGATIVE for the scan (scan_prob < diag_threshold),
    any error must be a False Negative (missed bleeds spanning multiple slices).
    We use Top-K Mean pooling for the monitor.
    """
    scan_dict_diag = defaultdict(list)
    scan_dict_mon = defaultdict(list)
    gt_dict = defaultdict(list)

    for i, scan_id in enumerate(scan_ids):
        if is_valid_scan_id(scan_id):
            scan_dict_diag[scan_id].append(diag_probs[i])
            scan_dict_mon[scan_id].append(mon_probs[i])
            if slice_gts is not None:
                gt_dict[scan_id].append(slice_gts[i])

    unique_scan_ids = []
    scan_probs_out = []
    scan_gts_out = []

    for scan_id in scan_dict_diag.keys():
        d_probs = np.array(scan_dict_diag[scan_id])
        m_probs = np.array(scan_dict_mon[scan_id])

        top_k = min(k, len(d_probs))
        diag_scan_prob = np.mean(np.sort(d_probs)[-top_k:])
        is_positive = diag_scan_prob >= diag_threshold

        if is_positive:
            mon_scan_prob = np.max(m_probs)
        else:
            top_k_m = min(k, len(m_probs))
            mon_scan_prob = np.mean(np.sort(m_probs)[-top_k_m:])

        unique_scan_ids.append(scan_id)
        scan_probs_out.append(mon_scan_prob)

        if slice_gts is not None:
            scan_gt = np.max(gt_dict[scan_id])
            scan_gts_out.append(scan_gt)

    if slice_gts is not None:
        return unique_scan_ids, np.array(scan_probs_out), np.array(scan_gts_out)

    return unique_scan_ids, np.array(scan_probs_out)


def aggregate_topk_saliency_to_scan_level(scan_ids, diag_probs, mon_probs, k=3):
    """
    Aggregate monitor probabilities using only the top-K diagnostic driving slices.
    Evaluates visual reliability specifically on the slices where the diagnostic model detected pathology.
    """
    scan_dict_diag = defaultdict(list)
    scan_dict_mon = defaultdict(list)

    for i, scan_id in enumerate(scan_ids):
        if is_valid_scan_id(scan_id):
            scan_dict_diag[scan_id].append(diag_probs[i])
            scan_dict_mon[scan_id].append(mon_probs[i])

    unique_scan_ids = []
    scan_probs_out = []

    for scan_id in scan_dict_diag.keys():
        d_probs = np.array(scan_dict_diag[scan_id])
        m_probs = np.array(scan_dict_mon[scan_id])

        top_k = min(k, len(d_probs))
        top_k_diag_idx = np.argsort(d_probs)[-top_k:]
        saliency_mon_prob = np.mean(m_probs[top_k_diag_idx])

        unique_scan_ids.append(scan_id)
        scan_probs_out.append(saliency_mon_prob)

    return unique_scan_ids, np.array(scan_probs_out)
