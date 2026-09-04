import numpy as np
from collections import defaultdict

from ai_safety.utils.helpers import is_valid_scan_id


def aggregate_to_scan_level(scan_ids, slice_probs, slice_gts=None, k=3):
    """
    Aggregate slice-level probabilities to scan-level probabilities using Top-K Mean.

    Args:
        scan_ids: list/array of strings indicating the parent scan (series) for each slice.
        slice_probs: 1D array of slice-level probabilities.
        slice_gts: 1D array of slice-level ground truths (optional).
        k: The number of top slices to average (default: 3).

    Returns:
        unique_scan_ids: list of unique scan IDs.
        scan_probs: 1D array of aggregated scan probabilities.
        scan_gts: 1D array of aggregated scan ground truths (only if slice_gts provided).
    """
    scan_dict = defaultdict(list)
    gt_dict = defaultdict(list)

    for i, scan_id in enumerate(scan_ids):
        if is_valid_scan_id(scan_id):
            scan_dict[scan_id].append(slice_probs[i])
            if slice_gts is not None:
                gt_dict[scan_id].append(slice_gts[i])

    unique_scan_ids = []
    scan_probs_out = []
    scan_gts_out = []

    for scan_id, probs in scan_dict.items():
        probs = np.array(probs)
        top_k = min(k, len(probs))

        top_probs = np.sort(probs)[-top_k:]
        scan_prob = np.mean(top_probs)

        unique_scan_ids.append(scan_id)
        scan_probs_out.append(scan_prob)

        if slice_gts is not None:
            scan_gt = np.max(gt_dict[scan_id])
            scan_gts_out.append(scan_gt)

    if slice_gts is not None:
        return unique_scan_ids, np.array(scan_probs_out), np.array(scan_gts_out)

    return unique_scan_ids, np.array(scan_probs_out)
