import numpy as np
from collections import defaultdict

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
        # Only aggregate if we actually have a scan ID
        if scan_id is not None and str(scan_id) != "None" and str(scan_id) != "nan":
            scan_dict[scan_id].append(slice_probs[i])
            if slice_gts is not None:
                gt_dict[scan_id].append(slice_gts[i])
            
    unique_scan_ids = []
    scan_probs_out = []
    scan_gts_out = []
    
    for scan_id, probs in scan_dict.items():
        probs = np.array(probs)
        top_k = min(k, len(probs))
        
        # Sort ascending, take the last `top_k` elements (highest probabilities), and average
        top_probs = np.sort(probs)[-top_k:]
        scan_prob = np.mean(top_probs)
        
        unique_scan_ids.append(scan_id)
        scan_probs_out.append(scan_prob)
        
        if slice_gts is not None:
            # Ground truth aggregation: a scan is positive if ANY slice is positive
            scan_gt = np.max(gt_dict[scan_id])
            scan_gts_out.append(scan_gt)
            
    if slice_gts is not None:
        return unique_scan_ids, np.array(scan_probs_out), np.array(scan_gts_out)
        
    return unique_scan_ids, np.array(scan_probs_out)

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
        if scan_id is not None and str(scan_id) != "None" and str(scan_id) != "nan":
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
        
        # 1. Determine the diagnostic scan-level prediction (Top-K Mean)
        top_k = min(k, len(d_probs))
        diag_scan_prob = np.mean(np.sort(d_probs)[-top_k:])
        is_positive = diag_scan_prob >= diag_threshold
        
        # 2. Dual-Pooling for the Monitor
        if is_positive:
            # False Positive mode -> use MAX pooling (k=1)
            mon_scan_prob = np.max(m_probs)
        else:
            # False Negative mode -> use Top-K Mean pooling
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
