import numpy as np

from ai_safety.models.diagnostic.aggregation import aggregate_to_scan_level
from ai_safety.models.monitor.aggregation import aggregate_topk_saliency_to_scan_level
from ai_safety.models.monitor.threshold_distance import compute_decision_distance


def compute_aggregation_strategies(df, subtype, diag_t_scan, k=3):
    series_ids = df["series_id"].values
    orig_diag_probs = df[f"prob_{subtype}_diag"].values
    orig_diag_gts = df[f"label_{subtype}_diag"].values
    risk_slice = df[f"prob_{subtype}_mon"].values

    _, scan_diag_probs, scan_diag_gts = aggregate_to_scan_level(series_ids, orig_diag_probs, orig_diag_gts, k=k)
    scan_diag_preds = (scan_diag_probs >= diag_t_scan).astype(int)
    scan_true = (scan_diag_gts != scan_diag_preds).astype(int)

    _, r_pure_top3 = aggregate_to_scan_level(series_ids, risk_slice, k=k)
    r_sdist = compute_decision_distance(scan_diag_probs, diag_threshold=diag_t_scan)
    _, r_top3_diag = aggregate_topk_saliency_to_scan_level(series_ids, orig_diag_probs, risk_slice, k=k)
    r_hybrid = (r_top3_diag + r_sdist) / 2.0

    strategies = {
        "Pure Black-Box (Top-3 Mean)": {
            "scores": r_pure_top3,
            "input_req": "Image Pixels Only (0 Model Access)",
        },
        "Decision Boundary Baseline": {
            "scores": r_sdist,
            "input_req": "Model Probability & Threshold (|p - τ|)",
        },
        "Top-3 Diagnostic Slices Monitor": {
            "scores": r_top3_diag,
            "input_req": "Image Pixels + Diagnostic Saliency",
        },
        "Hybrid Safety Monitor (Top-3 + Decision)": {
            "scores": r_hybrid,
            "input_req": "Image Pixels + Diagnostic Output + τ",
        },
    }
    return scan_diag_probs, scan_diag_gts, scan_diag_preds, scan_true, strategies
