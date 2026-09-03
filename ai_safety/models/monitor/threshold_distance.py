import numpy as np

def compute_s_dist(probs: np.ndarray, thresholds) -> np.ndarray:
    """
    Computes the threshold distance baseline (s_dist) from the primary diagnostic model.
    
    Formula: s_dist(f(x)) = min( f(x)/tau, (1-f(x))/(1-tau) )
    
    Where s_dist = 1 represents maximum uncertainty right at the decision boundary, 
    and 0 indicates maximum confidence.
    
    Args:
        probs (np.ndarray): Primary model predicted probabilities.
        thresholds: List or array of decision thresholds (tau_diag).
        
    Returns:
        np.ndarray: Failure risk score in [0, 1].
    """
    t_vals = np.asarray(thresholds, dtype=np.float32)
    
    # Broadcast threshold array if multi-class
    if t_vals.ndim == 1 and probs.ndim > 1:
        t_vals = t_vals[np.newaxis, :]
        
    term1 = probs / t_vals
    term2 = (1.0 - probs) / (1.0 - t_vals)
    
    return np.minimum(term1, term2)


def compute_decision_distance(scan_diag_probs, diag_threshold=0.5):
    """
    Compute normalized decision boundary uncertainty (s_dist) at the scan level.
    Score is 1.0 at the decision threshold tau, and scales to 0.0 at maximum distance.
    """
    scan_diag_probs = np.asarray(scan_diag_probs)
    max_dist_below = max(diag_threshold, 1e-4)
    max_dist_above = max(1.0 - diag_threshold, 1e-4)

    s_dist_norm = np.where(
        scan_diag_probs < diag_threshold,
        1.0 - (diag_threshold - scan_diag_probs) / max_dist_below,
        1.0 - (scan_diag_probs - diag_threshold) / max_dist_above
    )
    return np.clip(s_dist_norm, 0.0, 1.0)
