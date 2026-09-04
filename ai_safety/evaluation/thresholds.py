import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

from ai_safety.constants import DEFAULT_TARGET_SENSITIVITY
from ai_safety.models.diagnostic.aggregation import aggregate_to_scan_level


def _threshold_for_sensitivity(y_true, y_prob, target_sens=DEFAULT_TARGET_SENSITIVITY):
    if y_true.sum() == 0:
        return 0.5
    fpr, tpr, th = roc_curve(y_true, y_prob, drop_intermediate=False)
    valid_idx = np.where(tpr >= target_sens)[0]
    if len(valid_idx) > 0:
        return float(np.clip(th[valid_idx[0]], 0.0, 1.0))
    return 0.5


def _threshold_for_f1(y_true, y_prob):
    if y_true.sum() == 0:
        return 0.5
    precision, recall, th = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    idx = np.argmax(f1)
    if idx < len(th):
        return float(np.clip(th[idx], 0.0, 1.0))
    return 1.0


def get_thresholds_for_sensitivity(preds, target_sens: float = DEFAULT_TARGET_SENSITIVITY):
    """Computes the probability threshold required to hit target sensitivity per class."""
    return [
        _threshold_for_sensitivity(preds.labels[:, i], preds.probs[:, i], target_sens)
        for i in range(preds.probs.shape[1])
    ]


def get_thresholds_for_f1(preds):
    """Computes the probability threshold required to maximize the F1 score per class."""
    return [
        _threshold_for_f1(preds.labels[:, i], preds.probs[:, i])
        for i in range(preds.probs.shape[1])
    ]


def get_scan_thresholds_for_sensitivity(preds, target_sens: float = DEFAULT_TARGET_SENSITIVITY, k: int = 3):
    """Computes the probability threshold required to hit target sensitivity at the scan level."""
    thresholds = []
    for i in range(preds.probs.shape[1]):
        _, scan_prob, scan_true = aggregate_to_scan_level(preds.series_id, preds.probs[:, i], preds.labels[:, i], k=k)
        thresholds.append(_threshold_for_sensitivity(scan_true, scan_prob, target_sens))
    return thresholds


def get_scan_thresholds_for_f1(preds, k: int = 3):
    """Computes the probability threshold required to maximize F1 score at the scan level."""
    thresholds = []
    for i in range(preds.probs.shape[1]):
        _, scan_prob, scan_true = aggregate_to_scan_level(preds.series_id, preds.probs[:, i], preds.labels[:, i], k=k)
        thresholds.append(_threshold_for_f1(scan_true, scan_prob))
    return thresholds

