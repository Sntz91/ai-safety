from sklearn.metrics import roc_curve, precision_recall_curve


def get_roc_curve(y_prob, y_true):
    """Returns FPR, TPR, and Thresholds."""
    return roc_curve(y_true, y_prob)


def get_pr_curve(y_prob, y_true):
    """Returns Precision, Recall, and Thresholds."""
    return precision_recall_curve(y_true, y_prob)
