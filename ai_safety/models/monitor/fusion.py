import numpy as np


def fuse_risk(monitor_risk, decision_risk, weight=0.5):
    """ Weighted monitor- and decision-boundary risk. """
    monitor_risk = np.asarray(monitor_risk, dtype=np.float32)
    decision_risk = np.asarray(decision_risk, dtype=np.float32)
    return weight * monitor_risk + (1.0 - weight) * decision_risk
