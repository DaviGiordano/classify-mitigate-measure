from fairlearn.metrics import MetricFrame
import numpy as np


def positive_rate(y_true, y_pred, *, sample_weight=None):
    """
    Proportion of positive predictions.
    """
    y_pred = np.asarray(y_pred)
    if sample_weight is None:
        return float((y_pred == 1).mean())
    sw = np.asarray(sample_weight)
    return float(sw[y_pred == 1].sum() / sw.sum())
