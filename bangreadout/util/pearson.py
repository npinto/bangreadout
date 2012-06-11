__all__ = ['normalize_pearson_like']

import numpy as np

EPSILON = 1e-6


def normalize_pearson_like(X):

    assert X.ndim == 2
    axis = 1

    rows, cols = X.shape

    # XXX: you should use keepdims with numpy-1.7
    X_sum = np.apply_over_axes(np.sum, X, (axis,))
    X_ssq = np.apply_over_axes(np.sum, X ** 2., (axis,))

    X_num = X - X_sum / cols

    X_div = np.sqrt((X_ssq - (X_sum ** 2.0) / cols).clip(0, np.inf))
    np.putmask(X_div, X_div < EPSILON, EPSILON)  # avoid zero division

    out = X_num / X_div

    return out
