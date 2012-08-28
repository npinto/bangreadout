"""ZScorer

A sklearn-like object to "z-score" normalize (i.e. zero-mean
/ unit-variance) each feature (i.e. column).

Status: alpha
This is very crude (mostly untested) implementation.

"""


# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#
# Licence: BSD


import numpy as np

from bangreadout.zmuv import zmuv_rows, zmuv_rows_apply_inplace


def _check_X(X):
    assert X.dtype == 'float32'
    assert X.ndim == 2


class ZScorer(object):

    def __init__(self):
        self._rows_mean = None
        self._rows_std = None

    def fit(self, X):
        _check_X(X)
        self._rows_mean, self._rows_std = zmuv_rows(X.T)
        return self

    def transform(self, X):
        _check_X(X)
        assert self._rows_mean is not None
        assert self._rows_std is not None
        out = X.copy()
        zmuv_rows_apply_inplace(out.T, self._rows_mean, self._rows_std)
        out[~np.isfinite(out)] = 0
        return out
