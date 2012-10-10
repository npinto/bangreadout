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
        self._n_samples = 0

    def fit(self, X):
        _check_X(X)
        self._rows_mean, self._rows_std = zmuv_rows(X.T)
        return self

    def partial_fit(self, X):
        _check_X(X)
        rows_mean, rows_std = zmuv_rows(X.T)

        if self._rows_mean is None:

            self._rows_mean = rows_mean
            self._rows_std = rows_std
            self._n_samples = len(X)

        else:

            # XXX: change this, we should use sum and ssq in
            # cython-based helpers

            n_samples_new = self._n_samples + len(X)

            rows_mean_new_n = (
                self._n_samples * self._rows_mean
                +
                len(X) * rows_mean
            )
            rows_mean_new = rows_mean_new_n / n_samples_new

            rows_ssq_new = (
                self._n_samples * ((self._rows_std ** 2.)
                                   + self._rows_mean ** 2.)
                +
                len(X) * ((rows_std ** 2.) + rows_mean ** 2.)
            )
            rows_std_new = np.sqrt(
                rows_ssq_new / n_samples_new
                -
                rows_mean_new ** 2.
                )

            self._rows_mean = rows_mean_new
            self._rows_std = rows_std_new
            self._n_samples = n_samples_new

    def transform(self, X):
        _check_X(X)
        assert self._rows_mean is not None
        assert self._rows_std is not None
        out = X.copy()
        zmuv_rows_apply_inplace(out.T, self._rows_mean, self._rows_std)
        out[~np.isfinite(out)] = 0
        return out
