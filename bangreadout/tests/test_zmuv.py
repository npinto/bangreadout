"""Simple test for Cython zmuv (zero-mean / unit-variance) function """

# Authors: Nicolas Poilvert <nicolas.poilvert@gmail.com>
#          Nicolas Pinto <nicolas.pinto@gmail.com>

# Licence: BSD


import numpy as np
from numpy.testing import assert_allclose

from bangreadout.zmuv import (
    zmuv_rows,
    zmuv_rows_apply_inplace,
    zmuv_rows_inplace
    )

RTOL = 1e-5
ATOL = 1e-6


def test_zmuv_rows():

    arr = np.random.randn(10000, 100).astype(np.float32)

    # -- numpy reference
    gt = (arr - arr.mean(0)) / arr.std(0)

    # -- cython
    rows_mean, rows_std = zmuv_rows(arr.T)
    gv = (arr - rows_mean) / rows_std
    assert_allclose(gt, gv, rtol=RTOL, atol=ATOL)
    # inplace
    zmuv_rows_apply_inplace(arr.T, rows_mean, rows_std)
    assert_allclose(gt, arr, rtol=RTOL, atol=ATOL)


def test_zmuv_rows_inplace():

    arr = np.random.randn(10000, 100).astype(np.float32)

    # -- numpy reference
    ref = (arr - arr.mean(axis=1)[:, np.newaxis])
    ref /= arr.std(axis=1)[:, np.newaxis]

    # -- Cython output
    zmuv_rows_inplace(arr)

    assert_allclose(ref, arr, rtol=RTOL, atol=ATOL)
