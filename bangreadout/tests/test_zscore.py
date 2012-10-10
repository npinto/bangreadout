# Authors: Nicolas Poilvert <nicolas.poilvert@gmail.com>
#          Nicolas Pinto <nicolas.pinto@gmail.com>

# Licence: BSD


import numpy as np
from numpy.testing import assert_allclose

from bangreadout.zscore import ZScorer

RTOL = 1e-5
ATOL = 1e-6


def test_zscorer():

    arr = np.random.randn(10000, 100).astype(np.float32)

    gt = (arr - arr.mean(0)) / arr.std(0)
    gv = ZScorer().fit(arr).transform(arr)
    assert_allclose(gt, gv, rtol=RTOL, atol=ATOL)


def test_zscorer_partial_fit():
    z = ZScorer()
    x = np.random.randn(10, 1000).astype('f')
    x[1::3] += 1
    x[1::11] += 2
    z.partial_fit(x[::3])
    z.partial_fit(x[1::3])
    z.partial_fit(x[2::3])
    xm = x.mean(0)
    xs = x.std(0)
    assert np.linalg.norm(xm - z._rows_mean) < RTOL
    assert np.linalg.norm(xs - z._rows_std) < RTOL
