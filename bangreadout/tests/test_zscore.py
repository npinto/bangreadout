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
