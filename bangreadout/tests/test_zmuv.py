#!/usr/bin/env python
"""Simple test for Cython zmuv function """

# Authors: Nicolas Poilvert <nicolas.poilvert@gmail.com>
#          Nicolas Pinto <nicolas.pinto@gmail.com>

# Licence: BSD


import numpy as np
from numpy.testing import assert_allclose

from bangreadout import zmuv_rows

RTOL = 1e-5
ATOL = 1e-6


def test_zmuv_against_numpy():

    arr = np.random.randn(10000,100).astype(np.float32)

    # -- numpy reference
    ref = (arr - arr.mean(axis=1)[:, np.newaxis]) / arr.std(axis=1)[:, np.newaxis]

    # -- Cython output
    res = zmuv_rows(arr)

    assert_allclose(ref, res, rtol=RTOL, atol=ATOL)
