# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# XXX: clean this up, remove code duplication

import numpy as np
cimport numpy as cnp

from libc.math cimport sqrt, pow
from cython.parallel import prange

# square root C function
cdef float _sqrt(float x):
        return sqrt(x)

def zmuv_rows(
    cnp.ndarray[cnp.float32_t, ndim=2] arr
    ):

    cdef Py_ssize_t rows = arr.shape[0]
    cdef Py_ssize_t cols = arr.shape[1]

    cdef cnp.ndarray[cnp.float32_t, ndim=1] rows_mean = np.zeros(rows, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] rows_std = np.zeros(rows, dtype=np.float32)

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef cnp.float32_t scaler = 1. / <cnp.float32_t>cols

    # -- computing the vector of mean values
    for i in prange(rows, nogil=True):

        for j in range(cols):
            rows_mean[i] += arr[i, j]
        rows_mean[i] *= scaler

    # -- now we compute the vector of standard deviations
    for i in prange(rows, nogil=True):

        for j in range(cols):
            rows_std[i] += (arr[i, j] - rows_mean[i]) ** 2
        rows_std[i] *= scaler

    for i in range(rows):
        rows_std[i] = (<cnp.float32_t>_sqrt(<float>rows_std[i]))

    return rows_mean, rows_std


def zmuv_rows_apply_inplace(
    cnp.ndarray[cnp.float32_t, ndim=2] arr,
    cnp.ndarray[cnp.float32_t, ndim=1] rows_mean,
    cnp.ndarray[cnp.float32_t, ndim=1] rows_std,
    ):

    cdef Py_ssize_t rows = arr.shape[0]
    cdef Py_ssize_t cols = arr.shape[1]

    cdef Py_ssize_t i
    for i in prange(rows, nogil=True):
        for j in range(cols):
            arr[i, j] = (arr[i, j] - rows_mean[i]) / rows_std[i]


def zmuv_rows_inplace(
    cnp.ndarray[cnp.float32_t, ndim=2] arr
    ):

    cdef Py_ssize_t rows = arr.shape[0]
    cdef Py_ssize_t cols = arr.shape[1]

    cdef cnp.ndarray[cnp.float32_t, ndim=1] mean = np.zeros(rows, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] std = np.zeros(rows, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] std_inv = np.zeros(rows, dtype=np.float32)

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef cnp.float32_t scaler = 1. / <cnp.float32_t>cols

    # -- computing the vector of mean values
    for i in prange(rows, nogil=True):

        for j in range(cols):
            mean[i] += arr[i, j]
        mean[i] *= scaler

    # -- now we compute the vector of standard deviations
    for i in prange(rows, nogil=True):

        for j in range(cols):
            std[i] += (arr[i, j] - mean[i]) ** 2
        std[i] *= scaler

    for i in range(rows):
        std_inv[i] = 1. / (<cnp.float32_t>_sqrt(<float>std[i]))

    # -- computing the final output array (zero mean, unit variance of the input)
    for i in prange(rows, nogil=True):

        for j in range(cols):
            arr[i, j] = std_inv[i] * (arr[i, j] - mean[i])


def zmuv_rows_inplace_untested(
    cnp.ndarray[cnp.float32_t, ndim=2] arr
    ):

    cdef Py_ssize_t rows = arr.shape[0]
    cdef Py_ssize_t cols = arr.shape[1]

    cdef cnp.ndarray[cnp.float32_t, ndim=1] _sum = np.zeros(rows, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] _ssq = np.zeros(rows, dtype=np.float32)

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef cnp.float32_t scaler = 1. / <cnp.float32_t>cols
    cdef cnp.float32_t val, _sum_i, _ssq_i, div

    # -- sum and ssq
    for i in prange(rows, nogil=True):
        for j in range(cols):
            val = arr[i, j]
            _sum[i] += val
            _ssq[i] += val * val

    # -- zmuv
    for i in prange(rows, nogil=True):
        _sum_i = _sum[i]
        _ssq_i = _ssq[i]
        for j in range(cols):
            val = arr[i, j]
            val = val - (scaler * _sum_i)  # zero-mean
            div = sqrt((scaler * _ssq_i) - pow((scaler * _sum_i), 2))  # unit-variance
            val = val / div
            arr[i, j] = val
