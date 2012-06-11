# Authors: Nicolas Pinto <pinto@alum.mit.edu>
# License: BSD

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import theano
from theano import tensor


EPSILON = 1e-6


# ----------------------------------------------------------------------------
# -- Helpers
# ----------------------------------------------------------------------------
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


def theano_corrcoef(X):
    """Returns correlation coefficients"""
    Xm = (X.T - X.mean(1).T).T
    Xmn = (Xm.T / tensor.sqrt((Xm ** 2.).sum(1)).T).T
    cc = tensor.dot(Xmn, Xmn.T)
    return cc


LBFGS_PARAMS = dict(
    iprint=1,
    factr=1e11,
    maxfun=20,
    )


# ----------------------------------------------------------------------------
# -- LBFGS-based Pearson Regressor
# ----------------------------------------------------------------------------
class LBFGSPearsonRegressor(object):

    def __init__(self, n_filters=128, lbfgs_params=LBFGS_PARAMS):

        self.rng = np.random.RandomState(42)

        # -- inputs
        t_X = tensor.matrix()
        t_gt = tensor.vector()
        t_fb = tensor.matrix()

        # -- normalize fb
        t_fbm = (t_fb.T - t_fb.mean(1).T).T
        t_fbmn = (t_fbm.T / tensor.sqrt((t_fbm ** 2.).sum(1)).T).T

        # -- projection
        t_Y = tensor.dot(t_X, t_fbmn.T)
        t_Y = tensor.clip(t_Y, 0, 1)

        t_gv = theano_corrcoef(t_Y).flatten()
        t_gv = t_gv - t_gv.mean()
        t_gv = t_gv / tensor.sqrt((t_gv **2.).sum())

        similarity = tensor.dot(t_gv, t_gt)
        loss = 1 - similarity
        dloss_fb = theano.grad(loss, t_fb).flatten()

        _f = theano.function([t_X, t_fb],
                             [t_Y],
                             allow_input_downcast=True)

        _f_df = theano.function([t_X, t_gt, t_fb],
                                [loss, dloss_fb, t_Y],
                                allow_input_downcast=True)

        self._f = _f
        self._f_df = _f_df
        self.n_filters = n_filters
        self.lbfgs_params = lbfgs_params

    def fit(self, X, Y):

        ridx = self.rng.permutation(len(X))
        fb0 = np.asarray(X[ridx[:self.n_filters]])
        print fb0.shape
        fb_shape = fb0.shape

        Xn = normalize_pearson_like(X)

        Yn = normalize_pearson_like(Y)
        Ycc = np.dot(Yn, Yn.T)
        Yccn = Ycc.ravel()
        Yccn -= Yccn.mean()
        Yccn /= np.linalg.norm(Yccn)

        def func(params):
            # -- unpack
            fb = params.reshape(fb_shape)
            # -- execute
            l, dl, _Y = self._f_df(Xn, Yccn, fb)
            # -- pack
            outputs = l.astype('float64'), dl.astype('float64')
            return outputs

        print self.lbfgs_params

        params = fb0.ravel()
        best, f, d = fmin_l_bfgs_b(
            func, params,
            **self.lbfgs_params
            )
        self.best_fb = best.reshape(fb_shape)

        return self

    def transform(self, X):
        Xn = normalize_pearson_like(X)
        Y = self._f(Xn, self.best_fb)[0]
        return Y
