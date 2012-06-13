# Authors: Nicolas Pinto <pinto@alum.mit.edu>
# License: BSD

__all__ = [
    'DEFAULT_LBFGS_PARAMS', 'LBFGSPearsonRegressor'
]

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import theano
from theano import tensor

from .util import normalize_pearson_like, theano_corrcoef


# ----------------------------------------------------------------------------
# -- LBFGS-based Pearson Regressor
# ----------------------------------------------------------------------------
DEFAULT_N_FILTERS = 128
DEFAULT_L2_REGULARIZATION = 1e-3
DEFAULT_L1_REGULARIZATION = 1e-3
DEFAULT_LBFGS_PARAMS = dict(
    iprint=1,
    factr=1e7,
    maxfun=1000,
    )


class LBFGSPearsonRegressor(object):

    def __init__(self, n_filters=DEFAULT_N_FILTERS,
                 l2_regularization=DEFAULT_L2_REGULARIZATION,
                 l1_regularization=DEFAULT_L1_REGULARIZATION,
                 lbfgs_params=DEFAULT_LBFGS_PARAMS):

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

        l2_loss = tensor.dot(t_fbmn.flatten(), t_fbmn.flatten())
        loss += l2_regularization * l2_loss

        l1_loss = tensor.sum(abs(t_fbmn.flatten()))
        loss += l2_regularization * l1_loss

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
