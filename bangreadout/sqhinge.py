"""Square Hinge Binary Classifier

The code internally uses {-1, +1} for the target
values, but outputs predictions between 0 and 1"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import theano
from theano import tensor as T


DEFAULT_LBFGS_PARAMS = dict(
    iprint=1,
    factr=1e7,
    maxfun=1e4,
    )


class LBFGSSqHingeClassifier(object):

    def __init__(self,
                 n_features,
                 lbfgs_params=DEFAULT_LBFGS_PARAMS,
                ):

        self.n_features = n_features
        self.lbfgs_params = lbfgs_params

        self.W = np.empty((n_features,), dtype=np.float32)
        self.b = np.empty((1), dtype=np.float32)


    def fit(self, X, Y):

        assert X.ndim == 2
        assert Y.ndim == 1
        assert Y.dtype == bool

        assert len(X) == len(Y)
        assert X.shape[1] == self.n_features

        Y_true = Y.ravel().astype(np.int32)

        # -- transform Y_true from {0, 1} to {-1, 1}
        Y_true = np.where(Y_true <= 0, -1, 1)

        w = np.zeros(X.shape[1], dtype=np.float32)
        b = np.zeros(1, dtype=np.float32)

        # -- initial variables
        w_size = w.size
        m = 0.1

        # -- theano program
        t_X = T.fmatrix()
        t_y = T.fvector()
        t_w = T.fvector()
        t_b = T.fscalar()

        t_H = T.dot(t_X, t_w) + t_b
        t_H = 2. * T.nnet.sigmoid(t_H) - 1

        t_M = t_y * t_H

        t_loss = T.mean(T.maximum(0, 1 - t_M - m) ** 2.)

        t_dloss_dw = T.grad(t_loss, t_w)
        t_dloss_db = T.grad(t_loss, t_b)

        # -- compiling theano functions
        _f = theano.function(
                [t_X, t_w, t_b],
                t_H,
                allow_input_downcast=True)

        _f_df = theano.function(
                [t_X, t_y, t_w, t_b],
                [t_H, t_loss, t_dloss_dw, t_dloss_db],
                allow_input_downcast=True)

        def minimize_me(vars):

            # unpack W and b
            w = vars[:w_size]
            b = vars[w_size:]

            # get loss and gradients from theano function
            Y_pred, loss, dloss_w, dloss_b = _f_df(X, Y_true, w, b[0])

            # pack dW and db
            dloss = np.concatenate([dloss_w.ravel(), dloss_b.ravel()])

            # fmin_l_bfgs_b needs double precision...
            return loss.astype(np.float64), dloss.astype(np.float64)

        # pack W and b
        vars = np.concatenate([w.ravel(), b.ravel()])
        best, bestval, info = fmin_l_bfgs_b(minimize_me, vars, **self.lbfgs_params)

        self.W = best[:w_size]
        self.b = best[w_size:][0]
        self._f = _f

        return self


    def transform(self, X):

        assert X.ndim == 2

        Y = self._f(X, self.W, self.b[0])
        # -- retransform Y from [-1, +1] to [0, 1]
        Y = 0.5 * (Y + 1.)

        return Y


    def predict(self, X):

        Y_pred = self.transform(X) > 0.5

        return Y_pred


class AverageLBFGSSqHingeClassifier(object):

    def __init__(self,
                 n_features,
                 lbfgs_params=DEFAULT_LBFGS_PARAMS,
                ):

        self.n_features = n_features
        self.lbfgs_params = lbfgs_params

        self.W = np.zeros((n_features), dtype=np.float32)
        self.b = np.zeros((1), dtype=np.float32)
        self.n_iter = 0
        self.clf = LBFGSSqHingeClassifier(n_features, lbfgs_params)


    def partial_fit(self, X, Y):

        self.clf.fit(X, Y)

        self.n_iter += 1

        alpha = 1.0 / self.n_iter
        self.W = (1.0 - alpha) * self.W + alpha * self.clf.W
        self.b = (1.0 - alpha) * self.b + alpha * self.clf.b

        return self


    def fit(self, X, Y):

        return self.partial_fit(X, Y)


    def transform(self, X):

        assert X.ndim == 2

        Y = self.clf._f(X, self.W, self.b[0])
        # -- retransform Y from [-1, +1] to [0, 1]
        Y = 0.5 * (Y + 1.)

        return Y


    def predict(self, X):

        Y_pred = self.transform(X) > 0.5

        return Y_pred
