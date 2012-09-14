"""Logistic Regression"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#
# License: BSD

# TODO:
#
# * multi-class
#
# * l2 and l1 penalties (regularization)
#
# * pure numpy version ?
#
# * move Average*() class to a general purpose decorator (do the same
# for splitting of features?)


__all__ = ['LBFGSLogisticClassifier', 'AverageLBFGSLogisticClassifier']

from sys import stdout

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import theano
from theano import tensor as T


DEFAULT_LBFGS_PARAMS = dict(
    iprint=1,
    factr=1e7,
    maxfun=1e4,
    )


class LBFGSLogisticClassifier(object):

    def __init__(self,
                 n_features,
                 lbfgs_params=DEFAULT_LBFGS_PARAMS,
                ):

        self.n_features = n_features
        self.lbfgs_params = lbfgs_params

        # stupid binary logreg implementation
        self.W = np.ones((n_features, 2), dtype='float32')
        self.b = np.zeros((2), dtype='float32')
        # XXX: coef_ & intercept_ ? (i.e. a-la sklearn)


    def fit(self, X, Y, reinitialize=True):

        assert X.ndim == 2
        assert Y.ndim == 1
        assert Y.dtype == bool

        assert len(X) == len(Y)
        assert X.shape[1] == self.n_features

        Y_true = Y.ravel().astype(np.int32)
        if reinitialize:
            self.W[:] = 1
            self.b[:] = 0

        # -- initial variables
        W_size = self.W.size
        W_shape = self.W.shape

        # -- theano program
        _X = T.fmatrix()
        _b = T.fvector()  # could be Theano shared variable
        _W = T.fmatrix()  # same
        _Y_true = T.lvector()

        _Y_pred = T.nnet.softmax(T.dot(_X, _W) + _b)
        # negative log likelihood
        _loss = -T.mean(T.log(_Y_pred)[T.arange(_Y_true.shape[0]), _Y_true])

        _dloss_W = T.grad(_loss, _W)
        _dloss_b = T.grad(_loss, _b)

        _f = theano.function([_X, _W, _b],
                             [_Y_pred],
                             allow_input_downcast=True)

        _f_df = theano.function([_X, _Y_true, _W, _b],
                                [_Y_pred, _loss, _dloss_W, _dloss_b],
                                allow_input_downcast=True)

        def minimize_me(vars):
            stdout.write('.')
            stdout.flush()
            # unpack W and b
            W = vars[:W_size].reshape(W_shape)
            b = vars[W_size:]
            # get loss and gradients from theano function
            Y_pred, loss, dloss_W, dloss_b = _f_df(X, Y_true, W, b)
            # pack dW and db
            dloss = np.concatenate([dloss_W.ravel(), dloss_b.ravel()])
            # fmin_l_bfgs_b needs double precision...
            return loss.astype('float64'), dloss.astype('float64')

        # pack W and b
        vars = np.concatenate([self.W.ravel(), self.b.ravel()])
        best, bestval, info = fmin_l_bfgs_b(minimize_me, vars, **self.lbfgs_params)
        # XXX: more verbosity ?

        self.W = best[:W_size].reshape(W_shape)
        self.b = best[W_size:]
        self._f = _f

        return self


    def transform(self, X):
        assert X.ndim == 2
        Y = self._f(X, self.W, self.b)[0][:, 1]
        return Y


    def predict(self, X):
        Y_pred = self.transform(X) > 0.5
        return Y_pred


class AverageLBFGSLogisticClassifier(object):

    def __init__(self,
                 n_features,
                 lbfgs_params=DEFAULT_LBFGS_PARAMS,
                ):

        self.n_features = n_features
        self.lbfgs_params = lbfgs_params

        # stupid binary logreg implementation
        self.W = np.zeros((n_features, 2), dtype='float32')
        self.b = np.zeros((2), dtype='float32')
        # XXX: coef_ & intercept_ ? (i.e. a-la sklearn)
        self.n_iter = 0
        self.clf = LBFGSLogisticClassifier(n_features, lbfgs_params)


    def partial_fit(self, X, Y, reinitialize=True):
        self.clf.fit(X, Y, reinitialize=reinitialize)
        self.n_iter += 1
        alpha = 1.0 / self.n_iter
        self.W = (1.0 - alpha) * self.W + alpha * self.clf.W
        self.b = (1.0 - alpha) * self.b + alpha * self.clf.b
        return self

    def fit(self, X, Y, reinitialize=True):
        return self.partial_fit(X, Y, reinitialize=reinitialize)


    def transform(self, X):
        assert X.ndim == 2
        self.clf.coef_ = self.W
        Y = self.clf._f(X, self.W, self.b)[0][:, 1]
        return Y


    def predict(self, X):
        Y_pred = self.transform(X) > 0.5
        return Y_pred


if __name__ == '__main__':
    # XXX: delete me
    np.random.seed(42)
    n_points = 10000
    n_features = 100
    X = np.random.randn(n_points, n_features)
    Y = np.random.randn(n_points) > 0
    X[Y] += .1
    print len(X), len(Y)

    clf = AverageLBFGSLogisticClassifier(n_features)
    clf.partial_fit(X[::2], Y[::2])
    Y_pred = clf.predict(X)
    print (Y_pred == Y).mean()
    clf.partial_fit(X[1::2], Y[1::2])
    Y_pred = clf.predict(X)
    print (Y_pred == Y).mean()
