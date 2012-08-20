"""Square Hinge Binary Classifier

The code internally uses {-1, +1} for the target values, but outputs predictions
between 0 and 1.

Everything inferior or equal to 0 is mapped to -1 and the rest is mapped to +1.
This concerns only the "ground truth" of course.

The code has many features. In the "fit" method, one can choose to use
mini-batches instead of using the full batch. One can also use a starting value
for the weight vector and the bias in the "fit" method. This allows, e.g. to use
"warm restarts" in the AverageClassifier. Finally, the classifier can be
"biased" towards the positive or negative class by playing with one of the
attributei (here ``negfrac``).
"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import theano
from theano import tensor as T


EPS = 1e-3
DEFAULT_LBFGS_PARAMS = dict(
    iprint=1,
    factr=1e7,
    maxfun=1e4,
    )
DEFAULT_EPS_SENS = 0.1


class LBFGSSqHingeClassifier(object):

    def __init__(self,
                 n_features,
                 lbfgs_params=DEFAULT_LBFGS_PARAMS,
                 eps_sens=DEFAULT_EPS_SENS,
                 negfrac=None,
                ):

        self.n_features = n_features
        self.lbfgs_params = lbfgs_params

        self.W = np.empty((n_features,), dtype=np.float32)
        self.b = np.empty((1), dtype=np.float32)

        self.eps_sens = eps_sens

        self.negfrac = negfrac


    def fit(self, X, Y, w_start=None, b_start=None,
            mini_batch_size=10000, n_maxfun=20, bfgs_m=10):
        """
        fit X to Y using an epsilon-insensitive square hinge classifier.

        Parameters
        ----------

        ``X``: 2-dimensional array-like
            the input matrix of shape [n_samples, n_features]

        ``Y``: 2-dimensional array-like
            the ouput vector telling what is the class label for each feature
            vector in ``X`` (i.e. for each row of ``X``). shape [n_samples,]

        ``w_start``: 1-dimensional array-like
            starting weight vectors of shape [n_features,]. If "None" the vector
            is initialized to a vector of length EPS in a random direction.

        ``b_start``: 1-dimensional vector
            starting bias vector of shape [1,]

        ``mini_batch_size``: integer
            size of the mini-batch, i.e. number of samples to use at one time in
            the optimization.

        ``n_maxfun``: integer
            number of authorized LBFGS iterations per mini-batch. The last
            mini-batch always goes to convergence, so that limit does not apply
            to that last mini-batch.

        ``bfgs_m``: integer
            number of dimensions in the Hessian estimation.
        """

        assert X.ndim == 2
        assert Y.ndim == 1

        assert len(X) == len(Y)
        assert X.shape[1] == self.n_features

        dtype = X.dtype

        # -- transform Y_true from R to {-1, 1}
        Y_true = Y.ravel().astype(np.int32)
        Y_true = np.where(Y_true <= 0, -1, 1)

        # -- if the starting values for the weights and bias are not given, we
        # initialize them
        if w_start == None and b_start == None:
            w = np.random.uniform(low=-EPS, high=EPS,
                                  size=X.shape[1]).astype(dtype)
            w /= np.linalg.norm(w)
            w *= EPS

            b = np.random.uniform(low=-EPS, high=EPS, size=1).astype(dtype)

        elif w_start == None and b_start != None:
            w = np.random.uniform(low=-EPS, high=EPS,
                                  size=X.shape[1]).astype(dtype)
            w /= np.linalg.norm(w)
            w *= EPS

            b_start = np.array(b_start)
            assert b_start.ndim == 1
            assert b_start.size == 1

            b = b_start.astype(dtype)

        elif w_start != None and b_start == None:

            w_start = np.array(w_start)
            assert w_start.ndim == 1
            assert w_start.size == X.shape[1]

            w = w_start.astype(dtype)
            b = np.random.uniform(low=-EPS, high=EPS, size=1).astype(dtype)

        else:
            w_start = np.array(w_start)
            b_start = np.array(b_start)
            assert w_start.ndim == 1
            assert w_start.size == X.shape[1]
            assert b_start.ndim == 1
            assert b_start.size == 1

            w = w_start.astype(dtype)
            b = b_start.astype(dtype)

        # -- initial variables
        w_size = w.size
        m_sens = self.eps_sens

        # -- theano program
        t_X = T.fmatrix()
        t_y = T.fvector()
        t_w = T.fvector()
        t_b = T.fscalar()

        t_H = T.dot(t_X, t_w) + t_b
        t_H = 2. * T.nnet.sigmoid(t_H) - 1

        t_M = t_y * t_H

        # -- here we compute key values for "balancing" the classifier
        t_y_true = (t_y + 1) / 2
        t_npos = T.sum(t_y_true)
        t_nneg = T.sum(1 - t_y_true)
        t_npos_inv = 1 / t_npos
        t_nneg_inv = 1 / t_nneg

        if self.negfrac is None:
            t_frac = t_nneg / (t_npos + t_nneg)
        else:
            t_frac = float(self.negfrac)

        t_loss_pos = t_npos_inv * T.sum(t_y_true * \
                                 (T.maximum(0, 1 - t_M - m_sens) ** 2.))
        t_loss_neg = t_nneg_inv * T.sum((1 - t_y_true) * \
                                 (T.maximum(0, 1 - t_M - m_sens) ** 2.))

        t_loss = (1 - t_frac) * t_loss_pos + t_frac * t_loss_neg

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

        # -- how many mini-batch in X
        n_mini_batch = int(X.shape[0] / mini_batch_size)
        if n_mini_batch <= 1:
            mini_batch_size = X.shape[0]
            n_mini_batch = 1

        # -- compute indices for mini-batch feature vectors
        ref_idx = np.random.permutation(X.shape[0])
        mini_batch_indices = []
        for i in xrange(n_mini_batch):
            mini_batch_indices += [ref_idx[i*mini_batch_size:(i+1)*mini_batch_size]]

        def minimize_me(vars, X_trn, Y_true_trn):

            # unpack W and b
            w_in = vars[:w_size]
            b_in = vars[w_size:]

            # get loss and gradients from theano function
            Y_pred, loss, dloss_w, dloss_b = _f_df(X_trn, Y_true_trn,
                                                   w_in, b_in[0])

            # pack dW and db
            dloss = np.concatenate([dloss_w.ravel(), dloss_b.ravel()])

            # fmin_l_bfgs_b needs double precision...
            return loss.astype(np.float64), dloss.astype(np.float64)

        # mini-batch L-BFGS iterations
        w_av = w.copy()
        b_av = b.copy()
        n_iter = 1.

        if len(mini_batch_indices) > 1:
            # -- mini-batch updates for the weights and bias
            for idx in mini_batch_indices[:-1]:

                X_mb = np.ascontiguousarray(X[idx])
                Y_true_mb = np.ascontiguousarray(Y_true[idx])

                vars = np.concatenate([w.ravel(), b.ravel()])
                best, bestval, info = fmin_l_bfgs_b(minimize_me, vars,
                                                    args=[X_mb, Y_true_mb],
                                                    factr=1e7, maxfun=n_maxfun,
                                                    iprint=1, m=bfgs_m)

                w = best[:w_size]
                b = best[w_size:]

                alpha = 1. / (n_iter + 1.)
                w_av = (1. - alpha) * w_av + alpha * w
                b_av = (1. - alpha) * b_av + alpha * b

            # -- last mini-batch is converged
            X_mb = np.ascontiguousarray(X[mini_batch_indices[-1]])
            Y_true_mb = np.ascontiguousarray(Y_true[mini_batch_indices[-1]])

            vars = np.concatenate([w_av.ravel(), b_av.ravel()])
            best, bestval, info = fmin_l_bfgs_b(minimize_me, vars,
                                                args=[X_mb, Y_true_mb],
                                                factr=1e7, maxfun=15000,
                                                iprint=1, m=bfgs_m)
        else:
            # -- if only one mini-batch exists we converge it
            X_mb = np.ascontiguousarray(X[mini_batch_indices[0]])
            Y_true_mb = np.ascontiguousarray(Y_true[mini_batch_indices[0]])

            vars = np.concatenate([w.ravel(), b.ravel()])
            best, bestval, info = fmin_l_bfgs_b(minimize_me, vars,
                                                args=[X_mb, Y_true_mb],
                                                factr=1e7, maxfun=15000,
                                                iprint=1, m=bfgs_m)

        self.W = w.astype(np.float32)
        self.b = b.astype(np.float32)
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
                 eps_sens=DEFAULT_EPS_SENS,
                 negfrac=None
                 ):

        self.n_features = n_features
        self.lbfgs_params = lbfgs_params
        self.W = np.zeros((n_features,), dtype=np.float32)
        self.b = np.zeros((1), dtype=np.float32)
        self.n_iter = 0

        self.clf = LBFGSSqHingeClassifier(n_features,
                                          lbfgs_params=lbfgs_params,
                                          eps_sens=eps_sens,
                                          negfrac=negfrac)
        self.last_w = self.W.copy()
        self.last_b = self.b.copy()


    def partial_fit(self, X, Y, w_start=None, b_start=None,
                mini_batch_size=10000, n_maxfun=20, bfgs_m=10):

        w_sta = self.last_w.copy()
        b_sta = self.last_b.copy()

        self.clf.fit(X, Y, w_start=w_sta, b_start=b_sta,
                     mini_batch_size=mini_batch_size,
                     n_maxfun=n_maxfun, bfgs_m=bfgs_m)

        self.n_iter += 1

        alpha = 1.0 / self.n_iter
        self.W = (1.0 - alpha) * self.W + alpha * self.clf.W
        self.b = (1.0 - alpha) * self.b + alpha * self.clf.b
        self.last_w = self.clf.W.copy()
        self.last_b = self.clf.b.copy()

        return self


    def transform(self, X):

        assert X.ndim == 2

        Y = self.clf._f(X, self.W, self.b[0])

        # -- retransform Y from [-1, +1] to [0, 1]
        Y = 0.5 * (Y + 1.)

        return Y


    def predict(self, X):

        Y_pred = self.transform(X) > 0.5

        return Y_pred
