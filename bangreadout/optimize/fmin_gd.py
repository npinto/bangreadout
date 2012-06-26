import numpy as np

DEFAULT_STEP_SIZE = 1e-3
DEFAULT_N_ITERATIONS = 10000000

EPSILON = 1e-6
DELTA = 1e-6


def fmin_autosgd(fun, x0, jac=None,
                 step_size=DEFAULT_STEP_SIZE,
                 n_iterations=DEFAULT_N_ITERATIONS):

    if jac is None:
        # TODO: finite difference
        raise NotImplementedError()

    x = x0
    for i in xrange(n_iterations):
        print i, step_size, fun(x)
        gradients = np.asarray(jac(x))
        gradients2 = np.asarray(jac(x + np.ones_like(x) * DELTA))
        diag_hess = gradients2 - gradients / DELTA

        #step_size = (gradients ** 2.) / (diag_hess * gradients.var())
        #step_size = 1. * step_size / np.max(diag_hess)
        #num = np.sum(gradients ** 2)
        #div = min(1e-3, np.mean(diag_hess)) * np.mean(gradients ** 2.)
        #div = np.mean(gradients ** 2.)
        #print num, div, step_size
        #step_size = 1. * num / div
        step_size = 1e-3 / (np.mean(diag_hess) + EPSILON)
        if (abs(gradients) < EPSILON).all():
            break
        if (np.asarray(step_size) == 0).all():
            break
        x -= step_size * gradients
        assert np.isfinite(x).all()

    return x


def fmin_cadieugd(fun, x0, jac=None,
                  step_size=DEFAULT_STEP_SIZE,
                  n_iterations=DEFAULT_N_ITERATIONS):

    if jac is None:
        # TODO: finite difference
        raise NotImplementedError()

    x = x0
    #xa = np.zeros_like(x0)
    for i in xrange(n_iterations):
        print i, step_size, fun(x)
        gradients = np.asarray(jac(x))
        th = max(abs(gradients) / abs(x)) 
        if th > 0.05:
            step_size *= 1.05
        else:
            step_size *= 0.95
        if (abs(gradients) < EPSILON).all():
            break
        x -= step_size * gradients
        #ssa = 1. / (i + 1)
        #xa = (1 - ssa) * xa + ssa * x

    return x


def fmin_gd(fun, x0, jac=None,
            step_size=DEFAULT_STEP_SIZE,
            n_iterations=DEFAULT_N_ITERATIONS):

    if jac is None:
        # TODO: finite difference
        raise NotImplementedError()

    x = x0
    for i in xrange(n_iterations):
        print i, fun(x)
        gradients = np.asarray(jac(x))
        if (abs(gradients) < EPSILON).all():
            break
        x -= step_size * gradients

    return x


fun = lambda a: (a ** 2.).sum()
jac = lambda a: 2 * a


x0 = np.random.randn(10)

#best = fmin_gd(fun, x0, jac=jac)
best = fmin_cadieugd(fun, x0, jac=jac)
#best = fmin_autosgd(fun, x0, jac=jac)
print best
#print loss


# -- create some fake data
x = np.random.rand(10, 5)
y = 2 * (np.random.rand(10) > 0.5) - 1
y = y[:, np.newaxis]
#l2_regularization = 1e-4
#import theano
#from theano import tensor

#t_x = tensor.fmatrix()
#t_y = tensor.fmatrix()
#t_weights = tensor.fvector()
##t_bias = tensor.fscalar()

##t_margin = t_y * (tensor.dot(t_x, t_weights) + t_bias)
#t_margin = t_y * tensor.dot(t_x, t_weights)
#t_loss = tensor.maximum(0, 1 - t_margin) ** 2.
#t_loss = tensor.mean(t_loss)

#t_dloss_dweights = tensor.grad(t_loss, t_weights)
##t_dloss_dbias = tensor.grad(t_loss, t_bias)

#print 'compiling theano function...'
#f = theano.function(
    ##[t_x, t_y, t_weights, t_bias],
    ##[t_loss, t_dloss_dweights, t_dloss_dbias],
    #[t_x, t_y, t_weights],
    #[t_loss, t_dloss_dweights],
    #allow_input_downcast=True,
    #)

#def loss_fn(params):
    ##weights = params[:-1]
    #weights = params
    ##weights = weights[:, np.newaxis]
    ##bias = params[-1]
    ##print bias
    #c, dw = f(x, y, weights)#, bias)
    #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
    ##d = np.concatenate((dw, db))
    #d = dw
    #return c.astype('d'), d.astype('d')

##def f(u):
    ##return fun(u).astype('d'), jac(u).astype('d')

#from scipy.optimize import fmin_l_bfgs_b
#best, loss = fmin_l_bfgs_b(loss_fn, np.zeros(5), iprint=1, m=1)[:2]


