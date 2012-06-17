from sys import stdout
import numpy as np
#from autodiff import fmin_l_bfgs_b
import time

#np.random.seed(1)

from scipy import misc
from scipy.optimize import fmin_l_bfgs_b
from skimage.util.shape import view_as_windows

import theano
from theano import tensor

lbfgs_params = dict(
    m=10,
    iprint=1,
    factr=1e7,
    maxfun=1000,
    )

def pearson_normalize(X):
    #assert X.ndim == 2
    Xm = (X.T - X.mean(1).T).T
    Xmn = (Xm.T / np.sqrt((Xm ** 2.).sum(1)).T).T
    return Xmn

def theano_pearson_normalize(X):
    Xm = (X.T - X.mean(1).T).T
    Xmn = (Xm.T / tensor.sqrt((Xm ** 2.).sum(1)).T).T
    return Xmn

def main():
    X = (misc.lena() / 255.).astype('f')
    fsize = (9, 9)
    n_filters = 8*8
    X = view_as_windows(X, fsize)
    X = X.reshape(np.prod(X.shape[:2]), -1)
    X = pearson_normalize(X)
    #ridx = np.random.permutation(len(X))[:20000]
    #X = X[ridx].copy()
    n_samples, n_features = X.shape

    #fbT_shape = n_features, n_filters
    fb_shape = n_filters, n_features

    # -- theano stuff
    t_X = tensor.fmatrix()
    #t_fbT = tensor.fmatrix()
    t_fb = tensor.fmatrix()

    #t_fb = theano_pearson_normalize(t_fb)
    t_fbm = (t_fb.T - t_fb.mean(1).T).T
    #tmp = tensor.sqrt((t_fbm ** 2.).sum(1)).T
    #t_fbmn = (t_fbm.T / tmp).T
    t_fbmn = (t_fbm.T / tensor.sqrt((t_fbm ** 2.).sum(1)).T).T
    #t_fbmn = tmp
    #t_fbmn = t_fbm
    #t_fbT = theano_pearson_normalize(t_fbT.T).T
    #t_fbT = (t_fbT - t_fbT.T.mean(1).T)
    #Xm = (X.T - X.mean(1).T).T
    #Xmn = (Xm.T / tensor.sqrt((Xm ** 2.).sum(1)).T).T
    #return Xmn

    t_out = tensor.dot(t_X, t_fbmn.T)
    t_out = tensor.clip(t_out, 0, 1)

    # -- coverage
    t_coverage_loss = 1 - t_out.max(1).mean()

    # -- dependency
    D = tensor.eye(n_filters, k=1) * 1
    #D = tensor.eye(n_filters, k=fsize[0]) * 0.5
    D = D + D.T + tensor.eye(n_filters)

    t_dependency_loss = ((
        tensor.clip(tensor.dot(t_fb, t_fb.T), 0, 1)
        -
        D
    ) ** 2.).mean()
    #t_dependency_loss = 0

    # -- final loss
    t_loss = t_coverage_loss + 1e-6 * <Wt_dependency_loss

    # -- gradients
    t_df_dfb = tensor.grad(t_loss, t_fb)

    #t_f = theano.function([X, fbT], [loss],
                          #allow_downcast=True)

    print 'compiling theano function...'
    f_df_dfb = theano.function(
        [t_X, t_fb],
        [t_loss, t_df_dfb,
         t_coverage_loss, t_dependency_loss],
        allow_input_downcast=True,
        )

    def func(params):
        # -- unpack
        fb = params.reshape(fb_shape).astype('f')
        # -- call theano function
        loss, gradients, c, d = f_df_dfb(X, fb)
        print 'coverage:', c
        print 'dependency:', d
        print 'loss:', loss
        stdout.write('.')
        stdout.flush()
        # -- pack
        loss = loss.astype('d')
        gradients = gradients.ravel().astype('d')
        return loss, gradients

    #fb = np.random.randn(*fb_shape).astype('f')
    #np.random.seed(41)
    ridx = np.random.permutation(len(X))[:n_filters]
    fb = X[ridx].copy()
    #fbT = pearson_normalize(fbT.T).T

    #print X[0].mean(), np.linalg.norm(X[0])
    #print fb[0].mean(), np.linalg.norm(fb[0])

    print "running lbfgs...."
    print "X.shape =", X.shape
    start = time.time()
    best, bestval, info_dct = fmin_l_bfgs_b(
        func,
        fb,
        m=1000,
        iprint=1,
        factr=1e12,
        maxfun=1000,
        )
    #print info_dct
    end = time.time()
    print 'time:', end - start
    fb = best.reshape(fb_shape)
    print fb.T[0].mean(), np.linalg.norm(fb.T[0])
    print fb.shape
    #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
    from skimage.util.montage import montage2d
    fb = fb.reshape((-1,) + fsize)
    m = montage2d(fb, rescale_intensity=True)
    misc.imsave('m.png', m)


if __name__ == '__main__':
    main()
