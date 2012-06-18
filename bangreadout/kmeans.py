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

    fsize = (7, 7)
    n_filters = 1
    N_ITER = 8

    X = view_as_windows(X, fsize)
    X = X.reshape(np.prod(X.shape[:2]), -1)
    X = pearson_normalize(X)
    #X = X[::2]
    #ridx = np.random.permutation(len(X))[:20000]
    #X = X[ridx].copy()
    n_samples, n_features = X.shape

    #fbT_shape = n_features, n_filters
    fb_shape = n_filters, n_features

    # -- theano stuff
    t_X = tensor.fmatrix()
    #t_fbT = tensor.fmatrix()
    t_fb = tensor.fmatrix()

    t_n_filters, t_n_features = t_fb.shape

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
    #D = tensor.eye(t_n_filters, k=1) * 0.5
    ##D += tensor.eye(t_n_filters, k=fsize[0]) * 0.5
    #D = D + D.T + tensor.eye(t_n_filters)
    D = tensor.eye(t_n_filters)

    t_dependency_loss = ((
        tensor.dot(t_fbmn, t_fbmn.T)
        #tensor.clip(tensor.dot(t_fbmn, t_fbmn.T), 0, 1)
        #abs(tensor.dot(t_fbmn, t_fbmn.T))
        -
        D
    ) ** 2.).mean()
    #t_dependency_loss = D.sum()
    #t_dependency_loss = tensor.dot(t_fbmn[2] , t_fbmn[1])
    #t_dependency_loss = tensor.dot(t_fbmn, t_fbmn.T).min()

    # -- final loss
    t_loss = t_coverage_loss + 0 * t_dependency_loss

    # -- gradients
    t_df_dfb = tensor.grad(t_loss, t_fb)

    #t_f = theano.function([X, fbT], [loss],
                          #allow_downcast=True)


    def func(params):
        # -- unpack
        fb = params.reshape(fb_shape).astype('f')
        # -- call theano function
        loss, gradients, c, d = f_df_dfb(X_func, fb)
        print 'coverage:', c
        print 'dependency:', d
        print 'loss:', loss
        stdout.write('.')
        stdout.flush()
        # -- pack
        loss = loss.astype('d')
        gradients = gradients.ravel().astype('d')
        return loss, gradients

    np.random.seed(4242)
    ridx = np.random.permutation(len(X))[:n_filters]
    fb = X[ridx].copy()

    print 'compiling theano function...'
    f_df_dfb = theano.function(
        [t_X, t_fb],
        [t_loss, t_df_dfb,
         t_coverage_loss, t_dependency_loss],
        allow_input_downcast=True,
        )

    for iter in xrange(N_ITER):
        print 'mergin filterbank'
        fbl = fb + [fb + 1e-2 * np.random.randn(*fb.shape) for _ in xrange(3)]
        fb = np.row_stack(fbl)
        fb_shape = fb.shape
        #fb += 1e-1 * np.random.randn(*fb_shape)
        print "running lbfgs...."
        print "X.shape =", X.shape
        X_func = X[iter % 2::2]
        start = time.time()
        best, bestval, info_dct = fmin_l_bfgs_b(
            func,
            fb,
            m=10,#000,
            iprint=1,
            factr=1e12,
            #factr=1e7,
            maxfun=1000#,00,
            )
        #print info_dct
        end = time.time()
        print 'time:', end - start
        fb = best.reshape(fb_shape)
        fb = pearson_normalize(fb)
        print fb[0].mean(), np.linalg.norm(fb[0])
        print fb.shape
        from skimage.util.montage import montage2d
        fb = fb.reshape((-1,) + fsize)
        m = montage2d(fb, rescale_intensity=True)
        misc.imsave('m%02d.png' % (iter + 1), misc.imresize(m, (512, 512), interp='nearest'))

        fb = fb.reshape(-1, np.prod(fsize))

    fb = best.reshape(fb_shape)
    fb = pearson_normalize(fb)
    print fb[0].mean(), np.linalg.norm(fb[0])
    print fb.shape
    fb = fb.reshape((-1,) + fsize)
    m = montage2d(fb, rescale_intensity=True)
    misc.imsave('m2.png', misc.imresize(m, (512, 512), interp='nearest'))

if __name__ == '__main__':
    main()
