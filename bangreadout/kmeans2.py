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
    Xmn = (Xm.T / (np.sqrt((Xm ** 2.).sum(1)) + 1e-3).T).T
    return Xmn

def theano_pearson_normalize(X):
    Xm = (X.T - X.mean(1).T).T
    Xmn = (Xm.T / (tensor.sqrt((Xm ** 2.).sum(1)) + 1e-3).T).T
    return Xmn

from skimage.util.montage import montage2d

def main():
    #X = (misc.lena() / 255.).astype('f')
    #X = misc.imread()
    fname = '/home/npinto/datasets/connectomics/isbi2012/pngs/train-volume.tif-29.png'
    Xorig = (misc.imread(fname, flatten=True) / 255.).astype('f')
    #fname = '/home/npinto/datasets/connectomics/isbi2012/pngs/train-labels.tif-29.png'
    #Xorig = ((misc.imread(fname, flatten=True) > 0.) / 255.).astype('f')
    #X = X[X>0]
    #print X.shape
    #X90 = (misc.imrotate(X, 90) / 255.).astype('f')
    #X180 = (misc.imrotate(X, 180) / 255.).astype('f')
    #X270 = (misc.imrotate(X, 270) / 255.).astype('f')
    #X = np.row_stack((X, X90, X180, X270))
    #X = np.row_stack((X, X90))

    #fsize = (2, 2)
    #n_filters = 1
    #N_ITER = 8
    fsize = 64, 64
    n_filters = 1024

    X = view_as_windows(Xorig, fsize)
    X = X.reshape(np.prod(X.shape[:2]), -1)
    #X = X[X[:, X.shape[1] / 2] == 0]
    X = pearson_normalize(X)
    n_samples, n_features = X.shape
    fb_shape = n_filters, n_features

    # -- theano stuff
    t_X = tensor.fmatrix()
    t_fb = tensor.fmatrix()

    t_n_filters, t_n_features = t_fb.shape

    # normalize filters
    t_fbm = (t_fb.T - t_fb.mean(1).T).T
    t_fbmn = (t_fbm.T / tensor.sqrt((t_fbm ** 2.).sum(1)).T).T

    # projection
    t_out = tensor.dot(t_X, t_fbmn.T)
    #t_out = tensor.clip(t_out, 0, 1)

    # -- coverage
    m = 0.1
    t_loss = tensor.mean(tensor.maximum(0, 1 - t_out.max(1) - m)) ** 2.
    #t_loss = 1 - t_out.max(1).mean()

    # -- gradients
    t_df_dfb = tensor.grad(t_loss, t_fb)

    def func(params):
        # -- unpack
        fb = params.reshape(fb_shape).astype('f')
        # -- call theano function
        loss, gradients = f_df_dfb(X_func, fb)
        print 'loss:', loss
        stdout.write('.')
        stdout.flush()
        # -- pack
        loss = loss.astype('d')
        gradients = gradients.ravel().astype('d')
        return loss, gradients

    np.random.seed(4242)
    ridx = np.random.permutation(len(X))[:n_filters]
    fb = X[ridx].copy().astype('f')
    #print fb[0].mean(), np.linalg.norm(fb[0])
    #print fb.shape
    #fb = fb.reshape((-1,) + fsize)
    #print fb.shape
    #m = montage2d(fb, rescale_intensity=True)
    #misc.imsave('m01.png', misc.imresize(m, (512, 512), interp='nearest'))
    #raise

    prev_val2 = 1.0
    prev_fb2 = None
    iter2 = 0
    #while True:
    if True:
        prev_val = 1.
        iter = 0
        prev_fb = None

        print 'compiling theano function...'
        f_df_dfb = theano.function(
            [t_X, t_fb],
            [t_loss, t_df_dfb],
            allow_input_downcast=True,
            )

        if True:
        #while True:
            print 'mergin filterbank'
            fbl = [fb + 1e-3 * np.random.randn(*fb.shape) for _ in xrange(2)]
            fb = np.row_stack(fbl)
            fb_shape = fb.shape
            #fb += 1e-1 * np.random.randn(*fb_shape)
            print "running lbfgs...."
            print "X.shape =", X.shape
            #X_func = X[iter % 2::2]
            X_func = X
            start = time.time()
            #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
            best, bestval, info_dct = fmin_l_bfgs_b(
                func,
                fb,
                m=10,#000,
                iprint=1,
                #factr=1e7,#12,
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
            fb = fb.reshape((-1,) + fsize)
            m = montage2d(fb, rescale_intensity=True)
            out_fn = 'm%02d-%02d.png' % ((iter2 + 1), (iter + 1))
            print out_fn
            misc.imsave(out_fn, misc.imresize(m, (512, 512), interp='nearest'))
            fb = fb.reshape(-1, np.prod(fsize))

            print "((prev_val - bestval) / pre_val) < 1e-2", ((prev_val - bestval) / prev_val), prev_val
            #if ((prev_val - bestval) / prev_val) < 1e-2:
            if abs(prev_val - bestval) < 1e-2:
                fb = prev_fb.copy()
                bestval2 = prev_val
                #break

            prev_fb = fb.copy()
            prev_val = bestval

            iter += 1

        print "((prev_val2 - bestval2) / pre_val2) < 1e-2", ((prev_val2 - bestval2) / prev_val2), prev_val2
        #if ((prev_val2 - bestval2) / prev_val2) < 1e-2:
        if abs(prev_val2 - bestval2) < 1e-2:
            fb = prev_fb2.copy()
            #break

        fb = fb.reshape((len(fb), ) + fsize)
        print fb.shape

        newfb = []
        for i in xrange(len(fb)):
            f = fb[i]
            f = misc.imresize(f, 2 * np.array(f.shape)) / 255.
            newfb += [f]
        fsize = f.shape
        newfb = np.array(newfb)
        newfb = newfb.reshape(len(newfb), -1)
        fb = pearson_normalize(newfb)
        print fb.shape

        X = view_as_windows(Xorig, fsize)
        X = X.reshape(np.prod(X.shape[:2]), -1)
        #X = X[X[:, X.shape[1] / 2] == 0]
        X = pearson_normalize(X)
        #print X.shape
        #import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
        #raise

        prev_fb2 = fb.copy()
        prev_val2 = bestval2

        iter2 += 1


    #fb = best.reshape(fb_shape)
    fb = pearson_normalize(fb)
    print fb[0].mean(), np.linalg.norm(fb[0])
    print fb.shape
    fb = fb.reshape((-1,) + fsize)
    m = montage2d(fb, rescale_intensity=True)
    misc.imsave('m99.png', misc.imresize(m, (512, 512), interp='nearest'))
    print prev_val

if __name__ == '__main__':
    main()
