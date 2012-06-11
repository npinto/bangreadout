__all__ = ['theano_corrcoef']

from theano import tensor


def theano_corrcoef(X):
    """Returns correlation coefficients"""
    Xm = (X.T - X.mean(1).T).T
    Xmn = (Xm.T / tensor.sqrt((Xm ** 2.).sum(1)).T).T
    cc = tensor.dot(Xmn, Xmn.T)
    return cc
