from sys import stdout
import numpy as np
import time
from zmuv import *

#np.random.seed(42)
#a = np.random.randn(512**2, 1e3).astype('f')
TEST = False
print 'data'
#shape = (16, 32)#(512**2., 256)
shape = (512**2., 256)
a = np.arange(np.prod(shape)).reshape(shape).astype('f')
#a = np.random.randn(*shape).astype('f')

if TEST:
    print 'gt'
    start = time.time()
    gt = a.T.copy()
    gt -= gt.mean(0)
    gt /= gt.std(0)
    gt = gt.T
    end = time.time()
    gt_time = end - start
    print gt

N_ITERATIONS = 50

time_l = []
print 'go'
for i in xrange(N_ITERATIONS):
    #a = a.copy()
    stdout.write('.%d' % i)
    stdout.flush()
    #print i
    start = time.time()
    zmuv_rows_inplace(a)
    #zmuv_rows_inplace_untested(a)
    end = time.time()
    time_l += [end - start]
    if i == 0 and TEST:
        print a
        assert (abs(a - gt) < 1e-3).all()
print
time_l = np.array(time_l)[np.argsort(time_l)[:int(N_ITERATIONS/2)]]
total = np.mean(time_l)
print 'avg time:', total
if TEST:
    print 'speedup:', gt_time / total

#print "fps:", N_ITERATIONS / (end-start)
