import numpy as np
import time
from zmuv import *

#np.random.seed(42)
#a = np.random.randn(512**2, 1e3).astype('f')
print 'data'
a = np.empty((512**2., 256)).astype('f')

#gt = a.copy()
#gt -= gt.mean(0)
#gt /= gt.std(0)
#print gt

N_ITERATIONS = 10

start = time.time()
print 'go'
for i in xrange(N_ITERATIONS):
    #a = a.copy()
    print i
    #zmuv_rows_inplace_untested(a)
    zmuv_rows_inplace(a)
    #print a
    #if i == 0:
        #assert (abs(a - gt) < 1e-3).all()

end = time.time()

print 'time:', end - start

print "fps:", N_ITERATIONS / (end-start)
