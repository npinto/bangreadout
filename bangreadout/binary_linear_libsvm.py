import time
import numpy as np

from sklearn import svm

# --
DEFAULT_C = 1e5

class BinaryLinearKernelLibSVM(object):

    def __init__(self, C=DEFAULT_C):
        """XXX: docstring for __init__"""

        self.C = C

    def fit(self, data, lbls):
        """XXX: docstring for fit"""

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        categories = np.unique(lbls)
        assert categories.size == 2

        ntrain = len(lbls)

        assert data.shape[0] == ntrain

        data = data.copy()

        data.shape = ntrain, -1

        print ">>> Computing traintrain linear kernel"
        start = time.time()
        print data.shape
        kernel_traintrain = np.dot(data, data.T)
        ktrace = kernel_traintrain.trace()
        ktrace = ktrace != 0 and ktrace or 1
        kernel_traintrain /= ktrace
        end = time.time()
        print "Time: %s" % (end-start)

        print ">>> Train LibSVM (C=%e)" % self.C
        start = time.time()

        cat = categories[0]
        print "> positive label: '%s'" % cat
        ltrain = np.zeros((len(lbls)))
        ltrain[lbls != cat] = -1
        ltrain[lbls == cat] = +1

        clf = svm.SVC(kernel='precomputed', C=self.C)
        clf.fit(kernel_traintrain, ltrain)

        end = time.time()
        print "Time: %s" % (end-start)

        self._ktrace = ktrace
        self._train_data = data
        self._clf = clf

        self.categories = categories

    def transform(self, data):

        assert np.isfinite(data).all()

        ntest = len(data)

        data = data.copy()

        data.shape = ntest, -1

        assert np.isfinite(data).all()

        print ">>> Computing traintest linear kernel"
        start = time.time()
        kernel_traintest = np.dot(data,
                                  self._train_data.T)

        assert not np.isnan(kernel_traintest).any()
        assert not np.isinf(kernel_traintest).any()

        kernel_traintest /= self._ktrace

        assert not np.isnan(kernel_traintest).any()
        assert not np.isinf(kernel_traintest).any()

        end = time.time()
        print "Time: %s" % (end-start)

        return self._clf.decision_function(kernel_traintest).ravel()

    def predict(self, data):
        """XXX: docstring for transform"""

        outputs = self.transform(data)

        preds = np.sign(outputs).ravel()

        cats = self.categories

        lbls = np.empty(preds.shape, dtype=cats.dtype)
        lbls[preds>=0] = cats[0]
        lbls[preds<0] = cats[1]

        return lbls
