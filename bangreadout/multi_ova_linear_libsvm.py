import time
import numpy as np

import scikits.learn.svm as svm
from shogun import Kernel, Classifier, Features

# XXX: clean up this ugly duplicated code please...
# --
DEFAULT_C = 1e5


# ----------------------------------------------------------------------------
# -- Using scikits
# ----------------------------------------------------------------------------
class MultiOVALinearLibSVMScikits(object):

    def __init__(self, C=DEFAULT_C):
        """XXX: docstring for __init__"""

        self.C = C

    def fit(self, data, lbls):
        """XXX: docstring for fit"""

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        categories = np.unique(lbls)
        assert categories.size > 2

        ntrain = len(lbls)

        assert data.shape[0] == ntrain

        data = data.copy()

        data.shape = ntrain, -1

        print ">>> Computing normalization vectors"
        start = time.time()
        fmean = data.mean(0)
        fstd = data.std(0)
        np.putmask(fstd, fstd==0, 1)
        end = time.time()
        print "Time: %s" % (end-start)

        print ">>> Normalizing training data"
        # XXX: use scikits...Scaler
        start = time.time()
        data -= fmean
        data /= fstd
        end = time.time()
        print "Time: %s" % (end-start)

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        print ">>> Computing traintrain linear kernel"
        start = time.time()
        print data.shape
        kernel_traintrain = np.dot(data, data.T)
        ktrace = kernel_traintrain.trace()
        ktrace = ktrace != 0 and ktrace or 1
        kernel_traintrain /= ktrace
        end = time.time()
        print "Time: %s" % (end-start)

        print ">>> Train LibSVMs (C=%e)" % self.C
        start = time.time()

        cat_index = {}

        alphas = {}
        support_vectors = {}
        biases = {}
        clfs = {}

        for icat, cat in enumerate(categories):
            print "> [%d] positive label: '%s'" % (icat+1, cat)
            ltrain = np.zeros(len(lbls))
            ltrain[lbls != cat] = -1
            ltrain[lbls == cat] = +1

            clf = svm.SVC(kernel='precomputed', C=self.C)
            clf.fit(kernel_traintrain, ltrain)

            alphas[cat] = clf.dual_coef_
            support_vectors[cat] = clf.support_
            biases[cat] = clf.intercept_
            cat_index[cat] = icat
            clfs[cat] = clf

        end = time.time()
        print "Time: %s" % (end-start)

        self._train_data = data
        self._ktrace = ktrace
        self._fmean = fmean
        self._fstd = fstd
        self._support_vectors = support_vectors
        self._alphas = alphas
        self._biases = biases
        self._clfs = clfs

        self.categories = categories

    def transform(self, data):

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        ntest = len(data)

        data = data.copy()

        data.shape = ntest, -1

        print ">>> Normalizing testing data"
        start = time.time()
        data -= self._fmean
        data /= self._fstd
        end = time.time()
        print "Time: %s" % (end-start)

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        print ">>> Computing traintest linear kernel"
        start = time.time()
        kernel_traintest = np.dot(self._train_data, data.T)

        assert not np.isnan(kernel_traintest).any()
        assert not np.isinf(kernel_traintest).any()

        kernel_traintest /= self._ktrace

        assert not np.isnan(kernel_traintest).any()
        assert not np.isinf(kernel_traintest).any()

        end = time.time()
        print "Time: %s" % (end-start)

        print ">>> Collecting %d testing outputs" % ntest
        start = time.time()
        categories = self.categories
        support_vectors = self._support_vectors
        alphas = self._alphas
        biases = self._biases
        clfs = self._clfs

        outputs = np.zeros((ntest, len(categories)), dtype='float32')

        print "Predicting testing data ..."
        for icat, cat in enumerate(categories):
            #index_sv = support_vectors[cat]
            #resps = np.dot(alphas[cat],
                           #kernel_traintest[index_sv]) + biases[cat]
            clf = clfs[cat]
            resps = clf.decision_function(kernel_traintest.T).ravel()
            outputs[:, icat] = resps

        end = time.time()
        print "Time: %s" % (end-start)

        return outputs

    def predict(self, data):
        """XXX: docstring for transform"""

        cats = self.categories

        outputs = self.transform(data)
        preds = outputs.argmax(1)
        lbls = [cats[pred] for pred in preds]
        return lbls

# ----------------------------------------------------------------------------
# -- Using shogun
# ----------------------------------------------------------------------------
class MultiOVALinearLibSVMShogun(object):

    def __init__(self, C=DEFAULT_C):
        """XXX: docstring for __init__"""

        self.C = C

    def fit(self, data, lbls):
        """XXX: docstring for fit"""

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        categories = np.unique(lbls)
        assert categories.size > 2

        ntrain = len(lbls)

        assert data.shape[0] == ntrain

        data = data.copy()

        data.shape = ntrain, -1

        print ">>> Computing normalization vectors"
        start = time.time()
        fmean = data.mean(0)
        fstd = data.std(0)
        np.putmask(fstd, fstd==0, 1)
        end = time.time()
        print "Time: %s" % (end-start)

        print ">>> Normalizing training data"
        # XXX: use scikits...Scaler
        start = time.time()
        data -= fmean
        data /= fstd
        end = time.time()
        print "Time: %s" % (end-start)

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        print ">>> Computing traintrain linear kernel"
        start = time.time()
        print data.shape
        kernel_traintrain = np.dot(data, data.T)
        ktrace = kernel_traintrain.trace()
        ktrace = ktrace != 0 and ktrace or 1
        kernel_traintrain /= ktrace
        end = time.time()
        print "Time: %s" % (end-start)

        print ">>> Set traintrain custom kernel"
        start = time.time()
        customkernel = Kernel.CustomKernel()
        customkernel.set_full_kernel_matrix_from_full(
                kernel_traintrain.T.astype('float64'))
        end = time.time()
        print "Time: %s" % (end-start)

        print ">>> Train LibSVMs (C=%e)" % self.C
        start = time.time()

        cat_index = {}

        alphas = {}
        support_vectors = {}
        biases = {}

        for icat, cat in enumerate(categories):
            print "> [%d] positive label: '%s'" % (icat+1, cat)
            ltrain = np.zeros(len(lbls))
            ltrain[lbls != cat] = -1
            ltrain[lbls == cat] = +1
            ltrain = ltrain.astype(np.float64)
            current_labels = Features.Labels(ltrain)
            svm = Classifier.LibSVM(self.C, customkernel, current_labels)
            assert(svm.train())
            alphas[cat] = svm.get_alphas()
            svs = svm.get_support_vectors()
            support_vectors[cat] = svs
            biases[cat] = svm.get_bias()
            cat_index[cat] = icat

        end = time.time()
        print "Time: %s" % (end-start)

        self._train_data = data
        self._ktrace = ktrace
        self._fmean = fmean
        self._fstd = fstd
        self._support_vectors = support_vectors
        self._alphas = alphas
        self._biases = biases

        self.categories = categories

    def transform(self, data):

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        ntest = len(data)

        data = data.copy()

        data.shape = ntest, -1

        print ">>> Normalizing testing data"
        start = time.time()
        data -= self._fmean
        data /= self._fstd
        end = time.time()
        print "Time: %s" % (end-start)

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        print ">>> Computing traintest linear kernel"
        start = time.time()
        kernel_traintest = np.dot(self._train_data, data.T)

        assert not np.isnan(kernel_traintest).any()
        assert not np.isinf(kernel_traintest).any()

        kernel_traintest /= self._ktrace

        assert not np.isnan(kernel_traintest).any()
        assert not np.isinf(kernel_traintest).any()

        end = time.time()
        print "Time: %s" % (end-start)

        print ">>> Collecting %d testing outputs" % ntest
        start = time.time()
        categories = self.categories
        support_vectors = self._support_vectors
        alphas = self._alphas
        biases = self._biases
        outputs = np.zeros((ntest, len(categories)), dtype='float32')

        print "Predicting testing data ..."
        for icat, cat in enumerate(categories):
            index_sv = support_vectors[cat]
            resps = np.dot(alphas[cat],
                           kernel_traintest[index_sv]) + biases[cat]
            outputs[:, icat] = resps

        #cat = categories[0]
        #for point in xrange(ntest):
            #index_sv = support_vectors[cat]
            #resp = np.dot(alphas[cat],
                          #kernel_traintest[index_sv, point]) \
                          #+ biases[cat]
            #outputs[point] = resp

        end = time.time()
        print "Time: %s" % (end-start)

        return outputs

    def predict(self, data):
        """XXX: docstring for transform"""

        cats = self.categories

        outputs = self.transform(data)
        preds = outputs.argmax(1)
        lbls = [cats[pred] for pred in preds]
        return lbls
        raise


        preds = np.sign(outputs).ravel()


        lbls = np.empty(preds.shape, dtype=cats.dtype)
        lbls[preds>=0] = cats[0]
        lbls[preds<0] = cats[1]

        return lbls

