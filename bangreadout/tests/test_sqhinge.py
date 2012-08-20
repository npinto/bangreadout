#!/usr/bin/env python

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>

# Licence: BSD

import numpy as np
from bangreadout.sqhinge import AverageLBFGSSqHingeClassifier


def test_on_simple_hyperplane():

    np.random.seed(42)

    # -- instantiate a Classifier
    clf = AverageLBFGSSqHingeClassifier(100)

    # -- a random unit vector
    vec = np.random.randn(100).astype(np.float32)
    vec /= np.linalg.norm(vec)

    # -- fake data matrix [n_samples, n_features]
    X = np.random.randn(100000, 100).astype(np.float32)

    # -- fake ground truth
    Y = np.sign(np.dot(X, vec))

    # -- fit to the data
    clf.partial_fit(X, Y)

    # -- fake testing data
    X_tst = np.random.randn(10000, 100).astype(np.float32)

    # -- fake testing ground truth
    Y_tst = np.sign(np.dot(X_tst, vec))

    # -- prediction of the Classifier on the testing data
    Y_pred = clf.predict(X_tst)

    # -- making sure that the Classifier did well
    n_errors = np.abs(Y_tst - np.where(Y_pred <= 0, -1, 1)).sum()

    assert n_errors <= int(0.01 * Y_tst.size)
