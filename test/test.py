# -*- coding: utf-8 -*-
# test.py
# author : Antoine Passemiers

from oriana.utils import digamma, inverse_digamma, sigmoid, logit

import numpy as np
from numpy.testing import assert_almost_equal


def test_sigmoid():
    x = np.asarray([-2.3, 1.5, 0.45, -0.78, 5.3, -.2, 0.])
    assert_almost_equal(logit(sigmoid(x)), x)


def test_logit():
    x = np.asarray([0.45, 0.001, 0.9987, 0.63, 0.745, 0.521, 0.32])
    assert_almost_equal(sigmoid(logit(x)), x)


def test_digamma():
    x = np.asarray([0.54, 6.2, 1.2, 0.3, 7.9, 4.5, 2.1])
    y = inverse_digamma(digamma(x))
    assert_almost_equal(x, y)


def test_digamma_inverse():
    x = np.asarray([0.54, 6.2, 1.2, 0.3, 7.9, 4.5, 2.1])
    y = digamma(inverse_digamma(x))
    assert_almost_equal(x, y)    

