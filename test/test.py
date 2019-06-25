# -*- coding: utf-8 -*-
# test.py
# author : Antoine Passemiers

from oriana import Dimensions, Parameter
from oriana.nodes import Poisson, Gamma, Bernoulli, Multinomial, Multiply
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


def test_bernoulli_mean():
    p = Parameter([[0.02, 0.34], [0.62, 0.79]])
    dims = Dimensions({ 'n': 2, 'm': 2, 'k': 2 })
    bern = Bernoulli(p, dims('m,k ~ d,d'))
    x = np.asarray([[0.02, 0.34], [0.62, 0.79]])
    y = bern.mean()
    assert_almost_equal(x, y)


def test_multinomial_mean():
    n = Parameter([[0, 1], [3, 1]])
    p = Parameter([[[0.50, 0.50], [0.21, 0.79]],
                   [[0.43, 0.57], [0.89, 0.11]]])
    dims = Dimensions({ 'n': 2, 'm': 2, 'k': 2 })
    mult = Multinomial(n, p, dims('n,m,k ~ d,d,c'))
    x = mult.mean()
    y = p.asarray()
    print(y[:].shape, n[:].shape)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                y[i, j, k] *= n[i, j]
    assert_almost_equal(x, y)


def test_gamma_mean():
    alpha1 = Parameter([[2.1, 1.8], [0.7, 2.3]])
    alpha2 = Parameter(np.ones((2, 2)))
    dims = Dimensions({ 'n': 2, 'm': 2, 'k': 2 })
    gamma = Gamma(alpha1, alpha2, dims('n,m,k ~ d,s,d'))
    x = gamma.mean()
    y = np.asarray([[[2.1, 1.8], [2.1, 1.8]],
                    [[0.7, 2.3], [0.7, 2.3]]])
    assert_almost_equal(x, y)


def test_gamma_mean_log():
    alpha1 = Parameter([[2.1, 1.8], [0.7, 2.3]])
    alpha2 = Parameter(np.ones((2, 2)))
    dims = Dimensions({ 'n': 2, 'm': 2, 'k': 2 })
    gamma = Gamma(alpha1, alpha2, dims('n,m,k ~ d,s,d'))
    x = gamma.meanlog()
    y = digamma(np.asarray([[[2.1, 1.8], [2.1, 1.8]],
                    [[0.7, 2.3], [0.7, 2.3]]]))
    assert_almost_equal(x, y)


def test_node_forward():
    n = Parameter([[0, 1], [3, 1]])
    p = Parameter([[[0.50, 0.50], [0.21, 0.79]],
                   [[0.43, 0.57], [0.89, 0.11]]])
    dims = Dimensions({ 'n': 2, 'm': 2, 'k': 2 })
    mult1 = Multinomial(n, p, dims('n,m,k ~ d,d,c'))
    q = Parameter(1-p.asarray())
    mult2 = Multinomial(n, q, dims('n,m,k ~ d,d,c'))
    prod = Multiply(mult1, mult2)
    prod.forward()
    mult1.sample()
    mult2.sample()
    assert(not np.allclose(prod[:], mult1[:] * mult2[:]))
    prod.forward()
    assert_almost_equal(prod[:], mult1[:] * mult2[:])
