# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

from oriana import Dimensions, Parameter
from oriana.nodes import Poisson, Gamma, Bernoulli
from oriana.nodes import Einsum
from oriana.singlecell import CountMatrix

import os
import numpy as np
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')


if __name__ == '__main__':
    """
    filepath = os.path.join(DATA_FOLDER, 'llorens.csv')
    X = CountMatrix.from_csv(filepath).T.as_array()

    print('Shape of X: %s' % str(X.shape))
    n, m = X.shape[:2]
    """

    n = 10
    m = p = 5
    K = 3
    dims = Dimensions({ 'n': n, 'm': m, 'p': p, 'K': K })


    pi_s = Parameter([0., 0., 1., 0.5, 0.5])
    mapping = dims('m,K ~ +,-')
    S = Bernoulli(pi_s, mapping, name='S')

    alpha = Parameter(np.random.rand(K, 2))
    mapping = dims('n,K ~ -,+')
    U = Gamma(alpha, mapping, name='U')

    beta = Parameter(np.random.rand(K, 2))
    mapping = dims('m,K ~ -,+')
    Vprime = Gamma(beta, mapping, name='Vprime')

    print(S.sample())
    print(U.sample())

    V = Einsum('nk,mk->nmk', [U, Vprime])

    print(V.sample())
