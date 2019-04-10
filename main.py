# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

from oriana import Dimensions, Parameter
from oriana.nodes import Poisson, Gamma, Bernoulli
from oriana.nodes import Einsum, Multiply
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


    pi_s = Parameter(np.random.rand(m))
    S = Bernoulli(pi_s, dims('m,K ~ +,-'), name='S')

    alpha1 = Parameter(np.random.rand(K))
    alpha2 = Parameter(np.random.rand(K))
    U = Gamma(alpha1, alpha2, dims('n,K ~ -,+'), name='U')

    beta1 = Parameter(np.random.rand(K))
    beta2 = Parameter(np.random.rand(K))
    Vprime = Gamma(beta1, beta2, dims('m,K ~ -,+'), name='Vprime')

    V = Multiply(S, Vprime)

    UV = Einsum('nk,mk->nmk', U, V)
    Z = Poisson(UV, dims('n,m,K ~ +,+,+'), name='Z')

    print('\nModel summary')
    print('-------------\n')
    print(S)
    print(U)
    print(Vprime)
    print(Z)
    print()

    print(Z.sample()[0])
