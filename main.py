# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

from oriana import Dimensions, Parameter
from oriana.nodes import Poisson, Gamma, Bernoulli, Multinomial
from oriana.nodes import Einsum, Multiply, Transpose
from oriana.inference import VariationalDistribution
from oriana.singlecell import CountMatrix

import os
import numpy as np
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')



def reconstruction_deviance(X, U, V, dims):
    X, U, V = X.buffer, U.buffer, V.buffer
    Lambda = np.dot(U, V.T)

    A = np.empty_like(X)
    valid = (X != 0)
    A[~valid] = 0
    A[valid] = X[valid] * np.log(X[valid] / Lambda[valid])

    return (A - X + Lambda).sum()


if __name__ == '__main__':
    """
    filepath = os.path.join(DATA_FOLDER, 'llorens.csv')
    X = CountMatrix.from_csv(filepath).T.as_array()

    print('Shape of X: %s' % str(X.shape))
    n, m = X.shape[:2]
    """

    n = 10
    m = p = 5
    k = 3
    dims = Dimensions({ 'n': n, 'm': m, 'p': p, 'k': k })


    pi_s = Parameter(np.random.rand(m))
    S = Bernoulli(pi_s, dims('m,k ~ d,s'), name='S')

    alpha1 = Parameter(np.random.rand(k))
    alpha2 = Parameter(np.random.rand(k))
    U = Gamma(alpha1, alpha2, dims('n,k ~ s,d'), name='U')

    beta1 = Parameter(np.random.rand(k))
    beta2 = Parameter(np.random.rand(k))
    Vprime = Gamma(beta1, beta2, dims('m,k ~ s,d'), name='Vprime')

    V = Multiply(S, Vprime)

    UV = Einsum('nk,mk->nmk', U, V)
    Z = Poisson(UV, dims('n,m,k ~ d,d,d'), name='Z')

    X = Einsum('nmk->nm', Z)

    print('\nModel summary')
    print('-------------\n')
    print(S)
    print(U)
    print(Vprime)
    print(Z)
    print('')


    X.sample(recursive=True)

    print(X.buffer)

    rho = np.random.rand(n, m, k)
    rho = Parameter(rho / rho.sum(axis=2)[..., None])
    Zq = Multinomial(X, rho, dims('n,m,k ~ d,d,c'))
    Zq.sample()

    #print(Zq.buffer)
    print(Zq.buffer.shape)

    print(Zq.logpdfs().shape)


    """
    q = VariationalDistribution()
    q.add_partition(U, )

    q.add_partition(Z, madafack)

    Dq = VariationalDistribution(D)
    Vq = VariationalDistribution    
    """

    divergence = reconstruction_deviance(X, U, V, dims)
    print('Bregman divergence: %f' % divergence)
