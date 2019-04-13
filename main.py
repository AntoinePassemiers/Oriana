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



def reconstruction_deviance(X, U, V, dims):
    U.fix(); V.fix()

    # Set Lambda = U.V^T
    Lambda = Einsum('nk,mk->nm', U, V)
    Lambda.forward() 

    # Compute p(X | Lambda = U.V^T)
    X_hat = Poisson(Lambda, dims('n,m ~ d,d'))
    logp_reconstructed = X_hat.logp()

    # Set Lambda = X
    Lambda[:] = X[:] # Set Lambda = X

    # Compute p(X | Lambda = X)
    logp = X_hat.logp()

    U.unfix(); V.unfix()

    return -2 * (logp_reconstructed - logp)


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
    X.unfix(recursive=True)

    print('Log-likelihood of matrix S: %f' % S.logp())
    print('Log-likelihood of matrix U: %f' % U.logp())
    print('')

    divergence = reconstruction_deviance(X, U, V, dims)
    print('Bregman divergence: %f' % divergence)
