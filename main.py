# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

from oriana import Dimensions, Parameter
from oriana.nodes import Poisson, Gamma, Bernoulli, Multinomial
from oriana.utils import inverse_digamma, sigmoid, logit
from oriana.nodes import Einsum, Multiply, Transpose
from oriana.inference import VariationalDistribution
from oriana.singlecell import CountMatrix

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')



def reconstruction_deviance(X, U, V):
    Lambda = np.dot(U, V.T)

    A = np.empty_like(X)
    valid = (X != 0)
    A[~valid] = 0
    #Lambda[Lambda == 0] = 1e-15 # TODO
    A[valid] = X[valid] * np.log(X[valid] / Lambda[valid])

    return (A - X + Lambda).sum()


if __name__ == '__main__':

    """
    filepath = os.path.join(DATA_FOLDER, 'llorens.csv')
    counts = CountMatrix.from_csv(filepath).T.as_array()
    print('Shape of X: %s' % str(counts.shape))
    """

    # TODO: parse it from csv file
    # once everything works
    counts = np.array(
                [[ 0,  0,  1,  0,  0],
                 [ 2,  0,  2,  3,  0],
                 [ 3,  0,  1,  9,  0],
                 [ 0,  1,  2,  1,  0],
                 [ 4,  0,  9,  8,  1],
                 [ 3,  0,  6,  1,  0],
                 [12,  0,  2, 26,  0],
                 [ 6,  1, 12,  5,  2],
                 [ 6,  0,  0,  6,  0],
                 [ 9,  0,  7, 23,  0]])


    n, m = counts.shape[0], counts.shape[1]
    p = m # TODO: ??
    k = 3
    dims = Dimensions({ 'n': n, 'm': m, 'p': p, 'k': k })


    pi_s = Parameter(np.random.rand(m))
    S = Bernoulli(pi_s, dims('m,k ~ d,s'), name='S')

    pi_d = Parameter(np.random.rand(p))
    D = Bernoulli(pi_d, dims('n,p ~ s,d'), name='D')
    # TODO: what to do with D then?

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

    X.buffer = counts


    ###################################
    # Define Variational Distribution #
    ###################################

    q = VariationalDistribution()

    r = np.random.rand(n, m, k)
    r = Parameter(r / r.sum(axis=2)[..., np.newaxis])
    Z_q = Multinomial(X, r, dims('n,m,k ~ d,d,c'))
    q.add_partition(Z, Z_q)

    tau = 0.5 # TODO
    p_s = Parameter(np.full((m, k), tau))
    S_q = Bernoulli(p_s, dims('m,k ~ d,d'))
    q.add_partition(S, S_q)

    p_d = Parameter((X[:] > 0).astype(np.int))
    D_q = Bernoulli(p_d, dims('n,p ~ d,d'))
    q.add_partition(D, D_q)

    a1 = Parameter(np.random.gamma(2., size=(n, k)))
    a2 = Parameter(np.ones((n, k)))
    U_q = Gamma(a1, a2, dims('n,k ~ d,d'))
    q.add_partition(U, U_q)

    b1 = Parameter(np.random.gamma(2., size=(m, k)))
    b2 = Parameter(np.ones((m, k)))
    Vprime_q = Gamma(b1, b2, dims('m,k ~ d,d'))
    q.add_partition(Vprime, Vprime_q)



    ##################
    # Initialization #
    ##################

    model = NMF(n_components=k)
    U[:] = model.fit_transform(X[:])
    Vprime[:] = model.components_.T

    pi_d[:] = np.mean(p_d[:], axis=0)
    pi_s[:] = np.mean(p_s[:], axis=1)



    for iteration in range(10):

        U_hat = U_q.mean()
        Vprime_hat = Vprime.mean()

        print(a1[:] / a2[:])
        print(U_hat)


        divergence = reconstruction_deviance(X.asarray(), U_hat, Vprime_hat) # TODO
        print('Iteration %i - Bregman divergence: %f' % (iteration + 1, divergence))

        ####################
        # Stationary point #
        ####################

        log_U_hat = U_q.meanlog()
        log_Vprime_hat = Vprime_q.meanlog()
        D_hat = D_q.mean()
        S_hat = S_q.mean()
        Z_hat = Z_q.mean()

        a1[:] = alpha1[:] + np.einsum('ij,jk,ijk->ik', D_hat, S_hat, Z_hat)
        a2[:] = alpha1[:] + np.einsum('ij,jk,jk->ik', D_hat, S_hat, Vprime_hat)

        b1[:] = beta1[:] + S_hat * np.einsum('ij,ijk->jk', D_hat, Z_hat)
        b2[:] = beta2[:] + S_hat * np.einsum('ij,ik->jk', D_hat, U_hat)

        S_tilde = (p_s[:] > tau)
        log_sum = log_U_hat.reshape(n, 1, k) + log_Vprime_hat.reshape(1, m, k)
        r[:] = S_tilde[np.newaxis, ...] * np.exp(log_sum)
        norm = r[:].sum(axis=2)
        indices = (norm != 0.)
        r[indices] /= norm[indices][..., np.newaxis]

        p_d[:] = sigmoid(logit(pi_d[:])[np.newaxis, ...] \
                - np.einsum('jk,ik,jk->ij', S_hat, U_hat, Vprime_hat))

        p_s[:] = sigmoid(logit(pi_s[:])[..., np.newaxis] \
                - np.einsum('ij,ik,jk->jk', D_hat, U_hat, Vprime_hat) \
                + np.einsum('ij,ijk,ijk->jk', D_hat, Z_hat, log_sum))


        ###########################
        # Update hyper-parameters #
        ###########################

        new_alpha1 = inverse_digamma(np.log(alpha2[:]) + np.mean(log_U_hat, axis=0))
        new_alpha2 = alpha1[:] / np.mean(U_hat, axis=0)
        alpha1[:], alpha2[:] = new_alpha1, new_alpha2

        new_beta1 = inverse_digamma(np.log(beta2[:]) + np.mean(log_Vprime_hat, axis=0))
        new_beta2 = beta1[:] / np.mean(Vprime_hat, axis=0)
        beta1[:], beta2[:] = new_beta1, new_beta2

        # For numerical purposes
        alpha2[:] = np.maximum(alpha2[:], 1e-15)
        beta2[:] = np.maximum(beta2[:], 1e-15)

        pi_d[:] = np.mean(p_d[:], axis=0)
        pi_s[:] = np.mean(p_s[:], axis=1)
