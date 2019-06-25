# -*- coding: utf-8 -*-
# zigap.py
# author : Antoine Passemiers

from oriana import Dimensions, Parameter
from oriana.models import GaP
from oriana.nodes import Poisson, Gamma, Bernoulli, Multinomial
from oriana.utils import inverse_digamma, sigmoid, logit
from oriana.nodes import Einsum, Multiply, Transpose

import numpy as np
from sklearn.decomposition import NMF


class SparseZIGaP(GaP):

    def __init__(self, *args, **kwargs):
        GaP.__init__(self, *args, **kwargs)

    def build_u_node(self):
        self.alpha1 = Parameter(np.random.gamma(2., size=self.k))
        self.alpha2 = Parameter(np.ones(self.k))
        return Gamma(self.alpha1, self.alpha2, self.dims('n,k ~ s,d'), name='U')

    def build_v_node(self):
        self.pi_s = Parameter(np.random.rand(self.m))
        self.S = Bernoulli(self.pi_s, self.dims('m,k ~ d,s'), name='S')
        self.beta1 = Parameter(np.random.gamma(2., size=self.k))
        self.beta2 = Parameter(np.ones(self.k))
        self.Vprime = Gamma(self.beta1, self.beta2, self.dims('m,k ~ s,d'), name='Vprime')
        U = Multiply(self.S, self.Vprime)
        U.forward()
        return U

    def build_x_node(self, cmatrix, UV):
        self.pi_d = Parameter(np.random.rand(self.p))
        self.D = Bernoulli(self.pi_d, self.dims('n,p ~ s,d'), name='D')
        self.L = Poisson(UV, self.dims('n,m ~ d,d'), name='X')
        X = Multiply(self.L, self.D)
        X.buffer = cmatrix.as_array()
        return X

    def define_variational_distribution(self, tau=0.5):

        # Initialize node Z_q
        r = np.random.rand(self.n, self.m, self.k)
        self.r = Parameter(r / r.sum(axis=2)[..., np.newaxis])
        self.Z_q = Multinomial(self.X, self.r, self.dims('n,m,k ~ d,d,c'))

        # Initialize node S_q
        self.p_s = Parameter(np.full((self.m, self.k), tau))
        self.S_q = Bernoulli(self.p_s, self.dims('m,k ~ d,d'))

        # Initialize node D_q
        self.p_d = Parameter((self.X[:] > 0).astype(np.float))
        self.D_q = Bernoulli(self.p_d, self.dims('n,p ~ d,d'))

        # Initialize node U_q
        self.a1 = Parameter(np.random.gamma(2., size=(self.n, self.k)))
        self.a2 = Parameter(np.ones((self.n, self.k)))
        self.U_q = Gamma(self.a1, self.a2, self.dims('n,k ~ d,d'))

        # Initialize node Vprime_q
        self.b1 = Parameter(np.random.gamma(2., size=(self.m, self.k)))
        self.b2 = Parameter(np.ones((self.m, self.k)))
        self.Vprime_q = Gamma(self.b1, self.b2, self.dims('m,k ~ d,d'))

    def initialize_variational_parameters(self, use_factors=True, tau=0.5):

        # Initialize parameters of Z_q
        r = np.random.rand(self.n, self.m, self.k)
        self.r[:] = r / r.sum(axis=2)[..., np.newaxis]

        # Initialize parameters of S_q
        self.p_s[:] = np.ones((self.m, self.k))

        # Initialize parameters of D_q
        self.p_d[:] = np.ones((self.n, self.m)) # (self.X[:] > 0).astype(np.float)

        # Initialize parameters of U_q
        if use_factors:
            self.a1[:] = self.U[:]
        else:
            self.a1[:] = np.random.gamma(1., size=(self.n, self.k))
        self.a2[:] = np.ones((self.n, self.k))

        # Initialize parameters of Vprime_q
        if use_factors:
            self.b1[:] = self.V[:]
        else:
            self.b1[:] = np.random.gamma(1., size=(self.m, self.k))
        self.b2[:] = np.ones((self.m, self.k))

    def initialize_prior_hyper_parameters(self):

        # Compute expectations
        U_hat = self.U_q.mean()
        Vprime_hat = self.Vprime_q.mean()
        log_U_hat = self.U_q.meanlog()
        log_Vprime_hat = self.Vprime_q.meanlog()

        # Initialize parameters of U
        self.alpha1[:] = np.ones(self.k) * 16.
        self.alpha2[:] = np.ones(self.k) * 4.

        # Initialize parameters of Vprime
        self.beta1[:] = np.ones(self.k) * 16.
        self.beta2[:] = np.ones(self.k) * 4.

        # Initialize parameters of D
        self.pi_d[:] = np.mean(self.p_d[:], axis=0)

        # Initialize parameters of S
        self.pi_s[:] = np.mean(self.p_s[:], axis=1)

    def update_variational_parameters(self):

        assert((0 <= self.r[:]).all() and (self.r[:] <= 1).all())

        # Compute expectations
        U_hat = self.U_q.mean()
        S_hat = self.S_q.mean()
        Vprime_hat = self.Vprime_q.mean()
        V_hat = S_hat * Vprime_hat
        log_U_hat = self.U_q.meanlog()
        log_Vprime_hat = self.Vprime_q.meanlog()
        D_hat = self.D_q.mean()
        Z_hat = self.Z_q.mean()

        # Update nodes
        self.U[:] = U_hat
        self.S[:] = (S_hat > self.tau).astype(np.float)
        self.Vprime[:] = Vprime_hat
        self.V[:] = V_hat
        self.D[:] = D_hat
        self.Z[:] = Z_hat
        self.UV.forward()

        # Update parameters of U_q
        assert((self.p_d[:] == D_hat).all())
        self.a1[:] = self.alpha1[:] + np.einsum('ij,jk,ijk->ik', D_hat, S_hat, Z_hat)
        self.a2[:] = self.alpha2[:] + np.dot(D_hat, V_hat)
        assert((self.a1[:] > 0).all())

        # Update parameters of Vprime_q
        self.b1[:] = self.beta1[:] + S_hat * np.einsum('ij,ijk->jk', D_hat, Z_hat)
        self.b2[:] = self.beta2[:] + S_hat * np.dot(D_hat.T, U_hat)
        assert((self.b1[:] > 0).all())

        # Update parameters of Z_q
        S_tilde = (self.p_s[:] > self.tau)
        log_sum = log_U_hat.reshape(self.n, 1, self.k) + log_Vprime_hat.reshape(1, self.m, self.k)
        self.r[:] = S_tilde[np.newaxis, ...] * np.exp(log_sum)
        norm = self.r[:].sum(axis=2)
        indices = (norm != 0.)
        self.r[indices] /= norm[indices][..., np.newaxis]
        assert((0 <= self.r[:]).all() and (self.r[:] <= 1).all())

        # Update parameters of D_q
        self.p_d[:] = sigmoid(logit(self.pi_d[:])[np.newaxis, ...] \
                - np.dot(U_hat, V_hat.T))
        self.p_d[:, self.pi_d[:] == 0] = 1e-10
        self.p_d[:, self.pi_d[:] == 1] = 1. - 1e-10
        self.p_d[self.X[:] != 0] = 1. - 1e-10
        assert((0 <= self.p_d[:]).all() and (self.p_d[:] <= 1).all())

        # Update parameters of S_q
        tmp = -np.nan_to_num(np.einsum('ij,ijk,ijk->jk', D_hat, Z_hat, log_sum))
        tmp += np.nan_to_num(np.dot(D_hat.T, U_hat) * Vprime_hat)
        self.p_s[:] = sigmoid(logit(self.pi_s[:])[..., np.newaxis] + tmp)
        self.p_s[:] = np.nan_to_num(self.p_s[:])
        self.p_s[self.pi_s[:] == 0] = 1e-10
        self.p_s[self.pi_s[:] == 1] = 1. - 1e-10
        assert((0 <= self.p_s[:]).all() and (self.p_s[:] <= 1).all())

    def update_prior_hyper_parameters(self):

        # Compute expectations
        U_hat = self.U_q.mean()
        Vprime_hat = self.Vprime_q.mean()
        log_U_hat = self.U_q.meanlog()
        log_Vprime_hat = self.Vprime_q.meanlog()

        # Update parameters of node U
        self.alpha1[:] = inverse_digamma(np.log(self.alpha2[:]) + np.mean(log_U_hat, axis=0))
        self.alpha2[:] = self.alpha1[:] / np.mean(U_hat, axis=0)

        # Update parameters of node Vprime
        self.beta1[:] = inverse_digamma(np.log(self.beta2[:]) + np.mean(log_Vprime_hat, axis=0))
        self.beta2[:] = self.beta1[:] / np.mean(Vprime_hat, axis=0)

        # Update parameters of node D
        self.pi_d[:] = np.mean(self.p_d[:], axis=0)

        # Update parameters of node S
        self.pi_s[:] = np.mean(self.p_s[:], axis=1)

    def step(self):
        self.update_variational_parameters() # E-step
        self.update_prior_hyper_parameters() # M-step

    def reconstruction_deviance(self):
        X = self.X.asarray()
        U = self.U.asarray()
        D = self.D.asarray()
        V = self.V.asarray()

        Lambda_mat = np.dot(U, V.T) * D

        print(np.round(Lambda_mat).astype(np.int))

        """
        A = np.empty_like(X)
        valid = (X != 0)
        A[~valid] = 0
        #Lambda[Lambda == 0] = 1e-15 # TODO
        A[valid] = X[valid] * np.log(X[valid] / Lambda[valid])
        return (A - X + Lambda).sum()
        """
        Lambda = Parameter(X[:])
        X_node = Poisson(Lambda, self.dims('n,m ~ d,d'), name='X')
        ll_X_given_X = X_node.loglikelihood()

        Lambda[:] = Lambda_mat
        ll_X_given_UV = X_node.loglikelihood()

        return -2. * (ll_X_given_UV - ll_X_given_X)

    def frobenius_norm(self):
        Lambda = np.dot(self.U.asarray(), self.V.asarray().T) * self.D.asarray()
        frob = np.sqrt(((Lambda.flatten() - self.X.asarray().flatten()) ** 2.).sum())
        return frob

    def loglikelihood(self):
        self.UV.forward()
        ll = 0.
        ll += self.D.loglikelihood()
        ll += self.U.loglikelihood()
        ll += self.Vprime.loglikelihood()
        ll += self.Z.loglikelihood()
        return ll
