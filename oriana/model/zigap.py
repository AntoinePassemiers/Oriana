# -*- coding: utf-8 -*-
# zigap.py
# author : Antoine Passemiers

from oriana import Dimensions, Parameter
from oriana.nodes import Poisson, Gamma, Bernoulli, Multinomial
from oriana.utils import inverse_digamma, sigmoid, logit
from oriana.nodes import Einsum, Multiply, Transpose

import numpy as np
from sklearn.decomposition import NMF


class ZIGaP:

    def __init__(self, cmatrix, k=2, tau=0.5, use_factors=False):

        # Count matrix
        self.cmatrix = cmatrix

        # Define dimensions
        self.k = k
        self.n = cmatrix.shape[0]
        self.m = self.p = cmatrix.shape[1]
        self.dims = Dimensions({ 'n': self.n, 'm': self.m, 'p': self.p, 'k': self.k })

        # Define model
        self.build_model()

        # Define corresponding variational distribution
        self.define_variational_distribution(tau=tau)

        # Initialize parameters
        self.tau = tau
        self.use_factors = use_factors
        model = NMF(n_components=self.k)
        self.U[:] = model.fit_transform(self.X[:])
        self.V.buffer = model.components_.T
        self.Vprime[:] = model.components_.T
        self.initialize_variational_parameters(use_factors=use_factors, tau=tau)
        self.initialize_prior_hyper_parameters()

    def build_model(self):
        # Initialize node S
        self.pi_s = Parameter(np.random.rand(self.m))
        self.S = Bernoulli(self.pi_s, self.dims('m,k ~ d,s'), name='S')

        # Initialize node D
        self.pi_d = Parameter(np.random.rand(self.p))
        self.D = Bernoulli(self.pi_d, self.dims('n,p ~ s,d'), name='D')

        # Initialize node U
        self.alpha1 = Parameter(np.random.gamma(2., size=self.k))
        self.alpha2 = Parameter(np.ones(self.k))
        self.U = Gamma(self.alpha1, self.alpha2, self.dims('n,k ~ s,d'), name='U')

        # Initialize node Vprime
        self.beta1 = Parameter(np.random.gamma(2., size=self.k))
        self.beta2 = Parameter(np.ones(self.k))
        self.Vprime = Gamma(self.beta1, self.beta2, self.dims('m,k ~ s,d'), name='Vprime')

        # Initialize node V
        self.V = Multiply(self.S, self.Vprime)

        # Initialize node UV
        self.UV = Einsum('nk,mk->nmk', self.U, self.V)

        # Initialize node Z
        self.Z = Poisson(self.UV, self.dims('n,m,k ~ d,d,d'), name='Z')

        # Initialize node X
        self.X = Einsum('nmk->nm', self.Z)
        self.X.buffer = self.cmatrix.as_array()

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
        self.p_s[:] = np.full((self.m, self.k), tau)

        # Initialize parameters of D_q
        self.p_d[:] = (self.X[:] > 0).astype(np.float)

        # Initialize parameters of U_q
        if use_factors:
            self.a1[:] = self.U[:]
        else:
            self.a1[:] = np.random.gamma(2., size=(self.n, self.k))
        self.a2[:] = np.ones((self.n, self.k))

        # Initialize parameters of Vprime_q
        if use_factors:
            self.b1[:] = self.V[:]
        else:
            self.b1[:] = np.random.gamma(2., size=(self.m, self.k))
        self.b2[:] = np.ones((self.m, self.k))

    def initialize_prior_hyper_parameters(self):
        # TODO: init alpha and beta

        self.pi_d[:] = np.mean(self.p_d[:], axis=0)
        self.pi_s[:] = np.mean(self.p_s[:], axis=1)

    def update_variational_parameters(self):

        # Compute expectations
        U_hat = self.U_q.mean()
        S_hat = self.S_q.mean()
        Vprime_hat = self.Vprime_q.mean()
        V_hat = Vprime_hat * S_hat
        log_U_hat = self.U_q.meanlog()
        log_Vprime_hat = self.Vprime_q.meanlog()
        D_hat = self.D_q.mean()
        Z_hat = self.Z_q.mean()

        # Update factor matrices
        self.U[:] = U_hat
        self.Vprime[:] = Vprime_hat
        self.V[:] = V_hat

        # Update parameters of U_q
        self.a1[:] = self.alpha1[:] + np.einsum('ij,jk,ijk->ik', D_hat, S_hat, Z_hat)
        self.a2[:] = self.alpha2[:] + np.einsum('ij,jk,jk->ik', D_hat, S_hat, Vprime_hat)

        # Update parameters of Vprime_q
        self.b1[:] = self.beta1[:] + S_hat * np.einsum('ij,ijk->jk', D_hat, Z_hat)
        self.b2[:] = self.beta2[:] + S_hat * np.einsum('ij,ik->jk', D_hat, U_hat)

        # Update parameters of Z_q
        S_tilde = (self.p_s[:] > self.tau)
        log_sum = log_U_hat.reshape(self.n, 1, self.k) + log_Vprime_hat.reshape(1, self.m, self.k)
        self.r[:] = S_tilde[np.newaxis, ...] * np.exp(log_sum)
        norm = self.r[:].sum(axis=2)
        indices = (norm != 0.)
        self.r[indices] /= norm[indices][..., np.newaxis]

        # Update parameters of D_q
        self.p_d[:, self.pi_d[:] == 0] = 0
        self.p_d[:, self.pi_d[:] == 1] = 1
        mask = np.logical_and(0 < self.pi_d[:], self.pi_d[:] < 1)
        self.p_d[:, mask] = sigmoid(logit(self.pi_d[:])[np.newaxis, ...] \
                - np.einsum('jk,ik,jk->ij', S_hat, U_hat, Vprime_hat))[:, mask]
        self.p_d[self.X[:] == 0] = 0

        # Update parameters of S_q
        self.p_s[self.pi_s[:] == 0] = 0
        self.p_s[self.pi_s[:] == 1] = 1
        mask = np.logical_and(0 < self.pi_s[:], self.pi_s[:] < 1)
        tmp = (np.einsum('ij,ijk,ijk->ijk', D_hat, Z_hat, log_sum) - np.einsum('ij,ik,jk->ijk', D_hat, U_hat, Vprime_hat)).sum(axis=0)
        self.p_s[mask] = sigmoid(logit(self.pi_s[:])[..., np.newaxis] - tmp)[mask]


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
        V = self.V.asarray()
        Lambda = np.dot(U, V.T)

        print(Lambda)

        A = np.empty_like(X)
        valid = (X != 0)
        A[~valid] = 0
        #Lambda[Lambda == 0] = 1e-15 # TODO
        A[valid] = X[valid] * np.log(X[valid] / Lambda[valid])

        return (A - X + Lambda).sum()
