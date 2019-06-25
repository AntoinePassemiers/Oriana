# -*- coding: utf-8 -*-
# gap.py
# author : Antoine Passemiers

from oriana import Dimensions, Parameter
from oriana.nodes import Poisson, Gamma, Bernoulli, Multinomial
from oriana.utils import inverse_digamma, sigmoid, logit
from oriana.nodes import Einsum, Multiply, Transpose

import numpy as np
from sklearn.decomposition import NMF


class GaP:

    def __init__(self, cmatrix, k=2, tau=0.5, use_factors=True):

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
        self.initialize_variational_parameters(use_factors=use_factors, tau=tau)
        self.initialize_prior_hyper_parameters()

    def build_model(self):

        # Initialize node U
        self.alpha1 = Parameter(np.random.gamma(2., size=self.k))
        self.alpha2 = Parameter(np.ones(self.k))
        self.U = Gamma(self.alpha1, self.alpha2, self.dims('n,k ~ s,d'), name='U')

        # Initialize node Vprime
        self.beta1 = Parameter(np.random.gamma(2., size=self.k))
        self.beta2 = Parameter(np.ones(self.k))
        self.V = Gamma(self.beta1, self.beta2, self.dims('m,k ~ s,d'), name='V')

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

        # Initialize node U_q
        self.a1 = Parameter(np.random.gamma(2., size=(self.n, self.k)))
        self.a2 = Parameter(np.ones((self.n, self.k)))
        self.U_q = Gamma(self.a1, self.a2, self.dims('n,k ~ d,d'))

        # Initialize node V_q
        self.b1 = Parameter(np.random.gamma(2., size=(self.m, self.k)))
        self.b2 = Parameter(np.ones((self.m, self.k)))
        self.V_q = Gamma(self.b1, self.b2, self.dims('m,k ~ d,d'))

    def initialize_variational_parameters(self, use_factors=True, tau=0.5):

        # Initialize parameters of Z_q
        r = np.random.rand(self.n, self.m, self.k)
        self.r[:] = r / r.sum(axis=2)[..., np.newaxis]

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
        V_hat = self.V_q.mean()
        log_U_hat = self.U_q.meanlog()
        log_V_hat = self.V_q.meanlog()

        # Initialize parameters of U
        self.alpha1[:] = np.ones(self.k) * 16.
        self.alpha2[:] = np.ones(self.k) * 4.

        # Initialize parameters of Vprime
        self.beta1[:] = np.ones(self.k) * 16.
        self.beta2[:] = np.ones(self.k) * 4.

    def update_variational_parameters(self):

        assert((0 <= self.r[:]).all() and (self.r[:] <= 1).all())

        # Compute expectations
        U_hat = self.U_q.mean()
        V_hat = self.V_q.mean()
        log_U_hat = self.U_q.meanlog()
        log_V_hat = self.V_q.meanlog()
        Z_hat = self.Z_q.mean()

        # Update nodes
        self.U[:] = U_hat
        self.V[:] = V_hat
        self.Z[:] = Z_hat
        self.UV.forward()

        # Update parameters of U_q
        self.a1[:] = self.alpha1[:] + Z_hat.sum(axis=1)
        self.a2[:] = self.alpha2[:] + V_hat.sum(axis=0)
        assert((self.a1[:] > 0).all())

        # Update parameters of Vprime_q
        print(self.b1[:].shape)
        self.b1[:] = self.beta1[:] + Z_hat.sum(axis=0)
        self.b2[:] = self.beta2[:] + U_hat.sum(axis=0)
        assert((self.b1[:] > 0).all())

        # Update parameters of Z_q
        log_sum = log_U_hat.reshape(self.n, 1, self.k) + log_V_hat.reshape(1, self.m, self.k)
        self.r[:] = np.exp(log_sum)
        norm = self.r[:].sum(axis=2)
        indices = (norm != 0.)
        self.r[indices] /= norm[indices][..., np.newaxis]
        assert((0 <= self.r[:]).all() and (self.r[:] <= 1).all())

    def update_prior_hyper_parameters(self):

        # Compute expectations
        U_hat = self.U_q.mean()
        V_hat = self.V_q.mean()
        log_U_hat = self.U_q.meanlog()
        log_V_hat = self.V_q.meanlog()

        # Update parameters of node U
        self.alpha1[:] = inverse_digamma(np.log(self.alpha2[:]) + np.mean(log_U_hat, axis=0))
        self.alpha2[:] = self.alpha1[:] / np.mean(U_hat, axis=0)

        # Update parameters of node Vprime
        self.beta1[:] = inverse_digamma(np.log(self.beta2[:]) + np.mean(log_V_hat, axis=0))
        self.beta2[:] = self.beta1[:] / np.mean(V_hat, axis=0)

    def step(self):
        self.update_variational_parameters() # E-step
        self.update_prior_hyper_parameters() # M-step

    def reconstruction_deviance(self):
        X = self.X.asarray()
        U = self.U.asarray()
        V = self.V.asarray()

        Lambda_mat = np.dot(U, V.T)

        print(np.round(Lambda_mat).astype(np.int))

        Lambda = Parameter(X[:])
        X_node = Poisson(Lambda, self.dims('n,m ~ d,d'), name='X')
        ll_X_given_X = X_node.loglikelihood()

        Lambda[:] = Lambda_mat
        ll_X_given_UV = X_node.loglikelihood()

        return -2. * (ll_X_given_UV - ll_X_given_X)

    def frobenius_norm(self):
        Lambda = np.dot(self.U.asarray(), self.V.asarray().T)
        frob = np.sqrt(((Lambda.flatten() - self.X.asarray().flatten()) ** 2.).sum())
        return frob

    def loglikelihood(self):
        self.UV.forward()
        ll = 0.
        ll += self.U.loglikelihood()
        ll += self.V.loglikelihood()
        ll += self.Z.loglikelihood()
        return ll
