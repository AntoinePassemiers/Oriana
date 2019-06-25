# -*- coding: utf-8 -*-
# gap.py
# author : Antoine Passemiers

from oriana import Parameter
from oriana.models import FactorModel
from oriana.nodes import Poisson, Gamma, Bernoulli, Multinomial
from oriana.utils import inverse_digamma, sigmoid, logit

import numpy as np


class GaP(FactorModel):

    def __init__(self, *args, **kwargs):
        FactorModel.__init__(self, *args, **kwargs)

    def build_model(self):
        pass

    def build_u_node(self):
        self.alpha1 = Parameter(np.random.gamma(2., size=self.k))
        self.alpha2 = Parameter(np.ones(self.k))
        return Gamma(self.alpha1, self.alpha2, self.dims('n,k ~ s,d'), name='U')

    def build_v_node(self):
        self.beta1 = Parameter(np.random.gamma(2., size=self.k))
        self.beta2 = Parameter(np.ones(self.k))
        return Gamma(self.beta1, self.beta2, self.dims('m,k ~ s,d'), name='V')

    def build_x_node(self, cmatrix, UV):
        X = Poisson(UV, self.dims('n,m ~ d,d'), name='X')
        X.buffer = cmatrix.as_array()
        return X

    def define_variational_distribution(self):

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

    def initialize_variational_parameters(self):

        # Initialize parameters of Z_q
        r = np.ones((self.n, self.m, self.k))
        self.r[:] = r / r.sum(axis=2)[..., np.newaxis]

        # Initialize parameters of U_q
        if self.use_factors:
            self.a1[:] = self.U[:]
        else:
            self.a1[:] = np.random.gamma(1., size=(self.n, self.k))
        self.a2[:] = np.ones((self.n, self.k))

        # Initialize parameters of Vprime_q
        if self.use_factors:
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

        # Compute expectations
        U_hat = self.U_q.mean()
        V_hat = self.V_q.mean()
        log_U_hat = self.U_q.meanlog()
        log_V_hat = self.V_q.meanlog()
        Z_hat = self.Z_q.mean()

        # Update nodes
        self.U[:] = U_hat
        self.V[:] = V_hat
        self.UV.forward()

        # Update parameters of U_q
        self.a1[:] = self.alpha1[:] + Z_hat.sum(axis=1)
        self.a2[:] = self.alpha2[:] + V_hat.sum(axis=0)
        self.a1[:] = np.maximum(1e-15, self.a1[:])

        # Update parameters of Vprime_q
        print(self.b1[:].shape)
        self.b1[:] = self.beta1[:] + Z_hat.sum(axis=0)
        self.b2[:] = self.beta2[:] + U_hat.sum(axis=0)
        self.b1[:] = np.maximum(1e-15, self.b1[:])

        # Update parameters of Z_q
        log_sum = log_U_hat.reshape(self.n, 1, self.k) + log_V_hat.reshape(1, self.m, self.k)
        max_values = log_sum.max(axis=2)
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
