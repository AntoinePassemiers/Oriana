# -*- coding: utf-8 -*-
# gap.py
# author : Antoine Passemiers

from oriana import Parameter
from oriana.models import FactorModel
from oriana.nodes import Poisson, Gamma, Bernoulli, Multinomial
from oriana.utils import inverse_digamma, sigmoid, logit

import numpy as np
import numba


class GaP(FactorModel):

    def __init__(self, *args, **kwargs):
        FactorModel.__init__(self, *args, **kwargs)

    def build_u_node(self):
        self.alpha1 = Parameter(np.ones(self.k))
        self.alpha2 = Parameter(np.ones(self.k))
        return Gamma(self.alpha1, self.alpha2, self.dims('n,k ~ s,d'), name='U')

    def build_v_node(self):
        self.beta1 = Parameter(np.ones(self.k))
        self.beta2 = Parameter(np.ones(self.k))
        return Gamma(self.beta1, self.beta2, self.dims('m,k ~ s,d'), name='V')

    def build_x_node(self, cmatrix, UV):
        X = Poisson(UV, self.dims('n,m ~ d,d'), name='X')
        X.buffer = cmatrix.as_array()
        return X

    def define_variational_distribution(self):

        # Initialize node U_q
        self.a1 = Parameter(np.random.gamma(2., size=(self.n, self.k)))
        self.a2 = Parameter(np.ones((self.n, self.k)))
        self.U_q = Gamma(self.a1, self.a2, self.dims('n,k ~ d,d'))

        # Initialize node V_q
        self.b1 = Parameter(np.random.gamma(2., size=(self.m, self.k)))
        self.b2 = Parameter(np.ones((self.m, self.k)))
        self.V_q = Gamma(self.b1, self.b2, self.dims('m,k ~ d,d'))

    def initialize_variational_parameters(self):

        # Initialize parameters of U_q
        if self.use_factors:
            self.a1[:] = self.U[:]
        else:
            self.a1[:] = np.random.gamma(1., size=(self.n, self.k))
        epsilon = max(np.max(self.U[:]), np.max(self.V[:]))
        #self.a1[:] += np.random.uniform(-epsilon, epsilon, size=self.a1[:].shape)
        self.a1[:] = np.maximum(1e-15, np.nan_to_num(self.a1[:]))
        self.a2[:] = np.ones((self.n, self.k))

        # Initialize parameters of Vprime_q
        if self.use_factors:
            self.b1[:] = self.V[:]
        else:
            self.b1[:] = np.random.gamma(1., size=(self.m, self.k))
        #self.b1[:] += np.random.uniform(-epsilon, epsilon, size=self.b1[:].shape)
        self.b1[:] = np.maximum(1e-15, np.nan_to_num(self.b1[:]))
        self.b2[:] = np.ones((self.m, self.k))

    @numba.jit('void(f4[:, :], f4[:, :], f4[:, :], f4[:, :], f4[:, :])')
    def compute_Z_q_expectations(Z_hat_i, Z_hat_j, log_U_hat, log_V_hat, X):
        Z_hat_i[:, :] = 0
        Z_hat_j[:, :] = 0
        n, p, latent_dim = log_U_hat.shape[0], log_V_hat.shape[0], log_U_hat.shape[1]
        for i in range(n):
            for j in range(p):
                exp_logsum = np.exp(log_U_hat[i, :] + log_V_hat[j, :])
                den = exp_logsum.sum()
                den = den if den > 0 else 1
                for k in range(latent_dim):
                    expectation = X[i, j] * exp_logsum[k] / den
                    Z_hat_j[j, k] += expectation
                    Z_hat_i[i, k] += expectation

    def update_variational_parameters(self):

        # Compute useful metrics based on the expectation of Z_q
        # E[Z_q] is memory-expensive, so it is itself not
        # explicitely computed.
        Z_hat_i = np.empty((self.n, self.k), dtype=np.float32)
        Z_hat_j = np.empty((self.m, self.k), dtype=np.float32)
        GaP.compute_Z_q_expectations(
                Z_hat_i,
                Z_hat_j,
                self.log_U_hat,
                self.log_V_hat,
                self.X[:].astype(np.float32))

        # Update parameters of U_q
        self.a1[:] = self.alpha1[np.newaxis, ...] + Z_hat_i
        self.a2[:] = self.alpha2[:] + self.V_hat.sum(axis=0)
        self.a1[:] = np.maximum(1e-15, np.nan_to_num(self.a1[:]))
        self.a2[:] = np.maximum(1e-15, np.nan_to_num(self.a2[:]))
        self.U_hat = self.U_q.mean()
        self.log_U_hat = self.U_q.meanlog()

        # Update parameters of Vprime_q
        self.b1[:] = self.beta1[np.newaxis, ...] + Z_hat_j
        self.b2[:] = self.beta2[:] + self.U_hat.sum(axis=0)
        self.b1[:] = np.maximum(1e-15, np.nan_to_num(self.b1[:]))
        self.b2[:] = np.maximum(1e-15, np.nan_to_num(self.b2[:]))
        self.V_hat = self.V_q.mean()
        self.log_V_hat = self.V_q.meanlog()

        # Update parameters of UV
        self.U[:] = self.U_hat
        self.V[:] = self.V_hat
        self.UV.forward()

    def update_prior_hyper_parameters(self):

        # Update parameters of node U
        self.alpha1[:] = inverse_digamma(np.log(self.alpha2[:]) + np.mean(self.log_U_hat, axis=0))
        self.alpha1[:] = np.maximum(1e-15, np.nan_to_num(self.alpha1[:]))
        self.alpha2[:] = self.alpha1[:] / np.mean(self.U_hat, axis=0)
        self.alpha2[:] = np.maximum(1e-15, np.nan_to_num(self.alpha2[:]))

        # Update parameters of node V
        self.beta1[:] = inverse_digamma(np.log(self.beta2[:]) + np.mean(self.log_V_hat, axis=0))
        self.beta1[:] = np.maximum(1e-15, np.nan_to_num(self.beta1[:]))
        self.beta2[:] = self.beta1[:] / np.mean(self.V_hat, axis=0)
        self.beta2[:] = np.maximum(1e-15, np.nan_to_num(self.beta2[:]))

    def update_expectations(self):
        self.U_hat = self.U_q.mean()
        self.V_hat = self.V_q.mean()
        self.log_U_hat = self.U_q.meanlog()
        self.log_V_hat = self.V_q.meanlog()
