# -*- coding: utf-8 -*-
# zigap.py
# author : Antoine Passemiers

from oriana import Dimensions, Parameter
from oriana.models import FactorModel, GaP
from oriana.nodes import Poisson, Gamma, Bernoulli, Multinomial
from oriana.utils import inverse_digamma, sigmoid, logit
from oriana.nodes import Einsum, Multiply, Transpose

import numba
import numpy as np


class ZIGaP(FactorModel):

    def __init__(self, *args, tau=0.5, **kwargs):
        self.tau = tau
        FactorModel.__init__(self, *args, **kwargs)

    def build_u_node(self):
        self.alpha1 = Parameter(np.random.gamma(2., size=self.k))
        self.alpha2 = Parameter(np.ones(self.k))
        return Gamma(self.alpha1, self.alpha2, self.dims('n,k ~ s,d'), name='U')

    def build_v_node(self):
        self.beta1 = Parameter(np.random.gamma(2., size=self.k))
        self.beta2 = Parameter(np.ones(self.k))
        return  Gamma(self.beta1, self.beta2, self.dims('m,k ~ s,d'), name='V')

    def build_x_node(self, cmatrix, UV):
        self.pi_d = Parameter(np.random.rand(self.p))
        self.D = Bernoulli(self.pi_d, self.dims('n,p ~ s,d'), name='D')
        self.L = Poisson(UV, self.dims('n,m ~ d,d'), name='X')
        X = Multiply(self.L, self.D)
        X.buffer = cmatrix.as_array()
        return X

    def define_variational_distribution(self):

        # Initialize node D_q
        self.p_d = Parameter((self.X[:] > 0).astype(np.float))
        self.D_q = Bernoulli(self.p_d, self.dims('n,p ~ d,d'))

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

        # Initialize parameters of V_q
        if self.use_factors:
            self.b1[:] = self.V[:]
        else:
            self.b1[:] = np.random.gamma(1., size=(self.m, self.k))
        #self.b1[:] += np.random.uniform(-epsilon, epsilon, size=self.b1[:].shape)
        self.b1[:] = np.maximum(1e-15, np.nan_to_num(self.b1[:]))
        self.b2[:] = np.ones((self.m, self.k))

        # Initialize parameters of D_q
        self.p_d[:] = (self.X[:] > 0).astype(np.float)

    @numba.jit('void(f4[:, :], f4[:, :], f4[:, :], f4[:, :], f4[:, :], f4[:, :], f4[:, :])')
    def compute_Z_q_expectations(DZ_hat_i, DZ_hat_j, DZ_exp_logsum_hat, log_U_hat, log_V_hat, D_hat, X):
        DZ_hat_i[:, :] = 0
        DZ_hat_j[:, :] = 0
        DZ_exp_logsum_hat[:, :] = 0        
        n, p, latent_dim = log_U_hat.shape[0], log_V_hat.shape[0], log_U_hat.shape[1]
        for i in range(n):
            for j in range(p):
                logsum = log_U_hat[i, :] + log_V_hat[j, :]
                exp_logsum = np.exp(logsum)
                den = exp_logsum.sum()
                den = den if den > 0 else 1
                for k in range(latent_dim):
                    expectation = X[i, j] * exp_logsum[k] / den
                    DZ_hat_i[i, k] += D_hat[i, j] * expectation
                    DZ_hat_j[j, k] += D_hat[i, k] * expectation
                    DZ_exp_logsum_hat[j, k] += D_hat[i, j] * expectation * logsum[k]

    def update_variational_parameters(self):

        # Compute useful metrics based on the expectation of Z_q
        # E[Z_q] is memory-expensive, so it is itself not
        # explicitely computed.
        DZ_hat_i = np.empty((self.n, self.k), dtype=np.float32)
        DZ_hat_j = np.empty((self.m, self.k), dtype=np.float32)
        DZ_exp_logsum_hat = np.empty((self.m, self.k), dtype=np.float32)
        ZIGaP.compute_Z_q_expectations(
                DZ_hat_i,
                DZ_hat_j,
                DZ_exp_logsum_hat,
                self.log_U_hat,
                self.log_V_hat,
                self.D_hat,
                self.X[:].astype(np.float32))

        # Update parameters of U_q
        self.a1[:] = self.alpha1[np.newaxis, ...] + DZ_hat_i
        self.a2[:] = self.alpha2[:] + np.dot(self.D_hat, self.V_hat)
        self.a1[:] = np.maximum(1e-15, np.nan_to_num(self.a1[:]))
        self.a2[:] = np.maximum(1e-15, np.nan_to_num(self.a2[:]))
        self.U_hat = self.U_q.mean()
        self.log_U_hat = self.U_q.meanlog()

        # Update parameters of V_q
        self.b1[:] = self.beta1[np.newaxis, ...] + DZ_hat_j
        self.b2[:] = self.beta2[:] + np.dot(self.D_hat.T, self.U_hat)
        self.b1[:] = np.maximum(1e-15, np.nan_to_num(self.b1[:]))
        self.b2[:] = np.maximum(1e-15, np.nan_to_num(self.b2[:]))
        self.V_hat = self.V_q.mean()
        self.log_V_hat = self.V_q.meanlog()

        # Update parameters of D_q
        self.p_d[:] = sigmoid(logit(self.pi_d[:])[np.newaxis, ...] \
                - np.dot(self.U_hat, self.V_hat.T))
        self.p_d[:, self.pi_d[:] <= 0] = 1e-10
        self.p_d[:, self.pi_d[:] >= 1] = 1. - 1e-10
        self.p_d[self.X[:] != 0] = 1. - 1e-10
        self.D_hat = self.D_q.mean()

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

        # Update parameters of node D
        self.pi_d[:] = np.mean(self.p_d[:], axis=0)

    def update_expectations(self):
        self.U_hat = self.U_q.mean()
        self.V_hat = self.V_q.mean()
        self.log_U_hat = self.U_q.meanlog()
        self.log_V_hat = self.V_q.meanlog()
        self.D_hat = self.D_q.mean()
