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

    def define_variational_distribution(self):

        # Initialize node S_q
        self.p_s = Parameter(np.full((self.m, self.k), self.tau))
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

    def initialize_variational_parameters(self):

        # Initialize parameters of U and Vprime
        GaP.initialize_variational_parameters(self)

        # Initialize parameters of S_q
        self.p_s[:] = np.ones((self.m, self.k))

        # Initialize parameters of D_q
        self.p_d[:] = (self.X[:] > 0).astype(np.float)

    def initialize_parameters(self):

        # Initialize variational parameters
        self.initialize_variational_parameters()

        # Update expectations
        self.U_hat = self.U_q.mean()
        self.Vprime_hat = self.Vprime_q.mean()
        self.log_U_hat = self.U_q.meanlog()
        self.log_V_hat = self.V_q.meanlog()
        self.D_hat = self.D_q.mean()

        # Update hyper-parameters
        self.update_prior_hyper_parameters()

    def update_variational_parameters(self):

        # Update parameters of Z_q
        self.Z_hat_i = np.empty((self.n, self.k), dtype=np.float32)
        self.Z_hat_j = np.empty((self.m, self.k), dtype=np.float32)
        GaP.compute_Z_q_expectations(
                self.Z_hat_i,
                self.Z_hat_j,
                self.log_U_hat,
                self.log_V_hat,
                self.X[:].astype(np.float32))

        # Update parameters of U_q
        self.a1[:] = self.alpha1[:] + np.einsum('ij,jk,ijk->ik', D_hat, S_hat, Z_hat)
        self.a2[:] = self.alpha2[:] + np.dot(D_hat, V_hat)

        # Update parameters of Vprime_q
        self.b1[:] = self.beta1[:] + S_hat * np.einsum('ij,ijk->jk', D_hat, Z_hat)
        self.b2[:] = self.beta2[:] + S_hat * np.dot(D_hat.T, U_hat)

        # Update parameters of Z_q
        S_tilde = (self.p_s[:] > self.tau)
        log_sum = log_U_hat.reshape(self.n, 1, self.k) + log_Vprime_hat.reshape(1, self.m, self.k)
        self.r[:] = S_tilde[np.newaxis, ...] * np.exp(log_sum)
        norm = self.r[:].sum(axis=2)
        indices = (norm != 0.)
        self.r[indices] /= norm[indices][..., np.newaxis]

        # Update parameters of D_q
        self.p_d[:] = sigmoid(logit(self.pi_d[:])[np.newaxis, ...] \
                - np.dot(U_hat, V_hat.T))
        self.p_d[:, self.pi_d[:] == 0] = 1e-10
        self.p_d[:, self.pi_d[:] == 1] = 1. - 1e-10
        self.p_d[self.X[:] != 0] = 1. - 1e-10
        self.D_hat = self.D_q.mean()

        # Update parameters of S_q
        tmp = -np.nan_to_num(np.einsum('ij,ijk,ijk->jk', D_hat, Z_hat, log_sum))
        tmp += np.nan_to_num(np.dot(D_hat.T, U_hat) * Vprime_hat)
        self.p_s[:] = sigmoid(logit(self.pi_s[:])[..., np.newaxis] + tmp)
        self.p_s[:] = np.nan_to_num(self.p_s[:])
        self.p_s[self.pi_s[:] == 0] = 1e-10
        self.p_s[self.pi_s[:] == 1] = 1. - 1e-10
        assert((0 <= self.p_s[:]).all() and (self.p_s[:] <= 1).all())

        # Update parameters of UV
        self.U[:] = self.U_hat
        self.V[:] = self.V_hat
        self.UV.forward()

    def update_prior_hyper_parameters(self):

        # Update Gamma parameters
        GaP.update_prior_hyper_parameters()

        # Update parameters of node D
        self.pi_d[:] = np.mean(self.p_d[:], axis=0)

        # Update parameters of node S
        self.pi_s[:] = np.mean(self.p_s[:], axis=1)
