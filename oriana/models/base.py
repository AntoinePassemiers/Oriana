# -*- coding: utf-8 -*-
# base.py: Base class for factor models
# author : Antoine Passemiers

from oriana import Dimensions
from oriana.nodes import Einsum, Multiply, Transpose

import numpy as np
from abc import abstractmethod, ABCMeta
from sklearn.decomposition import NMF


class FactorModel(metaclass=ABCMeta):

    def __init__(self, cmatrix, k=2, use_factors=True):

        # Count matrix
        self.cmatrix = cmatrix

        # Define dimensions
        self.k = k
        self.n = cmatrix.shape[0]
        self.m = self.p = cmatrix.shape[1]
        self.dims = Dimensions({ 'n': self.n, 'm': self.m, 'p': self.p, 'k': self.k })

        # Define model
        self.U = self.build_u_node()
        self.V = self.build_v_node()
        self.UV = Einsum('nk,mk->nm', self.U, self.V, name='UV')
        self.UV.forward()
        self.X = self.build_x_node(self.cmatrix, self.UV)

        # Define corresponding variational distribution
        self.define_variational_distribution()

        # Initialize parameters
        self.use_factors = use_factors
        model = NMF(n_components=self.k)
        self.U[:] = model.fit_transform(self.X[:])
        self.V.buffer = model.components_.T
        self.initialize_parameters()

    def initialize_parameters(self):

        # Initialize variational hyper-parameters
        self.initialize_variational_parameters()

        # Update expectations
        self.update_expectations()

        # Update hyper-parameters
        self.update_prior_hyper_parameters()

    def step(self):
        self.update_variational_parameters() # E-step
        self.update_prior_hyper_parameters() # M-step

    def reconstruction_deviance(self):
        self.UV[:] = self.X[:]
        self.D[:] = np.round(self.D_hat)
        mask = self.D[:] == 0
        assert not mask.all()
        ll_X_given_X = self.loglikelihood_X()
        self.U[:] = self.U_hat
        self.V[:] = self.Vprime_hat * self.S_hat
        self.UV.forward()
        self.UV[mask] = 0
        ll_X_given_UV = self.loglikelihood_X()
        return -2. * (ll_X_given_UV - ll_X_given_X)

    def explained_deviance(self):
        mask = self.D[:] == 0
        assert not mask.all()
        self.UV[:] = self.X[:]
        ll_X_given_X = self.loglikelihood_X()
        self.UV[:] = self.X[:].mean(axis=0)[np.newaxis, ...]
        ll_X_given_X_mean = self.loglikelihood_X()
        self.UV.forward()
        self.UV[mask] = 0
        ll_X_given_UV = self.loglikelihood_X()
        assert (ll_X_given_X >= ll_X_given_X_mean).all()
        return (ll_X_given_UV - ll_X_given_X_mean) / (ll_X_given_X - ll_X_given_X_mean)

    def frobenius_norm(self):
        Lambda = self.UV.asarray()
        frob = np.sqrt(((Lambda.flatten() - self.X.asarray().flatten()) ** 2.).sum())
        return frob

    def loglikelihood(self):
        self.UV.forward()
        ll = 0.
        ll += self.U.loglikelihood()
        ll += self.V.loglikelihood()
        ll += self.X.loglikelihood()
        return ll

    def factors(self):
        return self.U[:], self.V[:]

    @abstractmethod
    def build_u_node(self):
        pass

    @abstractmethod
    def build_v_node(self):
        pass

    @abstractmethod
    def build_x_node(self, cmatrix, UV):
        pass

    @abstractmethod
    def define_variational_distribution(self):
        pass

    @abstractmethod
    def initialize_variational_parameters(self):
        pass

    @abstractmethod
    def update_variational_parameters(self):
        pass

    @abstractmethod
    def update_prior_hyper_parameters(self):
        pass

    @abstractmethod
    def update_expectations(self):
        pass
