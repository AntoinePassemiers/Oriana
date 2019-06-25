# -*- coding: utf-8 -*-
# base.py: Base class for factor models
# author : Antoine Passemiers

from oriana import Dimensions
from oriana.nodes import Einsum, Multiply, Transpose

from abc import abstractmethod, ABCMeta
from sklearn.decomposition import NMF


class FactorModel(metaclass=ABCMeta):

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
        self.U = self.build_u_node()
        self.V = self.build_v_node()
        self.UV = Einsum('nk,mk->nm', self.U, self.V, name='UV')
        self.UV.forward()
        self.X = self.build_x_node(self.cmatrix, self.UV)

        # Define corresponding variational distribution
        self.define_variational_distribution()

        # Initialize parameters
        self.tau = tau
        self.use_factors = use_factors
        model = NMF(n_components=self.k)
        self.U[:] = model.fit_transform(self.X[:])
        self.V.buffer = model.components_.T
        self.initialize_variational_parameters()
        self.initialize_prior_hyper_parameters()

    def step(self):
        self.update_variational_parameters() # E-step
        self.update_prior_hyper_parameters() # M-step

    def reconstruction_deviance(self):
        self.UV[:] = self.X[:]
        ll_X_given_X = self.X.loglikelihood()
        self.UV.forward()
        ll_X_given_UV = self.X.loglikelihood()
        return -2. * (ll_X_given_UV - ll_X_given_X)

    def explained_deviance(self):
        self.UV[:] = self.X[:]
        ll_X_given_X = self.X.loglikelihood()
        self.UV[:] = self.X[:].mean(axis=1)[..., np.newaxis] # TODO: not sure about the axis
        ll_X_given_X_mean = self.X.loglikelihood()
        self.UV.forward()
        ll_X_given_UV = self.X.loglikelihood()
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
    def build_model(self):
        pass

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
    def initialize_prior_hyper_parameters(self):
        pass

    @abstractmethod
    def update_variational_parameters(self):
        pass

    @abstractmethod
    def update_prior_hyper_parameters(self):
        pass
