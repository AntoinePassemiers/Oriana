# -*- coding: utf-8 -*-
# graph.py
# author : Antoine Passemiers

import numpy as np
from functools import reduce
from abc import ABCMeta, abstractmethod


class IncompatibleShapeException(Exception):
    pass


class Dimensions:

    def __init__(self, dims):
        self.dims = dims

    def __call__(self, rel):
        sides = rel.split('~')
        lhs, rhs = sides[0].strip(), sides[1].strip()
        lhs, rhs = lhs.split(','), rhs.split(',')
        if len(lhs) != len(rhs):
            raise IncompatibleShapeException(
                'Relation "%s" format is not correct.' % rel)
        
        shape = tuple([self.dims[key] for key in lhs])

        s_indices = [i for i, symbol in enumerate(rhs) if symbol == '+']
        d_indices = [i for i, symbol in enumerate(rhs) if symbol == '-']
        c_indices = [i for i, symbol in enumerate(rhs) if symbol == '*']
        in_indices = s_indices + d_indices + c_indices

        mul = lambda x, y: x * y
        s_shape = [self.dims[lhs[i]] for i in s_indices]
        n_samples_per_distrib = reduce(mul, [1] + s_shape)
        d_shape = [self.dims[lhs[i]] for i in d_indices]
        n_distribs = reduce(mul, [1] + d_shape)
        c_shape = [self.dims[lhs[i]] for i in c_indices]
        n_components = reduce(mul, [1] + c_shape)

        def reshape(data):
            in_shape = (n_samples_per_distrib, n_distribs, n_components)
            assert(data.shape == in_shape)
            data = data.reshape(*(s_shape + d_shape + c_shape))
            return np.transpose(data, in_indices)

        return shape, n_samples_per_distrib, n_distribs, n_components, reshape

    def __setitem__(self, key, value):
        self.dims[key, value]

    def __getitem__(self, key):
        return self.dims[key]


class Parameter:

    def __init__(self, data):
        self.data = np.asarray(data)

    def asarray(self):
        return self.data


class Variable(metaclass=ABCMeta):

    def __init__(self, params, rel, name=''):
        self.params = params
        self.rel = rel
        self.name = name
        self.shape = rel[0]
        self.n_samples_per_distrib = rel[1]
        self.n_distribs = rel[2]
        self.n_components = rel[3]
        self.reshape_func = rel[4]
        self.__buffer = self._init_data(self.shape)

    def sample(self, *args, **kwargs):
        params = self.params.asarray()
        out = self._sample(params, **kwargs)
        out = self.reshape_func(out)
        self.__buffer[:] = out
        return out

    def __repr__(self):
        return 'Variable %s of shape %s' % (self.name, str(self.shape))

    @abstractmethod
    def _init_data(self, shape):
        pass

    @abstractmethod
    def _sample(self, params):
        pass

    @property
    def buffer(self):
        return self.__buffer


class Bernoulli(Variable):

    def __init__(self, *args, **kwargs):
        Variable.__init__(self, *args, **kwargs)

    def _init_data(self, shape):
        return np.zeros(shape, dtype=np.int)

    def _sample(self, params):
        n = self.n_samples_per_distrib
        m = self.n_distribs
        c = self.n_components
        out = np.empty((n, m, p), dtype=np.int)
        ones = np.ones(len(params), dtype=np.int)
        out = np.random.binomial(ones, params, size=(m, n)).T
        out = out[..., np.newaxis]
        assert(out.shape == (n, m, c))
        return out


class Gamma(Variable):

    def __init__(self, *args, **kwargs):
        Variable.__init__(self, *args, **kwargs)

    def _init_data(self, shape):
        return np.zeros(shape, dtype=np.float)

    def _sample(self, params):
        n = self.n_samples_per_distrib
        m = self.n_distribs
        c = self.n_components
        out = np.empty((n, m, p), dtype=np.int)
        shape_params = params[..., 0]
        scale_params = 1. / params[..., 1]
        out = np.random.gamma(shape_params, scale_params, size=(m, n)).T
        out = out[..., np.newaxis]
        assert(out.shape == (n, m, c))
        return out


class Poisson(Variable):

    def __init__(self, *args, **kwargs):
        Variable.__init__(self, *args, **kwargs)

    def _init_data(self, shape):
        return np.zeros(shape, dtype=np.int)

    def _sample(self, params):
        n = self.n_samples_per_distrib
        m = self.n_distribs
        c = self.n_components
        out = np.empty((n, m, p), dtype=np.int)
        out = np.random.poisson(params, size=(m, n)).T
        out = out[..., np.newaxis]
        assert(out.shape == (n, m, c))
        return out


class Einsum:

    def __init__(self, subscripts, variables):
        self.subscripts = subscripts
        self.variables = variables

    def sample(self):
        for var in self.variables:
            var.sample()
        arrays = [var.buffer for var in self.variables]
        return np.einsum(self.subscripts, *arrays)


n = 10
m = p = 5
K = 3
dims = Dimensions({ 'n': n, 'm': m, 'p': p, 'K': K })


pi_s = Parameter([0., 0., 1., 0.5, 0.5])
mapping = dims('m,K ~ +,-')
S = Bernoulli(pi_s, mapping, name='S')

alpha = Parameter(np.random.rand(K, 2))
mapping = dims('n,K ~ -,+')
U = Gamma(alpha, mapping, name='U')

beta = Parameter(np.random.rand(K, 2))
mapping = dims('m,K ~ -,+')
Vprime = Gamma(beta, mapping, name='Vprime')

print(S.sample())
print(U.sample())

V = Einsum('nk,mk->nmk', [U, Vprime])

print(V.sample())

# Z = Poisson('')
