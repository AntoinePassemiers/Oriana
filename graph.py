# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

from oriana.nodes import Poisson, Gamma, Bernoulli

import numpy as np
from functools import reduce
from abc import ABCMeta, abstractmethod


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
