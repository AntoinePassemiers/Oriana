# -*- coding: utf-8 -*-
# bernoulli.py
# author : Antoine Passemiers

from oriana.nodes.base import ProbabilisticNode

import numpy as np


class Bernoulli(ProbabilisticNode):

    def __init__(self, *args, **kwargs):
        ProbabilisticNode.__init__(self, *args, **kwargs)

    def _init_buffer(self, shape):
        return np.zeros(shape, dtype=np.int)

    def _sample(self, params):
        n = self.n_samples_per_distrib
        m = self.n_distribs
        c = self.n_components
        out = np.empty((n, m, c), dtype=np.int)
        ones = np.ones(len(params), dtype=np.int)
        out = np.random.binomial(ones, params, size=(m, n)).T
        out = out[..., np.newaxis]
        return out
