# -*- coding: utf-8 -*-
# bernoulli.py
# author : Antoine Passemiers

from oriana.nodes.base import ProbabilisticNode

import numpy as np
import scipy.stats


class Bernoulli(ProbabilisticNode):
    """Variable node following a Bernoulli distribution.

    Attributes:
        rel (:obj:`oriana.DimRelation`): Utility object
            for handling node dimensions
    """

    def __init__(self, pi, rel, **kwargs):
        ProbabilisticNode.__init__(self, pi, rel=rel, **kwargs)

    def _init_buffer(self, shape):
        return np.zeros(shape, dtype=np.int)

    def _sample(self, pi):
        """
        Parameters:
            pi (object): Bernoulli pi parameter, or the probability
                a Bernoulli variable takes value one.
                Can be either a Parameter or a Node object.
        """
        n = self.n_samples_per_distrib
        m = self.n_distribs
        params = pi.flatten(order='C')
        ones = np.ones(len(pi), dtype=np.int)
        out = np.random.binomial(ones, params, size=(m, n)).T
        out = out[..., np.newaxis]
        return out

    def _logpdfs(self, samples, pi):
        params = pi.flatten(order='C')
        return scipy.stats.bernoulli.logpmf(samples, params)

