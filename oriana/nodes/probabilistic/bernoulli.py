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
        return np.zeros(shape, dtype=np.float)

    def _sample(self, pi):
        """
        Parameters:
            pi (object): Bernoulli pi parameter, or the probability
                a Bernoulli variable takes value one.
                Can be either a Parameter or a Node object.
        """
        n = self.n_samples_per_distrib
        m = self.n_distribs
        params = pi.reshape(-1, order='C')
        ones = np.ones(len(pi), dtype=np.float)
        out = np.random.binomial(ones, params, size=(n, m))
        out = out[..., np.newaxis]
        return out

    def _mean(self, pi):
        _n = self.n_samples_per_distrib
        _m = self.n_distribs
        _c = self.n_components
        pi = pi.reshape(-1, order='C')
        out = np.tile(pi, (_n, 1))[..., np.newaxis]
        assert(out.shape == (_n, _m, _c))
        return out

    def _logp(self, samples, pi):
        p = pi.reshape(-1, order='C')
        log = lambda x: np.log(np.maximum(1e-15, x))
        return samples * log(p) + (1. - samples) * log(1. - p)
