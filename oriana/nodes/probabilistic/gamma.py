# -*- coding: utf-8 -*-
# gamma.py
# author : Antoine Passemiers

from oriana.nodes.base import ProbabilisticNode
from oriana.utils import digamma, gamma, log

import numpy as np
import scipy.stats


class Gamma(ProbabilisticNode):
    """Variable node following a Gamma distribution.

    Attributes:
        rel (:obj:`oriana.DimRelation`): Utility object
            for handling node dimensions
    """

    def __init__(self, alpha, beta, rel, **kwargs):
        ProbabilisticNode.__init__(self, alpha, beta, rel=rel, **kwargs)

    def _init_buffer(self, shape):
        return np.zeros(shape, dtype=np.float)

    def _sample(self, alpha, beta):
        _n = self.n_samples_per_distrib
        _m = self.n_distribs
        _c = self.n_components
        out = np.empty((_n, _m, _c), dtype=np.int)
        shape_params = alpha.reshape(-1, order='C')
        scale_params = 1. / beta.reshape(-1, order='C')
        out = np.random.gamma(shape_params, scale_params, size=(_n, _m))
        out = out[..., np.newaxis]
        return out

    def _mean(self, alpha, beta):
        _n = self.n_samples_per_distrib
        _m = self.n_distribs
        _c = self.n_components
        alpha = alpha.reshape(-1, order='C')
        beta = beta.reshape(-1, order='C')
        avg = alpha / beta
        out = np.tile(avg, (_n, 1))[..., np.newaxis]
        assert(out.shape == (_n, _m, _c))
        return out

    @ProbabilisticNode.updates_buffer
    def meanlog(self, *params):
        return self._meanlog(*params)

    def _meanlog(self, alpha, beta):
        _n = self.n_samples_per_distrib
        _m = self.n_distribs
        _c = self.n_components
        alpha = alpha.reshape(-1, order='C')
        beta = beta.reshape(-1, order='C')
        avglog = digamma(alpha) - np.log(beta)
        out = np.tile(avglog, (_n, 1))[..., np.newaxis]
        assert(out.shape == (_n, _m, _c))
        return out

    def _logp(self, samples, alpha, beta):
        alpha = alpha.reshape(-1, order='C')
        beta = beta.reshape(-1, order='C')
        logps = (alpha - 1.) * log(samples) - (samples / beta)
        logps += -alpha * log(beta) - log(gamma(alpha))
        return logps
