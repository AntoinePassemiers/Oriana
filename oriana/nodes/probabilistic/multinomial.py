# -*- coding: utf-8 -*-
# multinomial.py
# author : Antoine Passemiers

from oriana.nodes.base import ProbabilisticNode
from oriana.utils import factorial, log

import numpy as np
import scipy.stats


class Multinomial(ProbabilisticNode):
    """Variable node following a multinomial distribution.

    Attributes:
        rel (:obj:`oriana.DimRelation`): Utility object
            for handling node dimensions
    """

    def __init__(self, n, p, rel, **kwargs):
        ProbabilisticNode.__init__(self, n, p, rel=rel, **kwargs)

    def _init_buffer(self, shape):
        return np.zeros(shape, dtype=np.int)

    def _sample(self, n, p):
        _n = self.n_samples_per_distrib
        _m = self.n_distribs
        _c = self.n_components
        out = np.empty((_n, _m, _c), dtype=np.int)
        
        p = p.reshape((_m, _c), order='C')
        n = n.reshape(_m, order='C')
        for i in range(_m):
            out[:, i, :] = np.random.multinomial(n[i], p[i, :], size=_n)
        return out

    def _mean(self, n, p):
        _n = self.n_samples_per_distrib
        _m = self.n_distribs
        _c = self.n_components
        out = np.empty((_n, _m, _c), dtype=np.float)
        
        n = n.reshape(_m, order='C')
        p = p.reshape((_m, _c), order='C')

        p_Z = p.sum(axis=1)[..., np.newaxis]
        p[p_Z[:, 0] > 0, :] /= p_Z[p_Z[:, 0] > 0]
        p *= n[..., np.newaxis]
        for i in range(_n):
            out[i, :, :] = p # TODO: numpy.tile
        return out

    def _logp(self, samples, n, p):
        _n = self.n_samples_per_distrib
        _m = self.n_distribs
        _c = self.n_components
        p = p.reshape((_m, _c), order='C')
        n = n.reshape(_m, order='C')

        out = np.empty((_n, _m), dtype=np.float)
        for i in range(_m):
            x = samples[:, i, :]
            out[:, i] = log(factorial(np.floor(n))) + np.dot(x, p[i]) 
            out[:, i] -= log(factorial(np.floor(x))).sum(axis=1)
        return out
