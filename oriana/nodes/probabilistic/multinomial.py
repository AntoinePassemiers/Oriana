# -*- coding: utf-8 -*-
# multinomial.py
# author : Antoine Passemiers

from oriana.nodes.base import ProbabilisticNode

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
        print(p)
        print(n[..., np.newaxis])
        p *= n[..., np.newaxis]
        print(p)
        for i in range(_n):
            out[i, :, :] = p
        return out

    def _logpdfs(self, samples, n, p):
        _n = self.n_samples_per_distrib
        _m = self.n_distribs
        _c = self.n_components
        p = p.reshape((_m, _c), order='C')
        n = n.reshape(_m, order='C')

        out = np.empty((_n, _m), dtype=np.float)
        for i in range(_m):
            out[:, i] = scipy.stats.multinomial.logpmf(
                    samples[:, i, :], n=n[i], p=p[i])
        return out
