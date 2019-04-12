# -*- coding: utf-8 -*-
# poisson.py
# author : Antoine Passemiers

from oriana.nodes.base import ProbabilisticNode

import numpy as np
import scipy.stats


class Poisson(ProbabilisticNode):
    """Variable node following a Poisson distribution.

    Attributes:
        rel (:obj:`oriana.DimRelation`): Utility object
            for handling node dimensions
    """

    def __init__(self, l, rel, **kwargs):
        ProbabilisticNode.__init__(self, l, rel=rel, **kwargs)

    def _init_buffer(self, shape):
        return np.zeros(shape, dtype=np.int)

    def _sample(self, l):
        """
        Parameters:
            l (object): Poisson lambda parameter, or the number
                of event occurences in a fixed interval of time.
                Can be either a Parameter or a Node object.
        """
        n = self.n_samples_per_distrib
        m = self.n_distribs
        params = l.flatten(order='C')
        out = np.random.poisson(params, size=(m, n)).T
        out = out[..., np.newaxis]
        return out

    def _logpdfs(self, samples, l):
        params = l.flatten(order='C')
        return scipy.stats.poisson.logpmf(samples, params)
