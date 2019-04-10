# -*- coding: utf-8 -*-
# gamma.py
# author : Antoine Passemiers

from oriana.nodes.base import ProbabilisticNode

import numpy as np


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
        n = self.n_samples_per_distrib
        m = self.n_distribs
        c = self.n_components
        out = np.empty((n, m, c), dtype=np.int)
        shape_params = alpha
        scale_params = 1. / beta
        out = np.random.gamma(shape_params, scale_params, size=(m, n)).T
        out = out[..., np.newaxis]
        return out
