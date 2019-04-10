# -*- coding: utf-8 -*-
# gamma.py
# author : Antoine Passemiers

from oriana.nodes.base import ProbabilisticNode

import numpy as np


class Gamma(ProbabilisticNode):

    def __init__(self, *args, **kwargs):
        ProbabilisticNode.__init__(self, *args, **kwargs)

    def _init_buffer(self, shape):
        return np.zeros(shape, dtype=np.float)

    def _sample(self, params):
        n = self.n_samples_per_distrib
        m = self.n_distribs
        c = self.n_components
        out = np.empty((n, m, c), dtype=np.int)
        shape_params = params[..., 0]
        scale_params = 1. / params[..., 1]
        out = np.random.gamma(shape_params, scale_params, size=(m, n)).T
        out = out[..., np.newaxis]
        return out
