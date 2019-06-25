# -*- coding: utf-8 -*-
# poisson.py
# author : Antoine Passemiers

from oriana.nodes.base import ProbabilisticNode
from oriana.utils import factorial, log

import numpy as np
import scipy.stats


class Poisson(ProbabilisticNode):
    """Variable node following a Poisson distribution.

    Attributes:
        rel (:obj:`oriana.DimRelation`): Utility object
            for handling node dimensions
    """

    def __init__(self, l, rel, **kwargs):
        """Constructs a Poisson node.
        
        Parameters:
            l (object): Node or Parameter object representing
                the average number of event occurences in a fixed
                interval of time.
        """
        ProbabilisticNode.__init__(self, l, rel=rel, **kwargs)

    def _init_buffer(self, shape):
        """Initializes an empty buffer of integer values.

        Parameters:
            shape (tuple): Buffer shape.

        Todo:
            * Handle sparse arrays
        """
        return np.zeros(shape, dtype=np.float)

    def _sample(self, l):
        """
        Parameters:
            l (object): Poisson lambda parameter, or the average
                number of event occurences in a fixed interval of time.
                Can be either a Parameter or a Node object.
        """
        n = self.n_samples_per_distrib
        m = self.n_distribs
        params = l.reshape(-1, order='C')
        out = np.random.poisson(params, size=(n, m))
        out = out[..., np.newaxis]
        return out

    def _mean(self, l):
        _n = self.n_samples_per_distrib
        _m = self.n_distribs
        _c = self.n_components
        l = l.reshape(-1, order='C')
        out = np.tile(l, (_n, 1))[..., np.newaxis]
        assert(out.shape == (_n, _m, _c))
        return out

    def _logp(self, samples, l):
        l = l.reshape(-1, order='C')
        logps = -l + samples * log(l) + log(factorial(np.floor(samples)))
        return logps
