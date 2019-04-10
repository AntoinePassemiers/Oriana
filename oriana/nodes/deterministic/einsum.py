# -*- coding: utf-8 -*-
# einsum.py
# author : Antoine Passemiers

from oriana.nodes.base import DeterministicNode

import numpy as np


class Einsum(DeterministicNode):

    def __init__(self, subscripts, *nodes, **kwargs):
        DeterministicNode.__init__(self, *nodes, **kwargs)
        self.subscripts = subscripts

    def _sample(self, *params):
        return np.einsum(self.subscripts, *params)
