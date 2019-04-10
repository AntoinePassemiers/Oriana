# -*- coding: utf-8 -*-
# einsum.py
# author : Antoine Passemiers

from oriana.nodes.base import DeterministicNode

import numpy as np


class Einsum(DeterministicNode):

    def __init__(self, subscripts, *nodes):
        DeterministicNode.__init__(self, *nodes)
        self.subscripts = subscripts

    def _sample(self, *params):
        return np.einsum(self.subscripts, *params)
