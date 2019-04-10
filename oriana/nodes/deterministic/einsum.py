# -*- coding: utf-8 -*-
# einsum.py
# author : Antoine Passemiers

from oriana.nodes.base import DeterministicNode

import numpy as np


class Einsum(DeterministicNode):

    def __init__(self, subscripts, variables):
        self.subscripts = subscripts
        self.variables = variables

    def sample(self):
        for var in self.variables:
            var.sample()
        arrays = [var.buffer for var in self.variables]
        return np.einsum(self.subscripts, *arrays)