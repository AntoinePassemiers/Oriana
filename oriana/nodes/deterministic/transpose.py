# -*- coding: utf-8 -*-
# transpose.py
# author : Antoine Passemiers

from oriana.nodes.base import DeterministicNode

import numpy as np


class Transpose(DeterministicNode):

    def __init__(self, node, **kwargs):
        DeterministicNode.__init__(self, node, **kwargs)

    def _sample(self, arr):
        return arr.T
