# -*- coding: utf-8 -*-
# multiply.py
# author : Antoine Passemiers

from oriana.nodes.base import DeterministicNode

import numpy as np


class Multiply(DeterministicNode):

    def __init__(self, left_node, right_node, **kwargs):
        DeterministicNode.__init__(self, left_node, right_node, **kwargs)

    def _sample(self, left, right):
        return left * right
