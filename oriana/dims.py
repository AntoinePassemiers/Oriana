# -*- coding: utf-8 -*-
# dims.py
# author : Antoine Passemiers

from oriana.exceptions import IncompatibleShapeException

import numpy as np
from functools import reduce


class Dimensions:

    def __init__(self, dims):
        self.dims = dims

    def __call__(self, rel):
        sides = rel.split('~')
        lhs, rhs = sides[0].strip(), sides[1].strip()
        lhs, rhs = lhs.split(','), rhs.split(',')
        if len(lhs) != len(rhs):
            raise IncompatibleShapeException(
                'Relation "%s" format is not correct.' % rel)
        
        shape = tuple([self.dims[key] for key in lhs])

        s_indices = [i for i, symbol in enumerate(rhs) if symbol == '+']
        d_indices = [i for i, symbol in enumerate(rhs) if symbol == '-']
        c_indices = [i for i, symbol in enumerate(rhs) if symbol == '*']
        in_indices = s_indices + d_indices + c_indices

        mul = lambda x, y: x * y
        s_shape = [self.dims[lhs[i]] for i in s_indices]
        n_samples_per_distrib = reduce(mul, [1] + s_shape)
        d_shape = [self.dims[lhs[i]] for i in d_indices]
        n_distribs = reduce(mul, [1] + d_shape)
        c_shape = [self.dims[lhs[i]] for i in c_indices]
        n_components = reduce(mul, [1] + c_shape)

        def reshape(data):
            in_shape = (n_samples_per_distrib, n_distribs, n_components)
            assert(data.shape == in_shape)
            data = data.reshape(*(s_shape + d_shape + c_shape))
            return np.transpose(data, in_indices)

        return shape, n_samples_per_distrib, n_distribs, n_components, reshape

    def __setitem__(self, key, value):
        self.dims[key, value]

    def __getitem__(self, key):
        return self.dims[key]
