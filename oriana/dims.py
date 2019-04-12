# -*- coding: utf-8 -*-
# dims.py
# author : Antoine Passemiers

from oriana.exceptions import IncompatibleShapeException

import numpy as np
from functools import reduce


class DimRelation:

    def __init__(self, shape, n_samples_per_distrib,
                 n_distribs, n_components, reshape_func,
                 inv_reshape_func):
        self.shape = shape
        self.n_samples_per_distrib = n_samples_per_distrib
        self.n_distribs = n_distribs
        self.n_components = n_components
        self.reshape_func = reshape_func
        self.inv_reshape_func = inv_reshape_func


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

        in_shape = (n_samples_per_distrib, n_distribs, n_components)
        out_shape = tuple(s_shape + d_shape + c_shape)

        def reshape(data):
            assert(data.shape == in_shape)
            data = data.reshape(*out_shape, order='C')
            return np.transpose(data, in_indices)

        def inv_reshape(data):
            assert(data.shape == shape)
            inv_in_indices = [in_indices.index(i) for i in range(len(in_indices))]
            data = np.transpose(data, inv_in_indices)
            return data.reshape(*in_shape, order='C')

        return DimRelation(shape, n_samples_per_distrib,
                           n_distribs, n_components, reshape, inv_reshape)

    def __setitem__(self, key, value):
        self.dims[key, value]

    def __getitem__(self, key):
        return self.dims[key]
