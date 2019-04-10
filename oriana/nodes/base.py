# -*- coding: utf-8 -*-
# base.py
# author : Antoine Passemiers

import numpy as np
from abc import ABCMeta, abstractmethod


class Node:

    pass


class DeterministicNode(Node):

    pass


class ProbabilisticNode(Node, metaclass=ABCMeta):

    def __init__(self, params, rel, name=''):
        self.params = params
        self.rel = rel
        self.name = name
        self.shape = rel[0]
        self.n_samples_per_distrib = rel[1]
        self.n_distribs = rel[2]
        self.n_components = rel[3]
        self.reshape_func = rel[4]
        self.__buffer = self._init_buffer(self.shape)

    def sample(self, *args, **kwargs):
        params = self.params.asarray()
        out = self._sample(params, **kwargs)

        n = self.n_samples_per_distrib
        m = self.n_distribs
        c = self.n_components
        assert(out.shape == (n, m, c))
        
        out = self.reshape_func(out)
        self.__buffer[:] = out        
        return out

    def __repr__(self):
        return 'Variable %s of shape %s' % (self.name, str(self.shape))

    @abstractmethod
    def _init_buffer(self, shape):
        pass

    @abstractmethod
    def _sample(self, params):
        pass

    @property
    def buffer(self):
        return self.__buffer
