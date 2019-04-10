# -*- coding: utf-8 -*-
# base.py
# author : Antoine Passemiers

import numpy as np
from abc import ABCMeta, abstractmethod


class Node(metaclass=ABCMeta):

    def __init__(self, *parents):
        self.children = list()
        self.parents = parents
        for parent in self.parents:
            if isinstance(parent, Node):
                parent.add_child(self)

    def add_child(self, node):
        if not node in self.children:
            self.children.append(node)

    @abstractmethod
    def sample(self, **kwargs):
        pass


class DeterministicNode(Node, metaclass=ABCMeta):

    def __init__(self, *parents):
        Node.__init__(self, *parents)

    def sample(self, **kwargs):
        params = list()
        for param in self.parents:
            if isinstance(param, Node):
                params.append(param.sample())
            else:
                params.append(param.asarray())
        out = self._sample(*params, **kwargs)
        return out

    @abstractmethod
    def _sample(self, params):
        pass


class ProbabilisticNode(Node, metaclass=ABCMeta):

    def __init__(self, *parents, rel='', name=''):
        Node.__init__(self, *parents)
        self.name = name
        self.shape = rel.shape
        self.n_samples_per_distrib = rel.n_samples_per_distrib
        self.n_distribs = rel.n_distribs
        self.n_components = rel.n_components
        self.reshape_func = rel.reshape_func
        self.__buffer = self._init_buffer(self.shape)

    def sample(self, **kwargs):
        # TODO: reshape params
        arr_params = list()
        for param in self.parents:
            if isinstance(param, Node):
                arr_params.append(param.sample())
            else:
                arr_params.append(param.asarray())
        out = self._sample(*arr_params, **kwargs)

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
