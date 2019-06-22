# -*- coding: utf-8 -*-
# base.py
# author : Antoine Passemiers

import numpy as np
from copy import deepcopy
from abc import ABCMeta, abstractmethod


class Node(metaclass=ABCMeta):

    def __init__(self, *parents):
        self.children = list()
        self.parents = parents
        for parent in self.parents:
            if isinstance(parent, Node):
                parent.add_child(self)
        self.fixed = False

    def add_child(self, node):
        if not node in self.children:
            self.children.append(node)

    def fix(self, recursive=False):
        self.fixed = True
        if recursive:
            for parent in self.parents:
                if isinstance(parent, Node):
                    parent.fix(recursive=recursive)

    def unfix(self, recursive=False):
        self.fixed = False
        if recursive:
            for parent in self.parents:
                if isinstance(parent, Node):
                    parent.unfix(recursive=recursive)

    @abstractmethod
    def sample(self, **kwargs):
        pass

    @property
    def buffer(self):
        return self._buffer

    def asarray(self):
        return np.asarray(self.buffer)

    def __setitem__(self, key, value):
        self._buffer[key] = value

    def __getitem__(self, key):
        return self._buffer[key]

    @property
    def buffer(self):
        return self._buffer
    
    @buffer.setter
    def buffer(self, data):
        self._buffer = data


class DeterministicNode(Node, metaclass=ABCMeta):

    def __init__(self, *parents):
        Node.__init__(self, *parents)
        self._buffer = None

    def sample(self, recursive=False):
        arr_params = list()
        for param in self.parents:
            if isinstance(param, Node) and recursive:
                param.sample(recursive=recursive)
            arr_params.append(param.asarray())
        if not self.fixed:
            out = self._sample(*arr_params)
            self._buffer = out
        else:
            out = self._buffer
            assert(out is not None)
        self.fix()
        return out

    def forward(self):
        return self.sample(recursive=False)

    @abstractmethod
    def _sample(self, params):
        pass


class ProbabilisticNode(Node, metaclass=ABCMeta):

    def __init__(self, *parents, rel='', name=''):
        Node.__init__(self, *parents)
        self.name = name
        self.rel = rel
        self.shape = rel.shape
        self.n_samples_per_distrib = rel.n_samples_per_distrib
        self.n_distribs = rel.n_distribs
        self.n_components = rel.n_components
        self.reshape_func = rel.reshape_func
        self.inv_reshape_func = rel.inv_reshape_func
        self._buffer = self._init_buffer(self.shape)

    def logpdfs(self):
        samples = self.inv_reshape_func(self._buffer)
        n = self.n_samples_per_distrib
        m = self.n_distribs
        c = self.n_components
        assert(samples.shape == (n, m, c))

        arr_params = list()
        for param in self.parents:
            arr_params.append(param.asarray())

        pdfs = self._logpdfs(samples, *arr_params)
        return np.nan_to_num(pdfs)

    def logp(self):
        return self.logpdfs().sum()

    def updates_buffer(func):
        def new_func(self, recursive=False):
            arr_params = list()
            for param in self.parents:
                if isinstance(param, Node) and recursive:
                    param.sample(recursive=recursive)
                arr_params.append(param.asarray())

            if not self.fixed:
                out = func(self, *arr_params) 
                n = self.n_samples_per_distrib
                m = self.n_distribs
                c = self.n_components
                assert(out.shape == (n, m, c))
                out = self.reshape_func(out)
                self._buffer[:] = out
            else:
                out = self._buffer
            return out
        new_func.__name__ = func.__name__
        return new_func

    @updates_buffer
    def sample(self, *params):
        return self._sample(*params)

    @updates_buffer
    def mean(self, *params):
        return self._mean(*params)

    @abstractmethod
    def _init_buffer(self, shape):
        pass

    @abstractmethod
    def _sample(self, *params):
        pass

    @abstractmethod
    def _mean(self, *params):
        pass
        
    @abstractmethod
    def _logpdfs(self, samples, *params):
        pass

    def __repr__(self):
        return 'Variable %s of shape %s' % (self.name, str(self.shape))
