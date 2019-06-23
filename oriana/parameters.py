# -*- coding: utf-8 -*-
# parameters.py
# author : Antoine Passemiers

import numpy as np


class Parameter:

    def __init__(self, data):
        self._buffer = np.asarray(data, dtype=np.float)

    def asarray(self):
        return self._buffer

    def __getitem__(self, key):
        return self._buffer[key]

    def __setitem__(self, key, value):
        self._buffer[key] = value

    @property
    def buffer(self):
        return self._buffer
    
    @buffer.setter
    def buffer(self, data):
        self._buffer = data
