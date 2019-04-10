# -*- coding: utf-8 -*-
# parameters.py
# author : Antoine Passemiers

import numpy as np


class Parameter:

    def __init__(self, data):
        self.data = np.asarray(data)

    def asarray(self):
        return self.data