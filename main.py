# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

from oriana import CountMatrix

import os
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')


if __name__ == '__main__':

    filepath = os.path.join(DATA_FOLDER, 'llorens.csv')
    X = CountMatrix.from_csv(filepath).T.as_array()

    print('Shape of X: %s' % str(X.shape))
