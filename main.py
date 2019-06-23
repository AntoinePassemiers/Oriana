# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

from oriana.model import ZIGaP
from oriana.singlecell import CountMatrix

import os
import numpy as np
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')


if __name__ == '__main__':


    #filepath = os.path.join(DATA_FOLDER, 'llorens.csv')
    #counts = CountMatrix.from_csv(filepath).T
    #print('Shape of X: %s' % str(counts.shape))

    # TODO: parse it from csv file
    # once everything works
    counts = CountMatrix(np.array(
                [[ 0,  0,  1,  0,  0],
                 [ 2,  0,  2,  3,  0],
                 [ 3,  0,  1,  9,  0],
                 [ 0,  1,  2,  1,  0],
                 [ 4,  0,  9,  8,  1],
                 [ 3,  0,  6,  1,  0],
                 [12,  0,  2, 26,  0],
                 [ 6,  1, 12,  5,  2],
                 [ 6,  0,  0,  6,  0],
                 [ 9,  0,  7, 23,  0]]))


    history = list()
    zigap = ZIGaP(counts, k=2)
    for iteration in range(10):
        zigap.step()
        divergence = zigap.reconstruction_deviance()
        print('Iteration %i - Bregman divergence: %f' % (iteration + 1, divergence))
        history.append(divergence)


    plt.plot(history)
    plt.ylabel('Bregman divergence')
    plt.xlabel('Variational E-M iterations')
    plt.show()
