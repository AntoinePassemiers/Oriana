# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

from oriana.models import GaP, SparseZIGaP
from oriana.singlecell import CountMatrix, generate_factor_matrices

import os
import numpy as np
import matplotlib.pyplot as plt


#ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = '../'
DATA_FOLDER = os.path.join(ROOT, 'data')

def frobenius(X):
    return np.sqrt((X**2).sum())

if __name__ == '__main__':


    filepath = os.path.join(DATA_FOLDER, 'llorens.csv')
    counts = CountMatrix.from_csv(filepath)#.T
    X = counts.as_array()
    print('Shape of X: %s' % str(counts.shape))

    history = list()
    model = SparseZIGaP(counts, k=2, use_factors=True)
    best_divergence = model.reconstruction_deviance()
    U, V = model.factors()
    Lambda = np.dot(U, V.T)
    kappa = (X*Lambda).sum() / (Lambda**2).sum()
    print('Initial Bregman divergence: %f' % best_divergence)
    print('Initial Frobenius distance: %f' % frobenius(X - kappa*Lambda))
    history.append(best_divergence)
    for iteration in range(50):
        Lambda = np.dot(U, V.T)
        #Lambda = np.round(Lambda).astype(np.int)
        kappa = (X*Lambda).sum() / (Lambda**2).sum()
        model.step()
        divergence = model.reconstruction_deviance()
        print('Iteration %3i - Bregman divergence: %f' % (iteration + 1, divergence))
        print('              - Frobenius distance: %f' % frobenius(X - kappa*Lambda))

        if True:#divergence <= best_divergence:
            best_divergence = divergence
            history.append(divergence)
            U, V = model.factors()
        elif divergence > best_divergence:
            break
    print(np.round(np.dot(U, V.T)).astype(np.int))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Variational E-M iterations')
    ax1.set_ylabel('Bregman divergence', color='salmon')
    ax1.plot(history, color='salmon')
    ax1.tick_params(axis='y', labelcolor='salmon')
    #ax2 = ax1.twinx()
    #ax2.set_ylabel('Log-likelihood', color='steelblue')
    #ax2.plot(history[:, 1], color='steelblue')
    #ax2.tick_params(axis='y', labelcolor='steelblue')
    plt.show()
