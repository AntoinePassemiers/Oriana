# -*- coding: utf-8 -*-
# clustering.py
# author : Antoine Passemiers

from oriana.models import GaP, SparseGaP, ZIGaP, SparseZIGaP
from oriana.singlecell import CountMatrix, generate_factor_matrices

import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':

    zero_inflation_level = 0.5
    n, m, k = 100, 800, 2
    X, U, V = generate_factor_matrices(
            n, m, k, sparsity_degree_in_v=0.2,
            beta=80, theta=0.8, n_groups=2,
            zero_inflation_level=zero_inflation_level)
    counts = CountMatrix(X)

    history = list()
    model = SparseGaP(counts, k=2, use_factors=True)
    best_divergence = model.reconstruction_deviance()
    print('Initial Bregman divergence: %f' % best_divergence)
    history.append(best_divergence)
    U, V = model.factors()
    for iteration in range(50):
        print(np.round(np.dot(U, V.T)).astype(np.int))
        model.step()
        divergence = model.reconstruction_deviance()
        print('Iteration %i - Bregman divergence: %f' % (iteration + 1, divergence))

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
