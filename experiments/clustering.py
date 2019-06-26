# -*- coding: utf-8 -*-
# clustering.py
# author : Antoine Passemiers

from oriana.models import GaP, SparseGaP, ZIGaP, SparseZIGaP
from oriana.singlecell import CountMatrix, generate_factor_matrices

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def project_with_sparse_zigap(counts, k=2):
    model = SparseZIGaP(counts, k=k, use_factors=True)
    best_divergence = model.reconstruction_deviance()
    print('Initial Bregman divergence: %f' % best_divergence)
    U, V = model.factors()
    for iteration in range(200):
        model.step()
        divergence = model.reconstruction_deviance()
        print('Iteration %i - Bregman divergence: %f' % (iteration + 1, divergence))
        if divergence <= best_divergence:
            best_divergence = divergence
            U, V = model.factors()
        elif divergence > best_divergence:
            if iteration > 10:
                break
    return U, V


if __name__ == '__main__':


    # --- Generate random cells and genes ---

    zero_inflation_level = 0.5
    n_groups = 2
    n, m, k = 100, 800, 40
    X, _, _, labels = generate_factor_matrices(
            n, m, k,
            sparsity_degree_in_v=0.9,
            beta=80,
            theta=0.5, # Degree of separation between clusters
            n_groups=n_groups, # Number of clusters
            zero_inflation_level=zero_inflation_level)
    counts = CountMatrix(X)

    plt.imshow(X)
    plt.title('Synthetic count matrix')
    plt.xlabel('Cells')
    plt.ylabel('Genes')
    plt.show()


    # --- Clustering ---

    U, V = project_with_sparse_zigap(counts, k=40)
    log_U, lo_V = np.log(U), np.log(V)

    predicted_labels = KMeans(n_clusters=n_groups, n_init=100).fit(log_U).labels_

    ari = adjusted_rand_score(labels, predicted_labels)
    print(labels)
    print(predicted_labels)
    print('Adjusted Rand Index: %f' % ari)


    # --- Visualization ---

    U, V = project_with_sparse_zigap(counts, k=2)
    log_U, lo_V = np.log(U), np.log(V)

    colors = ['salmon', 'blue', 'purple', 'gold']
    for i in range(n_groups):
        idx = (labels == i)
        plt.scatter(log_U[idx, 0], log_U[idx, 1], color=colors[i])
    plt.title('Projection of synthetic data with Sparse ZIGaP (k=2)')
    plt.ylabel(r'Factor $\hat{U_2}$')
    plt.xlabel(r'Factor $\hat{U_1}$')
    plt.show()
