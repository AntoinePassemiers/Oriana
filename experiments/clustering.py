# -*- coding: utf-8 -*-
# clustering.py
# author : Antoine Passemiers

from oriana.models import GaP, SparseGaP, ZIGaP, SparseZIGaP
from oriana.singlecell import CountMatrix, generate_factor_matrices

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from os.path import join
import pandas as pd

VERBOSE = True

def project_with_sparse_zigap(counts, k=2):
    model = SparseZIGaP(counts, k=k, use_factors=False)#True)
    best_divergence = model.reconstruction_deviance()
    if VERBOSE:
        print('Initial Bregman divergence: %f' % best_divergence)
    U, V = model.factors()
    for iteration in range(200):
        model.step()
        divergence = model.reconstruction_deviance()
        exp_deviance = model.explained_deviance()
        print('%dev:', exp_deviance)
        if VERBOSE:
            print('Iteration %3i - Bregman divergence: %f' % (iteration + 1, divergence))
            print('\t\tVariance U:', model.factors()[0].std()**2)
        if divergence <= best_divergence:
            best_divergence = divergence
            U, V = model.factors()
        elif divergence > best_divergence:
            if iteration > 10:
                break
    return U, V


def test_on_generated_dataset(K, theta, plot=False):
    # --- Generate random cells and genes ---

    zero_inflation_level = 0.5
    n_groups = 2
    n, m, k = 100, 800, K
    X, _, _, labels = generate_factor_matrices(
            n, m, k,
            sparsity_degree_in_v=0.9,
            beta=80,
            theta=theta, # Degree of separation between clusters
            n_groups=n_groups, # Number of clusters
            zero_inflation_level=zero_inflation_level)
    counts = CountMatrix(X)

    if plot:
        plt.imshow(X)
        plt.title('Synthetic count matrix')
        plt.ylabel('Genes')
        plt.xlabel('Cells')
        plt.show()


    # --- Clustering ---

    U, V = project_with_sparse_zigap(counts, k=k)
    log_U, log_V = np.log(U), np.log(V)

    predicted_labels = KMeans(n_clusters=n_groups, n_init=100).fit(log_U).labels_

    ari_k = adjusted_rand_score(labels, predicted_labels)
    if VERBOSE:
        print(labels)
        print(predicted_labels)
    print('Adjusted Rand Index (k): %f' % ari_k)


    # --- Visualization ---

    U, V = project_with_sparse_zigap(counts, k=2)
    log_U, log_V = np.log(U), np.log(V)
    predicted_labels = KMeans(n_clusters=n_groups, n_init=100).fit(log_U).labels_
    ari_2 = adjusted_rand_score(labels, predicted_labels)
    print('Adjusted Rand Index (2): %f' % ari_2)

    if plot:
        colors = ['salmon', 'steelblue', 'purple', 'gold']
        markers = ['o', '^', 'x', 'P']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(n_groups):
            for j in range(n_groups):
                idx = np.logical_and(labels == i, predicted_labels == j)
                ax.plot(log_U[idx, 0], log_U[idx, 1], color=colors[i], marker=markers[j], label=(i,j), lw=0, ms=8)
        ax.set_title('Projection of synthetic data with Sparse ZIGaP ($k=2$)')
        ax.set_ylabel(r'Factor $\widehat{U_2}$')
        ax.set_xlabel(r'Factor $\widehat{U_1}$')
        ax.legend()
        plt.show()
    return ari_k, ari_2

def test_generated(repeat=False):
    if repeat:
        K = 10
        global VERBOSE
        VERBOSE = False
        ari_k = list()
        ari_2 = list()
        thetas = np.linspace(0, 1, 3)
        for theta in thetas:
            print('\t\t-- \\theta =', theta, '--')
            ari_k.append(list())
            ari_2.append(list())
            for i in range(20):
                aris = test_on_generated_dataset(K, theta, plot=False)
                ari_k[-1].append(aris[0])
                ari_2[-1].append(aris[1])
        ari_k = np.array(ari_k)
        ari_2 = np.array(ari_2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(thetas, ari_k.mean(axis=1), yerr=ari_k.std(axis=1), label='ARI ({})'.format(K), capsize=6, marker='*', ms=8)
        ax.errorbar(thetas, ari_2.mean(axis=1), yerr=ari_2.std(axis=1), label='ARI (2)', capsize=6, marker='*', ms=8)
        ax.legend()
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel('ARI')
        ax.grid()
        ax.legend(loc='upper left')
        plt.show()
        #print(ari_k)
        #print(ari_2)
        #print(np.mean(ari_k), np.mean(ari_2))
        #print(np.std(ari_k), np.std(ari_2))
    else:
        test_on_generated_dataset(10, .5, plot=True)

def get_data_path(file_name):
    return join('../../data/', file_name)


def test_dataset():
    counts = CountMatrix.from_csv(get_data_path('llorens.csv'))
    cells_and_types = pd.read_csv(get_data_path('LlorensBobadilla2015_cells_and_types.csv'))
    cells_and_types.sort_values('cell')
    indices = list(set(counts.row_names) & set(cells_and_types['cell']))
    cells_and_types = cells_and_types[cells_and_types.cell.isin(indices)]
    counts.filter_rows(indices, inplace=True)
    labels = LabelEncoder().fit_transform(cells_and_types['type'])
    assert labels.shape[0] == counts.shape[0], (labels.shape, counts.shape)
    U, V = project_with_sparse_zigap(counts, k=10)
    predicted_labels = KMeans(n_clusters=np.unique(labels).shape[0], n_init=100).fit(log_U).labels_
    ari = adjusted_rand_score(labels, predicted_labels)
    print('Adjusted Rand Index:', ari)

if __name__ == '__main__':
    test_generated(repeat=False)
    #test_dataset()
