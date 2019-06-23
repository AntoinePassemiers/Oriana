# -*- coding: utf-8 -*-
# generation.py
# author : Antoine Passemiers

import numpy as np


def generate_u(n, k, n_groups=3, theta=0.5):
    block_row_indices = list(range(0, (n // n_groups) \
            * n_groups, n // n_groups)) + [n]
    block_col_indices = list(range(0, (k // n_groups) \
            * n_groups, k // n_groups)) + [k]
    
    alpha = np.random.choice([100, 250], size=n_groups)
    EMPTY = -1
    U = np.full((n, k), EMPTY, dtype=np.float)

    for g in range(n_groups):
        i_start = block_row_indices[g]
        i_end = block_row_indices[g + 1]
        j_start = block_col_indices[g]
        j_end = block_col_indices[g + 1]

        block_height = i_end - i_start
        block_width = j_end - j_start
        block_size = (block_height, block_width)

        B_g = np.random.gamma(1., 1. / alpha[g], size=block_size)
        U[i_start:i_end, j_start:j_end] = B_g

    alpha_bar = np.mean(alpha)
    U[U == EMPTY] = np.random.gamma(
            1., 1. / ((1. - theta) * alpha_bar), size=(n, k))[U == EMPTY]
    return U


def generate_v(m, k, sparsity_degree=0.2, beta=80, theta=0.8, n_groups=2):
    m0 = int(np.round(m * sparsity_degree))

    block_row_indices = list(range(0, (m0 // n_groups) \
            * n_groups, m0 // n_groups)) + [m0]
    block_col_indices = list(range(0, (k // n_groups) \
            * n_groups, k // n_groups)) + [k]
    EMPTY = -1
    V = np.full((m, k), EMPTY, dtype=np.float)

    for g in range(n_groups):
        i_start = block_row_indices[g]
        i_end = block_row_indices[g + 1]
        j_start = block_col_indices[g]
        j_end = block_col_indices[g + 1]

        block_height = i_end - i_start
        block_width = j_end - j_start
        block_size = (block_height, block_width)

        B_g = np.random.gamma(1., 1. / beta, size=block_size)
        V[i_start:i_end, j_start:j_end] = B_g

    V[V == EMPTY] = np.random.gamma(
            1., 1. / ((1. - theta) * beta), size=(m, k))[V == EMPTY]
    return V


def generate_factor_matrices(n, m, k, sparsity_degree_in_v=0.5,
                             beta=80, theta=0.8, n_groups=2):
    U = generate_u(n, k, n_groups=n_groups, theta=theta)
    V = generate_v(m, k, sparsity_degree=sparsity_degree_in_v,
                   beta=beta, theta=theta, n_groups=n_groups)
    Lambda = np.dot(U, V.T)

    pi_d = np.ones(m) * .72 # TODO

    # Sample Bernoulli distributions
    ones = np.ones(m, dtype=np.int)
    D = np.random.binomial(ones, pi_d, size=(n, m))
    X = D * Lambda
    return X, U, V
