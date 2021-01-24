import numpy as np


def step_function(y_prim, negative_class=-1):
    if y_prim > 0:
        return 1

    return negative_class


def compute_weight_update(x, y, weights):
    y_pred = step_function(np.matmul(weights.T, x))
    if y == y_pred:
        return np.zeros(x.shape[0])

    return y * x


def delta_rule(X, labels, lr=0.001, max_iters=20, seed=42):
    # add bias
    X = np.c_[X, np.ones(X.shape[0])]

    # make row represent one dimension (as in the instruction)
    X = X.T
    dim_num = X.shape[0]

    weights = np.random.default_rng(seed).normal(0, 0.5, dim_num)
    for i in range(max_iters):
        weight_update = -lr*(weights @ X - labels) @ X.T
        weights += weight_update

    return weights
