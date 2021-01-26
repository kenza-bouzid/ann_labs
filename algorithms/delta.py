import numpy as np


def delta_rule(X, labels, lr=0.001, max_iters=20, seed=42):
    # add bias
    X = np.c_[X, np.ones(X.shape[0])]

    # make row represent one dimension (as in the instruction)
    X = X.T
    dim_num = X.shape[0]
    weight_history = list()
    weights = np.random.default_rng(seed).normal(0, 0.5, dim_num)
    for i in range(max_iters):
        weight_history.append(weights.copy())
        weight_update = -lr*(weights @ X - labels) @ X.T
        weights += weight_update

    return weights, weight_history
