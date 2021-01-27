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


def perceptron_learning(X, labels, lr=0.001, max_iters=100, seed=42):
    # add bias
    X = np.c_[X, np.ones(X.shape[0])]
    dim_num = X.shape[1]

    it_num = 0
    weights = np.random.default_rng(seed).normal(0, 0.5, dim_num)
    weight_history = list()
    weight_update = np.ones((1, dim_num))
    while np.any(weight_update != 0):
        weight_history.append(weights.copy())
        weight_update = np.sum(
            list(
                map(
                    lambda x, y: compute_weight_update(x, y, weights=weights), X, labels)),
            axis=0)
        weights += lr*weight_update
        it_num += 1

        if it_num >= max_iters:
          print('Warning: reached maximum numbers of iterations.')
          break

    print(f'Number of epochs: {it_num}')
    return weights, weight_history