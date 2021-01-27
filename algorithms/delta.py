import numpy as np


def delta_rule(X, labels, lr=0.001, max_iters=20, seed=42, sequential=0, threshold=10e-4):
    # add bias
    X = np.c_[X, np.ones(X.shape[0])]

    # make row represent one dimension (as in the instruction)
    X = X.T
    dim_num = X.shape[0]
    weight_history = list()
    weights = np.random.default_rng(seed).normal(0, 0.5, dim_num)
    weight_history.append(weights.copy())
    for _ in range(max_iters):
        if sequential:

            for idx, x in enumerate(X.T):
                #weights /= np.linalg.norm(weights,ord=2)
                weight_update = -lr*(weights @ x - np.array(labels[idx])) * x.T
                weights += weight_update

            diff = np.linalg.norm(weights-weight_history[-1], ord=2)
        else:
            weight_update = -lr*(weights @ X - labels) @ X.T
            weights += weight_update
            #weights /= np.linalg.norm(weights,ord=2)
            diff = np.linalg.norm(weights-weight_history[-1], ord=2)
        weight_history.append(weights.copy())
        if diff <= threshold:
            print('break after iteration {}'.format(_))
            break
    return weights, weight_history


def delta_predict(X, weights):
    # add bias
    X = np.c_[X, np.ones(X.shape[0])]
    # predict
    y_pred = X @ weights
    # threshold
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = -1
    return y_pred
