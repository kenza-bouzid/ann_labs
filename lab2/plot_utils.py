from rbf import RBF
from dataset import SinusData
from dataset import SquareData
from rbf import CentersSampling
from rbf import LearningMode

import matplotlib.pyplot as plt
import numpy as np


def plot_RBF_grid_search(data, weights= [0.6, 0.8, 1, 1.2], n_nodes=[10, 15, 20, 25], learning_mode=LearningMode.BATCH, centers_sampling = CentersSampling.LINEAR):
    results = {}

    for n in n_nodes:
        for w in weights:
            rbf_net = RBF(centers_sampling, n_nodes=n, n_inter=n, sigma=w)
            if learning_mode == LearningMode.BATCH:
                y_hat, error = rbf_net.batch_learning(
                    data.x, data.y, data.x_test, data.y_test)
            else:
                y_hat, error = rbf_net.delta_learning(
                    data.x, data.y, data.x_test, data.y_test, max_iters=20, lr=0.001)
            results[(rbf_net.n_nodes, w)] = error

    keys = np.array(list(results.keys()))
    plt.scatter(keys[:, 0], keys[:, 1], c=list(
        results.values()), cmap='tab20b', s=200)
    plt.xlabel('units')
    plt.ylabel('width')
    plt.title('Absolute residual error for different RBF configurations')
    plt.colorbar()
    plt.show()
    return results


def error_estimate_batch(data, n_nodes=20, sigma=0.5):
    rbf_net = RBF(CentersSampling.LINEAR, n_nodes=n_nodes, sigma=0.5)
    y_hat, error = rbf_net.batch_learning(
        data.x, data.y, data.x_test, data.y_test)
    print(f'Error for batch learning: {error}')


def error_estimate_delta(data, n_nodes=20, sigma=0.5):
    errors = []
    for seed in range(20):
        rbf_net = RBF(CentersSampling.LINEAR, seed=seed,
                      n_nodes=n_nodes, sigma=sigma)
        y_hat, error = rbf_net.delta_learning(
            data.x, data.y, data.x_test, data.y_test, seed=seed, max_iters=100, lr=0.1)
        errors.append(error)

    print(
        f'Error for delta rule avg: {np.mean(errors)}, variance: {np.var(errors)}')


def plot_error_delta(data, n_nodes=10, lr=0.01, max_iters=100):
    rbf_net = RBF(CentersSampling.LINEAR, n_nodes=n_nodes)
    y_hat, error = rbf_net.delta_learning(
        data.x, data.y, data.x_test, data.y_test, max_iters=max_iters, lr=lr)
    print(f'Error on the test set: {error}')
    plt.plot(rbf_net.training_errors)
    plt.xlabel("Sample number")
    plt.ylabel('absolute residual error')
    plt.title(f'Train error for delta rule with {lr} lr')
    plt.show()


def plot_estimate(data, centers_sampling=CentersSampling.LINEAR, learning_type='batch', n_nodes=20, delta_max_iters=100, sigma=0.5, delta_lr=0.1, weight=1):
    rbf_net = RBF(centers_sampling, n_nodes=n_nodes,
                  sigma=sigma, weight=weight)
    if learning_type == 'batch':
        y_hat, error = rbf_net.batch_learning(
            data.x, data.y, data.x_test, data.y_test)
    else:
        y_hat, error = rbf_net.delta_learning(
            data.x, data.y, data.x_test, data.y_test, max_iters=delta_max_iters, lr=delta_lr)
    centers, n_nodes = rbf_net.centers, rbf_net.n_nodes
    plt.plot(data.x_test, data.y_test, label="Target")
    plt.plot(data.x_test, y_hat, label="Estimate")
    plt.scatter(centers, [0]*n_nodes, c="r", label="RBF Centers")
    plt.xlabel("x")
    plt.ylabel("y")
    if sigma != 0.5:
        plt.title(
            f'{learning_type}, {n_nodes} units, {sigma} width, error= {round(error,5)}')
    else:
        plt.title(
            f'{learning_type} learning, {n_nodes} RBF units, error= {round(error,5)}')
    plt.legend()
    plt.grid(True)
    plt.show()
