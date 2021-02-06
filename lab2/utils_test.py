from rbf import RBF
from rbf import LearningMode
from rbf import CentersSampling
import matplotlib.pyplot as plt
import numpy as np


def experiment(data, learning_mode, centers_sampling, n_nodes=None, error=None, n=20, n_iter=3, weight=1.0, drop=2**9-1, sigma=1.0):

    rbf_net = RBF(centers_sampling, n_nodes=n, n_inter=n_iter,
                  weight=weight, drop=drop, x=data.x, sigma=sigma)

    if learning_mode == LearningMode.BATCH:
        y_hat, err = rbf_net.batch_learning(
            data.x, data.y, data.x_test, data.y_test)
    elif learning_mode == LearningMode.DELTA:
        y_hat, err = rbf_net.delta_learning(
            data.x, data.y, data.x_test, data.y_test)
    else:
        y_hat, err = rbf_net.hybrid_learning(
            data.x, data.y, data.x_test, data.y_test)

    if n_nodes!=None and error!=None:
        n_nodes.append(rbf_net.n_nodes)
        error.append(err)

    return y_hat, err, rbf_net.centers, rbf_net.n_nodes


def experiment_nodes(data, learning_mode, centers_sampling, weight=1.0, drop=2**9-1, sigma=1.0):
    error = []
    n_nodes = []
    start, end = 4, 40

    if centers_sampling == CentersSampling.WEIGHTED:
        start, end = 0, 7

    for i in range(start, end):
        experiment(data, learning_mode, centers_sampling,
                   n_nodes, error, n=i, n_iter=i, weight=weight, drop=drop, sigma=sigma)

    return error, n_nodes


def get_optimal_n_nodes(error, n_nodes):
    thresholds = [1e-1, 1e-2, 1e-3]
    optimal = []

    for i, e in enumerate(error):

        if (e < thresholds[0] and len(optimal) < 1) or \
           (e < thresholds[1] and len(optimal) < 2) or \
           (e < thresholds[2] and len(optimal) < 3):
            optimal.append({"n_nodes": n_nodes[i], "err": error[i]})

        if len(optimal) == 3:
            break

    if optimal:
        for i in range(len(optimal)):
            print(f"{thresholds[i]}{optimal[i]}")
    else:
        print("N_nodes not found for the given thresolds!\n")
    print(
        f"min_error={min(error)} n_nodes={n_nodes[np.argmin(error)]}\n")


def plot_error(n_nodes, error, data, learning_mode, centers_sampling):
    print(data, learning_mode.name, centers_sampling.name)
    plt.scatter(n_nodes, error)
    plt.xlabel("number of hidden nodes")
    plt.ylabel("absolute residual error")
    plt.title("Absolute Residual Error vs number of hidden nodes")
    plt.grid(True)
    plt.savefig(
        f"images/error/{data}_{learning_mode.name}_{centers_sampling.name}.png")
    plt.show()


def plot_estimate(data, type, learning_mode, centers_sampling, n=20, n_iter=3, weight=1.0, drop=2**9-1, sigma=1.0):

    y_hat, error, centers, n_nodes = experiment(
        data, learning_mode, centers_sampling, n=n, n_iter=n_iter, weight=weight, drop=drop, sigma=sigma)

    plt.plot(data.x_test, data.y_test, label="Target")
    plt.plot(data.x_test, y_hat, label="Estimate")
    plt.scatter(centers, [0]*n_nodes, c="r", label="RBF Centers")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        f"Target vs Estimated values with {n_nodes} hidden nodes, error= {round(error,5)}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"images/{type}/{type}_{learning_mode.name}_{centers_sampling.name}.png")
    plt.show()


def error_experiment(data, type, learning_mode, centers_sampling, weight=1.0, drop=2**9-1, sigma=1.0):

    error, n_nodes = experiment_nodes(data, learning_mode=learning_mode,
                                      centers_sampling=centers_sampling, weight=weight, drop=drop, sigma=sigma)
    plot_error(n_nodes, error, type, learning_mode=learning_mode,
               centers_sampling=centers_sampling)
    print(f"Optimal n_nodes RBF for {type}!")
    get_optimal_n_nodes(error, n_nodes)
