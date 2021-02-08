from rbf import RBF
from rbf import LearningMode
from rbf import CentersSampling
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def experiment(data, learning_mode, centers_sampling, n_nodes=None, error=None, n=20, n_iter=3, weight=1.0, drop=2**9-1, sigma=1.0, neigh=1, max_iter=20, lr=0.1):

    rbf_net = RBF(centers_sampling, n_nodes=n, n_inter=n_iter,
                  weight=weight, drop=drop, x=data.x, sigma=sigma)

    if learning_mode == LearningMode.BATCH:
        y_hat, err = rbf_net.batch_learning(
            data.x, data.y, data.x_test, data.y_test)
    elif learning_mode == LearningMode.DELTA:
        y_hat, err = rbf_net.delta_learning(
            data.x, data.y, data.x_test, data.y_test, lr=lr, max_iter=max_iter)
    else:
        y_hat, err = rbf_net.hybrid_learning(
            data.x, data.y, data.x_test, data.y_test, lr=lr, neigh=neigh, max_iters=max_iter)

    if n_nodes != None and error != None:
        n_nodes.append(rbf_net.n_nodes)
        error.append(err)

    return y_hat, err, rbf_net


def experiment_nodes(data, learning_mode, centers_sampling, weight=1.0, drop=2**9-1, sigma=1.0):
    error = []
    n_nodes = []
    start, end = 6, 30

    if centers_sampling == CentersSampling.WEIGHTED:
        start, end = 0, 7

    # if centers_sampling == CentersSampling.DATA:
    #     start, end = 20, 60

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
    plt.title(f"Absolute Residual Error vs number of hidden nodes - {data}")
    plt.grid(True)
    plt.savefig(
        f"images/error/{data}_{learning_mode.name}_{centers_sampling.name}.png")
    plt.show()

def plot_estimate(data, type, learning_mode, centers_sampling, n=20, n_iter=3, weight=1.0, drop=2**9-1, sigma=1.0, plot_convergence=False, neigh=1):

    y_hat, error, rbf_net = experiment(
        data, learning_mode, centers_sampling, n=n, n_iter=n_iter, weight=weight, drop=drop, sigma=sigma, neigh=neigh)

    centers, n_nodes = rbf_net.centers, rbf_net.n_nodes
    plt.plot(data.x_test, data.y_test, label="Target")
    plt.plot(data.x_test, y_hat, label="Prediction")
    plt.scatter(centers, [0]*n_nodes, c="r", label="RBF Centers")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        f"Target vs Predictions, {n_nodes} hidden nodes, {sigma} sigma, error= {round(error,5)}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"images/{type}/{type}_{learning_mode.name}_{centers_sampling.name}_{sigma}_{n_nodes}.png")
    plt.show()

    if plot_convergence:
        plt.plot(rbf_net.training_errors)
        plt.xlabel("Sample number")
        plt.ylabel('absolute residual error')
        plt.title(f'Train error for RBF network with CL')
        plt.show()

def plot_RBF_centers_2d(data, n=4, sigma=1.0, neigh=1):

    y_hat, error, rbf_net = experiment(
        data, LearningMode.HYBRID, CentersSampling.UNIFORM, n_iter=n, sigma=sigma, neigh=neigh)

    centers, n_nodes = rbf_net.centers, rbf_net.n_nodes
    plt.scatter(data.x[:, 0], data.x[:, 1], label="Patterns")
    plt.scatter(centers[:, 0], centers[:, 1], label="RBF Centers")
    plt.xlabel("angle")
    plt.ylabel("velocity")
    plt.title(
        f"Position of RBF centers, {n} nodes, {sigma} sigma")
    plt.legend()
    plt.savefig(
        f"images/2d/centers_{n}_{sigma}.png")
    plt.show()

def plot_RBF_predictions_3d(data, y_hat, error, centers, n_nodes, axis=0):
    
    sns.set(style="darkgrid")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data.x_test[:, 0]
    y = data.x_test[:, 1]
    z = data.y_test[:, axis]

    ax.set_xlabel("angle")
    ax.set_ylabel("velocity")
    zlabel = 'distance' if axis == 0 else 'height'
    ax.set_zlabel(zlabel)

    s1 = ax.plot_trisurf(x, y, z, label="Target")
    s1._facecolors2d=s1._facecolors3d
    s1._edgecolors2d = s1._edgecolors3d
    s2 = ax.plot_trisurf(x, y, y_hat[:, axis], label="Prediction")
    s2._facecolors2d = s2._facecolors3d
    s2._edgecolors2d = s2._edgecolors3d
    ax.scatter(centers[:, 0], centers[:, 1], np.zeros(
        (centers.shape[0])), label='centers', c='g')
    plt.title(
        f'RBF predictions for {zlabel}, {n_nodes} nodes, error={round(error,6)}')
    ax.legend()
    plt.show()


def plot_surface_predictions(data, n=4, sigma=1.0, neigh=1, max_iter=100, lr=0.1, centers_sampling=CentersSampling.UNIFORM):
    y_hat, error, rbf_net = experiment(
        data, LearningMode.HYBRID, centers_sampling, n_iter=n, sigma=sigma, neigh=neigh, max_iter=max_iter, lr=lr)
    centers, n_nodes = rbf_net.centers, rbf_net.n_nodes

    plot_RBF_predictions_3d(data, y_hat, error, centers, n_nodes, axis=0)
    plot_RBF_predictions_3d(data, y_hat, error, centers, n_nodes, axis=1)
    return y_hat, error, centers

def error_experiment(data, type, learning_mode, centers_sampling, weight=1.0, drop=2**9-1, sigma=1.0):
    print("HERE", learning_mode.name)
    error, n_nodes = experiment_nodes(data, learning_mode=learning_mode,
                                      centers_sampling=centers_sampling, weight=weight, drop=drop, sigma=sigma)
    plot_error(n_nodes, error, type, learning_mode=learning_mode,
               centers_sampling=centers_sampling)
    print(f"Optimal n_nodes RBF for {type}!")
    get_optimal_n_nodes(error, n_nodes)
