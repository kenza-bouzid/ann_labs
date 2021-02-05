from rbf import RBF
from dataset import SinusData
from dataset import SquareData
import matplotlib.pyplot as plt
import numpy as np

def experiment(data, n_nodes, error, n, weight=1.0):
    rbf_net = RBF(n=n, weight=weight)
    _, err = rbf_net.batch_learning(
        data.x, data.y, data.x_test, data.y_test)
    n_nodes.append(rbf_net.n_nodes)
    error.append(err)

def experiment2(data, n_nodes, error, n):
    rbf_net = RBF(n=n)
    rbf_net.centers = np.linspace(0, 2*np.pi, n)
    rbf_net.n_nodes = n
    rbf_net.set_sigmas(0.5)
    _, err = rbf_net.batch_learning(
        data.x, data.y, data.x_test, data.y_test)
    n_nodes.append(rbf_net.n_nodes)
    error.append(err)

def experiment_nodes(data):
    error = []
    n_nodes = []
    for i in range(8,40):
        # experiment(data, n_nodes, error, n=i, weight=1.0)
        # experiment_delta(data, n_nodes, error, n=i, weight=1.0)
        # experiment2(data, n_nodes, error, n=i)
        experiment2_delta(data, n_nodes, error, n=i)

    return error, n_nodes

def experiment_delta(data, n_nodes, error, n, weight=1.0):
    rbf_net = RBF(n=n, weight=weight)
    _, err = rbf_net.delta_learning(
        data.x, data.y, data.x_test, data.y_test)
    n_nodes.append(rbf_net.n_nodes)
    error.append(err)

def experiment2_delta(data, n_nodes, error, n):
    rbf_net = RBF(n=n)
    rbf_net.centers = np.linspace(0, 2*np.pi, n)
    rbf_net.n_nodes = n
    rbf_net.set_sigmas(0.5)
    _, err = rbf_net.delta_learning(
        data.x, data.y, data.x_test, data.y_test)
    n_nodes.append(rbf_net.n_nodes)
    error.append(err)

def plot_error(n_nodes, error, data):
    plt.scatter(n_nodes, error)
    plt.xlabel("number of hidden nodes")
    plt.ylabel("absolute residual error")
    plt.title("Absolute Residual Error vs number of hidden nodes")
    plt.legend()
    plt.savefig(f"images/3.1/{data}_error2_{len(n_nodes)}.png")
    plt.show()


def plot_estimate(data, type, n=3):
    rbf_net = RBF(n=n)
    y_hat, error = rbf_net.delta_learning(
        data.x, data.y, data.x_test, data.y_test)
    centers, n_nodes = rbf_net.centers, rbf_net.n_nodes
    plt.plot(data.x_test, data.y_test, label="Target")
    plt.plot(data.x_test, y_hat, label="Estimate")
    plt.scatter(centers, [0]*n_nodes, c="r", label="RBF Centers")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f'Target vs Estimated values with {n_nodes} hidden nodes, error= {round(error,5)}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"images/3.1/{type}/{type}_{n_nodes}.png")
    plt.show()


sin_data = SinusData(noise=False)
# error, n_nodes = experiment_nodes(sin_data)
# plot_error(n_nodes, error, "sinus")
# plot_error(n_nodes, error, "sinus_delta")
plot_estimate(sin_data, type="sinus", n=3)


sqr_data = SquareData(noise=True)
# error, n_nodes = experiment_nodes(sqr_data)
# plot_error(n_nodes, error, "square")
# plot_error(n_nodes, error, "square_delta")

plot_estimate(sqr_data, type="square", n=2)


