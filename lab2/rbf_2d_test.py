from rbf import RBF
import matplotlib.pyplot as plt
import numpy as np

def experiment(x, y, x_test, y_test, n_nodes, error, n):
    rbf_net = RBF()
    rbf_net.set_centers_from_data(x, n)
    _, err = rbf_net.hybrid_learning(
        x, y, x_test, y_test)
    n_nodes.append(rbf_net.n_nodes)
    error.append(err)

def experiment_nodes(x, y, x_test, y_test):
    error = []
    n_nodes = []
    for i in range(10, 60):
        experiment(x, y, x_test, y_test, n_nodes, error, n=i)

    return error, n_nodes

def plot_error(n_nodes, error, data):
    plt.scatter(n_nodes, error)
    plt.xlabel("number of hidden nodes")
    plt.ylabel("absolute residual error")
    plt.title("Absolute Residual Error vs number of hidden nodes")
    plt.legend()
    plt.savefig(f"images/3.1/{data}_error_{len(n_nodes)}.png")
    plt.show()


def plot_estimate(x, y, x_test, y_test, n_nodes=20):
    rbf_net = RBF()
    rbf_net.set_centers_from_data(x, n_nodes)
    y_hat, error = rbf_net.hybrid_learning(
        x, y, x_test, y_test)
    centers, n_nodes = rbf_net.centers, rbf_net.n_nodes
    # plt.plot(data.x_test, data.y_test, label="Target")
    # plt.plot(data.x_test, y_hat, label="Estimate")
    # plt.scatter(centers, [0]*n_nodes, c="r", label="RBF Centers")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title(f'Target vs Estimated values with {n_nodes} hidden nodes, error= {round(error,5)}')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"images/3.1/{type}/{type}_{n_nodes}.png")
    # plt.show()
    print(error)


data = np.loadtxt('data_lab2/ballist.dat')
test = np.loadtxt('data_lab2/balltest.dat')

x = data[:, :2]
y = data[:, 2:]
x_test = test[:, :2]
y_test = test[:, 2:]

error, n_nodes = experiment_nodes(x, y, x_test, y_test)
plot_error(n_nodes, error, "2d")

plot_estimate(x, y, x_test, y_test, n_nodes=20)



