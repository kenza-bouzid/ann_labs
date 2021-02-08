from rbf import RBF
import matplotlib.pyplot as plt
import numpy as np
import utils_test as ut
import importlib
import numpy as np
from collections import namedtuple


# error, n_nodes = experiment_nodes(x, y, x_test, y_test)
# plot_error(n_nodes, error, "2d")

# plot_estimate(x, y, x_test, y_test, n_nodes=20)

Data = namedtuple('Data', 'x y x_test y_test')
train = np.loadtxt('data_lab2/ballist.dat')
test = np.loadtxt('data_lab2/balltest.dat')
data = Data(train[:, :2], train[:, 2:], test[:, :2], test[:, 2:])

ut.plot_RBF_predictions_3d(data, n=25, sigma=0.6, neigh=10, axis=0)
ut.plot_RBF_predictions_3d(data, n=25, sigma=0.6, neigh=10, axis=1)


