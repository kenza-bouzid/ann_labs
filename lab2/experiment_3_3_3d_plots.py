#%%
from rbf import CentersSampling
from rbf import RBF
from rbf import LearningMode
import matplotlib.pyplot as plt
import numpy as np
import utils_test as ut
import plot_utils as pu
import numpy as np
from collections import namedtuple
import importlib
#%%
Data = namedtuple('Data', 'x y x_test y_test')
train = np.loadtxt('data_lab2/ballist.dat')
test = np.loadtxt('data_lab2/balltest.dat')
data = Data(train[:, :2], train[:, 2:], test[:, :2], test[:, 2:])
#%%
importlib.reload(ut)
ut.plot_RBF_centers_2d(data, n=4, sigma=1.0, neigh=1)
#%%
importlib.reload(pu)
results = pu.plot_RBF_grid_search(data, weights=[0.1, 0.2, 0.3, 0.4], n_nodes=[1, 2, 3, 4, 5, 6, 7], learning_mode=LearningMode.DELTA, centers_sampling = CentersSampling.UNIFORM)
print(results)
#%%
sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}
sorted_results
#%%
importlib.reload(ut)
y_hat, error, centers = ut.plot_surface_predictions(data, n=2, sigma=0.4, neigh=2, max_iter=50)
#%%
plt.scatter(data.y_test[:, 0], data.y_test[:, 1], label="Target")
plt.scatter(y_hat_[:, 0], y_hat_[:, 1], label="Predictions")
plt.scatter(centers_[:, 0], centers_[:, 1], label="RBF Centers")
plt.xlabel("distance")
plt.ylabel("height")
plt.title(
    f"Distribution of true target and predictions")
plt.legend()
plt.savefig(
    f"images/2d/pred_targe.png")
plt.show()
#%%
importlib.reload(ut)
y_hat_, error_, centers_ = ut.plot_surface_predictions(
    data, n=2, sigma=0.4, neigh=4, max_iter=200, lr=0.001, centers_sampling=CentersSampling.UNIFORM)
