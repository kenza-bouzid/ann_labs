import numpy as np
from dataset import SinusData
from dataset import SquareData
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def rbf_kernel(centers, sigmas, x):
    return np.exp(-(x-centers)**2 / 2*sigmas**2)

def rbf_kernel_matrix(centers, sigmas, x):
  centers_matrix = np.repeat(centers[None], x.shape[0], axis=0)
  x_matrix = np.repeat(x[None], centers.shape[0], axis=0).T
  return np.exp(-(x_matrix-centers_matrix)**2 / 2*sigmas**2)


def delta(centers, X, y, lr=0.01, max_iters=250, seed=42):
  n_nodes = centers.shape[0]
  weights = np.random.default_rng(seed).normal(0, 0.5, n_nodes)
  sigmas = np.array([0.1 for i in range(n_nodes)])
  for _ in range(max_iters):
    X, y = shuffle(X, y)
    for idx, x in enumerate(X):
      phi = rbf_kernel_matrix(centers, sigmas, np.array([x]))
      weight_update = (lr*(np.array(y[idx]) - phi @ weights[:,None]) @ phi).flatten()
      weights += weight_update
  return weights

sin_data = SinusData(noise=True)
sqr_data = SquareData(noise=True)

n_nodes = 40
sigmas = np.array([1 for i in range(n_nodes)])
centers = np.array([i/6 for i in range(n_nodes)])

x = sin_data.x
y = sin_data.y
x_test = sin_data.x_test
y_test = sin_data.y_test

# x = sqr_data.x
# y = sqr_data.y
# x_test = sqr_data.x_test
# y_test = sqr_data.y_test

weights = delta(centers, x, y)

rbf_out = rbf_kernel_matrix(centers, sigmas, x_test)

print('error', np.sum(np.abs(rbf_out @ weights - y_test)))

y_pred = rbf_out @ weights

plt.scatter(x_test, y_pred)
plt.scatter(centers, [0 for i in range(len(centers))])
plt.show()
plt.scatter(x_test, y_test)
plt.show()