import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import indices
from scipy.sparse.construct import rand
from sklearn.utils import shuffle
from tqdm import tqdm
import random


class RBF():
    def __init__(self, n=1, drop=2**9-1, weight=1, sigma=0.5, dim=1):
        if dim == 1:
            self.set_centers(n, drop, weight)
            self.set_sigmas(sigma)

    def compute_phi(self, x):
        n, N = self.centers.shape[0], x.shape[0]
        phi = np.zeros((N, n))
        for i in range(N):
            for j in range(n):
                phi[i, j] = np.exp(-(np.linalg.norm(x[i] -
                                                    self.centers[j])**2) / (2*self.sigmas[j]**2))
        phi = np.array(phi)
        return phi

    def batch_learning(self, x, f, x_test, f_test):
        # compute phi
        phi = self.compute_phi(x)
        # find the weights that minimize the total error = |phi W - f|^2
        w = np.linalg.solve(phi.T @ phi, phi.T @ f)
        # Evaluate the error
        phi_test = self.compute_phi(x_test)
        f_hat = phi_test @ w
        error = np.mean(abs(f_hat-f_test))
        return f_hat, error

    def delta_learning(self, X, f, X_test, f_test, lr=0.01, max_iters=15, seed=42):
        weights = np.random.default_rng(seed).normal(
            0, 0.5, (self.n_nodes, X.shape[1]))

        for _ in tqdm(range(max_iters)):
            X, f = shuffle(X, f, random_state=seed)
            for idx, x in enumerate(X):
                try:
                    phi = self.compute_phi(x)
                except:
                    phi = self.compute_phi(np.array([x]))
                f_arr = np.array(f[idx])
                test = f_arr - phi @ weights
                weight_update = lr*(f_arr - phi @ weights).T @ phi
                weights += weight_update.T

        phi_test = self.compute_phi(X_test)
        f_hat = phi_test @ weights
        error = np.mean(abs(f_hat-f_test))
        return f_hat, error

    def competitive_learning(self, X, eta=0.1, neigh=3, max_iter=60, seed=42):
        np.random.seed(seed)
        for i in tqdm(range(max_iter)):
            x = X[np.random.randint(X.shape[0])]
            try:
                distances = [[i_c, np.linalg.norm(
                    x - self.centers[i_c])] for i_c in range(len(self.centers))]
            except:
                distances = [[i_c, np.linalg.norm(np.array(
                    [x]) - np.array([self.centers[i_c]]))] for i_c in range(len(self.centers))]

            distances.sort(key=lambda x: x[1])
            for i in range(neigh):
                coef = eta / (distances[i][1] + 1e-3)
                if coef >= 0.9:
                    coef = 0.9
                self.centers[distances[i][0]] += coef * \
                    (x - self.centers[distances[i][0]])

    def hybrid_learning(self, X, f, X_test, f_test, lr=0.01, max_iters=250, seed=42, eta=0.1, neigh=3):
        self.competitive_learning(X, eta, neigh)
        f_hat, error = self.delta_learning(
            X, f, X_test, f_test, lr, max_iters, seed)
        return f_hat, error

    def set_centers(self, n, drop, weight):
        x_line = np.linspace(0, 2, 9)*np.pi

        if n == 0:
            self.centers = x_line[1::2]
            self.n_nodes = len(self.centers)
            return

        self.centers = np.array([])

        for i in range(len(x_line)-1):
            bit = (drop >> (i+1) & 1) != 0
            self.centers = np.append(self.centers, np.linspace(
                x_line[i], x_line[i+1], num=int(n*bit*weight), endpoint=False))

        self.centers = np.sort(self.centers)
        self.n_nodes = len(self.centers)

    def set_sigmas(self, sigma):
        self.sigmas = np.full(self.n_nodes, sigma)

    def set_centers_from_data(self, x, n_nodes, sigma=1.0, seed=42):
        random.seed(seed)
        self.n_nodes = n_nodes
        indices = random.sample(range(len(x)), n_nodes)
        self.centers = x[indices]
        self.set_sigmas(sigma)
