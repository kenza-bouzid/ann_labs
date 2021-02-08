import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
import random
from enum import Enum


class CentersSampling(Enum):
    LINEAR = 0 
    WEIGHTED = 1
    DATA = 2
    RANDOM = 3

class LearningMode(Enum):
    BATCH = 0
    DELTA = 1
    HYBRID = 2

class RBF():
    def __init__(self, centers_sampling, n_nodes=20, n_inter=1, drop=2**9-1, weight=1.0, x=None, seed=42, sigma=0.5):
        
        if centers_sampling == CentersSampling.LINEAR:
            self.set_linear_centers(n_nodes)

        elif centers_sampling == CentersSampling.WEIGHTED:
            self.set_centers_weighted(n_inter, drop, weight)

        elif centers_sampling == CentersSampling.DATA:
            self.set_centers_from_data(x, n_nodes, seed)

        elif centers_sampling == CentersSampling.RANDOM:
            np.random.seed(seed)
            self.n_nodes = n_nodes
            self.centers = np.random.rand((n_nodes)) * 2*np.pi

        self.sigmas = np.full(self.n_nodes, sigma)
        self.training_errors = []
    
    def compute_phi(self, x):
        n, N = self.centers.shape[0], x.shape[0]
        phi = np.zeros((N, n))
        for i in range(N):
            for j in range(n):
                phi[i, j] = np.exp(-(np.linalg.norm(x[i] -
                                                    self.centers[j])**2) / (2*self.sigmas[j]**2))
        phi = np.array(phi)
        return np.c_[np.ones(phi.shape[0]), phi]

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
            0, 0.5, (self.n_nodes+1, X.shape[1]))

        self.training_errors = []

        for _ in tqdm(range(max_iters)):
            X, f = shuffle(X, f, random_state=seed)
            for idx, x in enumerate(X):
                try:
                    phi = self.compute_phi(x)
                except:
                    phi = self.compute_phi(np.array([x]))
                f_arr = np.array(f[idx])
                weight_update = lr*(f_arr - phi @ weights).T @ phi
                weights += weight_update.T

                phi_train = self.compute_phi(X)
                f_hat_train = phi_train @ weights
                self.training_errors.append(np.mean(abs(f_hat_train-f)))

        phi_test = self.compute_phi(X_test)
        f_hat = phi_test @ weights
        error = np.mean(abs(f_hat-f_test))
        return f_hat, error

    def competitive_learning(self, X, eta=0.1, neigh=3, max_iter=60, seed=42):
        np.random.seed(seed)
        annealing_factor = int(max_iter / neigh)
        for i in tqdm(range(max_iter)):
            x = X[np.random.randint(X.shape[0])]
            try:
                distances = [[i_c, np.linalg.norm(
                    x - self.centers[i_c])] for i_c in range(len(self.centers))]
            except:
                distances = [[i_c, np.linalg.norm(np.array(
                    [x]) - np.array([self.centers[i_c]]))] for i_c in range(len(self.centers))]

            distances.sort(key=lambda x: x[1])    
            for n in range(neigh):
                coef = eta / (distances[n][1] + 1)
                self.centers[distances[n][0]] += coef * \
                    (x - self.centers[distances[n][0]])
            if (i+1) % annealing_factor == 0:
                neigh = 1 if neigh <= 2 else neigh-1

    def hybrid_learning(self, X, f, X_test, f_test, lr=0.1, max_iters=20, seed=42, eta=0.1, neigh=3):
        self.competitive_learning(X, eta, neigh)
        f_hat, error = self.delta_learning(
            X, f, X_test, f_test, lr, max_iters, seed)
        return f_hat, error

    def set_centers_weighted(self, n_inter, drop, weight):
        x_line = np.linspace(0, 2, 9) * np.pi

        if n_inter == 0:
            self.centers = x_line[1::2]
            self.n_nodes = len(self.centers)
            return

        self.centers = np.array([])

        for i in range(len(x_line)-1):
            bit = (drop >> (i+1) & 1) != 0
            self.centers = np.append(self.centers, np.linspace(
                x_line[i], x_line[i+1], num=int(n_inter*bit*weight), endpoint=False))

        self.centers = np.sort(self.centers)
        self.n_nodes = len(self.centers)
    
    def set_linear_centers(self, n_nodes):
        self.centers = np.linspace(0, 2*np.pi, n_nodes)
        self.n_nodes = n_nodes

    def set_centers_from_data(self, x, n_nodes, seed=42):
        random.seed(seed)
        self.n_nodes = n_nodes
        indices = random.sample(range(len(x)), n_nodes)
        self.centers = x[indices]
        

