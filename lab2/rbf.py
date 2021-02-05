import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


class RBF():
    def __init__(self, n=1, drop=2**9-1, weight=1, sigma=0.5):

        self.set_centers(n, drop, weight)
        self.set_sigmas(sigma)

    def compute_phi(self, x):
        centers_matrix = np.repeat(self.centers[None], x.shape[0], axis=0)
        x_matrix = np.repeat(x[None], self.centers.shape[0], axis=0).T
        phi = np.exp(-(x_matrix-centers_matrix)**2 / (2*self.sigmas**2))
        plt.show()
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

    def delta_learning(self, X, f, X_test, f_test, lr=0.01, max_iters=250, seed=42):
        weights = np.random.default_rng(seed).normal(0, 0.5, self.n_nodes)
        for _ in range(max_iters):
            X, f = shuffle(X, f)
            for idx, x in enumerate(X):
                phi = self.compute_phi(np.array([x]))
                weight_update = (
                    lr*(np.array(f[idx]) - phi @ weights[:, None]) @ phi).flatten()
                weights += weight_update

        phi_test = self.compute_phi(X_test)
        f_hat = phi_test @ weights
        error = np.mean(abs(f_hat-f_test))
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
