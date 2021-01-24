import numpy as np


def generate_data(n, mA, sigA, mB, sigB, shuffle=True, negative_class=-1, seed=42):
    dim_num = 2
    classA = np.c_[np.random.default_rng(seed).multivariate_normal(
        mA, np.eye(dim_num)*sigA, n), np.ones(n)*negative_class]
    classB = np.c_[np.random.default_rng(seed).multivariate_normal(
        mB, np.eye(dim_num)*sigB, n), np.ones(n)]
    data = np.concatenate((classA, classB), axis=0)
    if shuffle:
        np.random.shuffle(data)
    return data[:, :2], data[:, 2]
