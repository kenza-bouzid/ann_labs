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


def generate_data_ex3(n=100, mA=[1, 0.3], sigA=0.2, mB=[0, -0.1], sigB=0.3, shuffle=True, negative_class=-1, seed=42):
    dim_num = 2
    negative_mA = [-1, 0.3]
    first_row = np.random.default_rng(seed).multivariate_normal(
        negative_mA, np.eye(dim_num)*sigA, 50)
    second_row = np.random.default_rng(seed).multivariate_normal(
        mA, np.eye(dim_num)*sigA, 50)
    classA_data = np.concatenate((first_row, second_row), axis=0)
    classA = np.c_[classA_data, np.ones(n)*negative_class]
    classB = np.c_[np.random.default_rng(seed).multivariate_normal(
        mB, np.eye(dim_num)*sigB, n), np.ones(n)]
    data = np.concatenate((classA, classB), axis=0)
    if shuffle:
        np.random.shuffle(data)
    return data[:, :2], data[:, 2]
