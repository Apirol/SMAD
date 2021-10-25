import numpy as np


def get_vector_function(X, theta) -> list:
    return [X[0] * theta[0], X[0] ** 2 * theta[1], X[1] * theta[2],
            X[1] ** 3 * theta[3], X[0] * X[1] * theta[4], X[1] ** 2 * theta[5]]


def get_factor_matrix(signs, m) -> np.array:
    X = np.ones((m, 100))
    X[4] = signs[0] * signs[1]
    X[0] = signs[0]
    signs[0] = signs[0] ** 2
    X[1] = signs[0]
    X[2] = signs[1]
    signs[1] = signs[1] ** 3
    X[3] = signs[1]
    return X
