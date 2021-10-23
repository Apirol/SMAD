from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_w_squared(u_: np.array, config):
    u_mean = np.full(100, np.mean(u_))
    return np.dot((u_ - u_mean).transpose(), (u_ - u_mean)) / (100 - 1)


def get_variance(w, coeff):
    return w * coeff


class Model:
    def __init__(self, X, y, function):
        self._theta = []
        self.X = X
        self.y = y
        self.y_calc = []
        self.e = []
        self.function = function
        self.variance: int = 0

    def fit(self, n: int, m: int):
        self._theta = np.dot(np.dot(np.linalg.inv(np.dot(self.X.transpose(), self.X)), self.X.transpose()), self.y)
        self.y_calc = np.dot(self.X, self._theta)
        self.e = self.y - self.y_calc
        self.variance = np.dot(self.e.transpose(), self.e / (n - m))[0][0]

    def predict(self, X) -> list:
        assert len(self._theta) > 0, 'Сначала запустите метод fit'
        return [self.function(i, self._theta) for i in X]

    def get_theta(self):
        return self._theta

    def get_info(self, n: int, m: int, variance):
        assert len(self._theta) > 0, 'Сначала запустите метод fit'
        F, F_t, res = self.__check(n, m, variance)
        return F, F_t, res

    def __check(self, n: int, m: int, variance):
        assert len(self._theta) > 0, 'Сначала запустите метод fit'
        res: bool
        F = self.variance / variance
        d1, d2 = n - m, 10000
        F_t = stats.f.ppf(1 - 0.05, d1, d2)
        if F <= F_t:
            res = True
        else:
            res = False
        return F, F_t, res


