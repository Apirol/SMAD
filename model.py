from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def get_w_squared(u_: np.array, config):
    u_mean = np.full(100, np.mean(u_))
    return np.dot((u_ - u_mean).transpose(), (u_ - u_mean)) / (100 - 1)


def get_variance(w, coeff):
    return w * coeff


class Model:
    def __init__(self, X, y, function, alpha=0.05):
        self.theta = []
        self.X = X
        self.y = y
        self.y_calc = []
        self.e = []
        self.function = function
        self.variance: int = 0
        self.interval = []
        self.alpha = alpha

    def fit(self, n: int, m: int):
        self.theta = np.dot(np.dot(np.linalg.inv(np.dot(self.X.transpose(), self.X)), self.X.transpose()), self.y)
        self.y_calc = np.dot(self.X, self.theta)
        self.e = self.y - self.y_calc
        self.variance = np.dot(self.e.transpose(), self.e / (n - m))[0][0]

    def predict(self, X) -> list:
        assert len(self.theta) > 0, 'Сначала запустите метод fit'
        return [self.function(i, self.theta) for i in X]

    def get_theta(self):
        return self.theta

    def get_info(self, n: int, m: int, variance=0, ):
        assert len(self.theta) > 0, 'Сначала запустите метод fit'
        F, F_t, res = self.__check(n, m, variance)
        interval = self.__get_interval(n, m)
        theta_significance = self.__check_theta(n, m)
        model_significance = self.__check_model(n, m)
        return F, F_t, res, interval, theta_significance, model_significance

    def __check_model(self, n: int, m: int) -> list:
        res = False
        RSS = np.sum((self.y - self.y_calc)**2)
        yMean = np.full(n, np.mean(self.y))
        TSS = np.sum((self.y - yMean) ** 2)
        F = ((TSS - RSS) / (m - 1)) / (RSS / (n - m))
        F_t = stats.f.ppf(1 - self.alpha, m - 1, n - m)
        if F < F_t:
            res = True
        return [res, TSS, RSS, F, F_t]

    def __check_theta(self,  n: int, m: int) -> list:
        res = []
        F_t = stats.f.ppf(1 - self.alpha, 1, n - m)
        matrix = np.linalg.inv(np.dot(self.X.transpose(), self.X))
        for i in range(m):
            current_F = self.theta[i] ** 2 / (self.variance * matrix[i][i])
            if current_F < F_t:
                res.append(False)
            else:
                res.append(True)
        return res

    def __get_interval(self, n: int, m: int) -> np.array:
        res = np.ndarray((2, m))
        matrix = np.linalg.inv(np.dot(self.X.transpose(), self.X))
        t = abs(stats.t.ppf(1 - self.alpha, n - m))
        for i in range(m):
            res[0][i] = self.theta[i] - t * matrix[i][i]
            res[1][i] = self.theta[i] + t * matrix[i][i]
        return res

    def __check(self, n: int, m: int, variance):
        res: bool
        F = self.variance / variance
        d1, d2 = n - m, 10000
        F_t = stats.f.ppf(1 - 0.05, d1, d2)
        if F <= F_t:
            res = True
        else:
            res = False
        return F, F_t, res


