from math import sqrt
from scipy import stats
import numpy as np
import process_data as pr
import matplotlib.pyplot as plt


def get_variance(u, coeff: float):
    def get_w_squared(u_: np.array):
        u_mean = np.full(100, np.mean(u_))
        return np.dot((u_ - u_mean).transpose(), (u_ - u_mean)) / (len(u) - 1)
    return get_w_squared(u) * coeff


class Model:
    def __init__(self, data, function, config):
        self.function = function
        self.config = config
        self.X, self.y = self.__get_data(data)
        self.y_calc = []
        self.e = []
        self.theta = []
        self.variance: int = 0
        self.interval = []

    def fit(self, n: int, m: int):
        self.theta = np.dot(np.dot(np.linalg.inv(np.dot(self.X.transpose(), self.X)), self.X.transpose()), self.y)
        self.y_calc = np.dot(self.X, self.theta)
        self.e = self.y - self.y_calc
        self.variance = np.dot(self.e.transpose(), self.e / (n - m))[0][0]

    def predict(self, X) -> list:
        assert len(self.theta) > 0, 'Сначала запустите метод fit'
        return [self.function(i, self.theta) for i in X]

    def draw_residuals(self):
        plt.figure(figsize=(15, 10))
        plt.scatter(self.y, self.y - self.y_calc, c='blue', marker='o', label='Training data')
        plt.title('Диаграмма рассеивания остатков отклика', fontsize=22)
        plt.xlabel('Predicted values', fontsize=15)
        plt.ylabel('Residuals', fontsize=15)
        plt.show()

    def draw_intervals(self, intervals: list, x_list: list, xlabel, ylabel, title):
        plt.plot(x_list, intervals[0], label='lower')
        plt.plot(x_list, intervals[1], label='center')
        plt.plot(x_list, intervals[2], label='upper')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def get_info(self, n: int, m: int, variance=0, ):
        assert len(self.theta) > 0, 'Сначала запустите метод fit'
        F, F_t, res = self.__check(n, m, variance)
        interval = self.__get_interval(n, m)
        theta_significance = self.__check_theta(n, m)
        model_significance = self.__check_model(n, m)
        return F, F_t, res, interval, theta_significance, model_significance

    def get_expected_interval(self) -> list:
        assert len(self.theta) > 0, 'Сначала запустите метод fit'
        x1_list = np.linspace(-1, 1, 100)
        x2 = 0
        theta_calc = [num[0] for num in self.theta]
        lower, center, upper = [], [], []
        t = abs(stats.t.ppf(1 - 0.05, 100 - 6))
        for i in range(100):
            x1 = x1_list[i]
            x = [x1, x2]
            fx = self.function(x, theta_calc)
            fx_vector = np.array(pr.get_vector_function(x, theta_calc))
            tmp = np.dot(np.dot(fx_vector.transpose(),
                                np.linalg.inv(np.dot(self.X.transpose(), self.X))), fx_vector)
            sigma = self.variance * (1 + tmp)
            lower.append(fx - t * sigma)
            center.append(fx)
            upper.append(fx + t * sigma)
        return [x1_list, lower, center, upper]

    def get_y_calc_interval(self) -> list:
        assert len(self.theta) > 0, 'Сначала запустите метод fit'
        x2_list = np.linspace(-1, 1, 10)
        x1 = 0
        theta_calc = [num[0] for num in self.theta]
        lower, center, upper = [], [], []
        t = abs(stats.t.ppf(1 - 0.05, 100 - 6))
        for i in range(10):
            x2 = x2_list[i]
            x = [x1, x2]
            fx = self.function(x, theta_calc)
            fx_vector = np.array(pr.get_vector_function(x, theta_calc))
            tmp = sqrt(np.dot(np.dot(fx_vector.transpose(), np.linalg.inv(np.dot(self.X.transpose(), self.X))), fx_vector))
            sigma = self.variance * tmp
            lower.append(fx - t * sigma)
            center.append(fx)
            upper.append(fx + t * sigma)
        return [x2_list, lower, center, upper]

    def __check_model(self, n: int, m: int) -> list:
        res = False
        RSS = np.sum((self.y - self.y_calc)**2)
        yMean = np.full(n, np.mean(self.y))
        TSS = np.sum((self.y - yMean) ** 2)
        F = ((TSS - RSS) / (m - 1)) / (RSS / (n - m))
        F_t = stats.f.ppf(1 - self.config[3], m - 1, n - m)
        if F < F_t:
            res = True
        return [res, TSS, RSS, F, F_t]

    def __check_theta(self,  n: int, m: int) -> list:
        res = []
        F_t = stats.f.ppf(1 - self.config[3], 1, n - m)
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
        t = abs(stats.t.ppf(1 - self.config[3], n - m))
        for i in range(m):
            res[0][i] = self.theta[i] - t * matrix[i][i]
            res[1][i] = self.theta[i] + t * matrix[i][i]
        return res

    def __get_data(self, data):
        signs = data[['X1', 'X2']].to_numpy().transpose()
        X = pr.get_factor_matrix(signs, self.config[1]).transpose()
        y = data[['Y']].to_numpy()
        return X, y

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


