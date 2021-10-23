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

 # TODO 1: Redirect to Menu
    def lab_exercise(self, config, data):
        plt.figure(figsize=(15, 10))
        plt.scatter(self.y, self.y - self.y_calc, c='blue', marker='o', label='Training data')
        plt.title('Диаграмма рассеивания остатков отклика', fontsize=22)
        plt.xlabel('Predicted values', fontsize=15)
        plt.ylabel('Residuals', fontsize=15)
        plt.show()

        print('СГЕНЕРИРОВАННЫЕ ДАННЫЕ')
        print(data[['X1', 'X2', 'Y']])
        print('')

        E = pd.Series([num[0] for num in self.e])
        Y_calc = pd.Series([num[0] for num in self.y_calc])
        data['E'] = E
        data['Y_calc'] = Y_calc
        print('Истинные значения функции, найденные значения отклика и вектор ошибки(**U,Y_CALC, Y - Y_CALC**)')
        print(data[['U', 'Y_calc', 'Y', 'E']])
        print('')
        data.to_excel('report.xlsx')

        theta = pd.DataFrame()
        theta['True'] = pd.Series(config[2])
        theta['Calc'] = pd.Series([num[0] for num in self._theta])
        print('Вектор истинных коэффициентов, вектор найденных коеффициентов')
        print(theta)
        print('')
        theta.to_excel('theta.xlsx')

        u = data[['U']].to_numpy()
        u = np.array([num[0] for num in u])
        print('Дисперсия ошибки незашумленного отлика и дисперсия, полученная на основе остаточной суммы квадратов')
        print(f"Var is {get_variance(get_w_squared(u, config), config[3])} \nVar_calc is {self.variance[0][0]}")
        print('')

        info = self.get_info(len(self.y), len(config[2]), get_variance(get_w_squared(u, config), config[3]))
        print('Значения F при проверке гипотезы об адекватности модели')
        if info[2]:
            print(f"Гипотеза не отвергается\nF is {info[0][0][0]}\nF_T is {info[1]}")
        else:
            print("Полученная модель неадекватна")

    def fit(self, n: int, m: int):
        self._theta = np.dot(np.dot(np.linalg.inv(np.dot(self.X.transpose(), self.X)), self.X.transpose()), self.y)
        self.y_calc = np.dot(self.X, self._theta)
        self.e = self.y - self.y_calc
        self.variance = np.dot(self.e.transpose(), self.e / (n - m))[0][0]

    def predict(self, X) -> list:
        assert len(self._theta) > 0, 'Сначала запустите метод fit'
        return [self.function(i, self._theta) for i in X]

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


