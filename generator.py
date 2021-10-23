import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt


class Generator(object):
    def __init__(self, function, config):
        self.function = function
        self.config = config
        self._m = config[1]
        self._n = config[1] * 20
        self.x = []
        self.u = []
        self.y = []
        self.e = []

    def generate(self):
        self.x = self.__get_x()
        self.u = self.__get_u()
        self.e = self.__get_e()
        self.y = self.__get_y()

    def get_variance(self, coeff):
        assert len(self.x) > 0, 'Сначала запустите метод generate'
        return self.get_w_squared() * coeff

    def get_w_squared(self):
        assert len(self.x) > 0, 'Сначала запустите метод generate'
        u_mean = np.full(self._n, np.mean(self.u))
        return np.dot((self.u - u_mean).transpose(), (self.u - u_mean)) / (self._n - 1)

    def save(self, filename='data.xlsx'):
        assert len(self.x) > 0, 'Сначала запустите метод generate'
        data = pd.DataFrame()
        for i in range(self._n):
            current_iter = [{'X1': self.x[i][0], 'X2': self.x[i][1], 'U': self.u[i], 'Y': self.y[i]}]
            data = data.append(current_iter, ignore_index=True)
        data.to_excel(filename)

    def draw(self):
        assert len(self.x) > 0, 'Сначала запустите метод generate'
        fig = plt.figure('Generator', figsize=(20, 10))
        ax = fig.gca(projection='3d')
        fig.subplots_adjust(bottom=-0.05, top=1, left=-0.05, right=1.05)
        tmp_range = np.linspace(-1, 1, int(sqrt(self._n)))
        x1, x2 = np.meshgrid(tmp_range, tmp_range)
        u = self.function([x1, x2], self.config[2])
        ax.plot_surface(x1, x2, u, zorder=2, alpha=0.2)
        x1 = [x[0] for x in self.x]
        x2 = [x[1] for x in self.x]
        ax.scatter(x1, x2, self.u, c='red', zorder=1)
        plt.title('Сечение для x1, x2, u', fontsize=19)
        plt.xlabel('Первый критерий')
        plt.ylabel('Второй критерий')
        plt.show()

    def draw_separately(self):
        grid = plt.GridSpec(1, 2, wspace=.25, hspace=.25)

        x1 = [x[0] for x in self.x]
        x2 = [x[1] for x in self.x]

        ax1 = plt.subplot(grid[0, 0])
        ax1.set_xlabel('Первый критерий')
        ax1.set_ylabel('Незашумленный отклик')
        ax1.set_title('Диаграмма рассеивания x1 и u', fontsize=10)
        plt.plot(x1, self.u, 'o', color='blue')

        ax2 = plt.subplot(grid[0, 1])
        ax2.set_xlabel('Второй критерий')
        ax2.set_ylabel('Незашумленный отклик')
        ax2.set_title('Диаграмма рассеивания x2 и u', fontsize=10)
        plt.plot(x2, self.u, 'o', color='blue')

        plt.show()

    def __get_x(self) -> type(np.ndarray):
        low, high = -1, 1
        rows = [np.random.uniform(low, high, self._n) for i in range(2)]
        return np.array(rows).transpose()

    def __get_u(self) -> type(np.array):
        res = np.ndarray(self._n)
        for i, row in enumerate(self.x):
            res[i] = self.function(row, self.config[2])
        return res

    def __get_e(self):
        np.random.seed(42)
        return np.random.normal(0, sqrt(self.get_variance(self.config[3])), self._n)

    def __get_y(self) -> type(np.array):
        return np.array([u + self.e[i] for i, u in enumerate(self.u)])