import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt


class Generator(object):
    def __init__(self, function, config, filename='data.xlsx'):
        self.m = config[0]
        self.n = config[1] * 4
        self.x = self.__get_x()
        self.u = self.__get_u(function, config[2])
        self.y = self.__get_y(config[3])
        self.filename: str = filename

    def draw(self, function: callable(object), theta: list):
        fig = plt.figure('Generator', figsize=(8, 10))
        ax = fig.gca(projection='3d')
        fig.subplots_adjust(bottom=-0.6, top=-0.5, left=-0.05, right=1.05)
        tmp_range = np.linspace(-1, 1, int(sqrt(self.n)))
        x1, x2 = np.meshgrid(tmp_range, tmp_range)
        u = function([x1, x2], theta)
        fig.subplots_adjust(bottom=-0.05, top=1, left=-0.05, right=1.05)
        ax.plot_surface(x1, x2, u, zorder=2, alpha=0.2)
        x1 = [x[0] for x in self.x]
        x2 = [x[1] for x in self.x]
        ax.scatter(x1, x2, self.u, c='red', zorder=1)
        plt.title('Зависимость отклика от двух критериев', fontsize=19)
        plt.xlabel('Первый критерий')
        plt.ylabel('Второй критерий')
        plt.grid(alpha=0.5)
        plt.savefig('lab1.png')
        plt.show()

    def draw_separately(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

        x1 = [x[0] for x in self.x]
        ax1.plot(x1, self.y, 'o', color='black')
        ax1.grid(axis='y')
        ax1.set_xlabel('Первый критерий')
        ax1.set_ylabel('Незашумленный отклик')
        ax1.set_title('Зависимость отклика от первого критерия')
        plt.setp(ax1.get_xticklabels(), rotation=45)

        x2 = [x[1] for x in self.x]
        ax2.plot(x2, self.y, 'o', color='black')

        ax2.grid(axis='y')
        ax2.set_xlabel('Второй критерий')
        ax2.set_ylabel('Незашумленный отклик')
        ax2.set_title('Зависимость отклика от второго критерия')
        plt.tight_layout()
        plt.savefig('lab1_sep.png')
        plt.show()

    def get_w_squared(self):
        u_mean = np.full(self.n, np.mean(self.u))
        return np.dot((self.u - u_mean), (self.u - u_mean)) / (self.n - 1)

    def __get_x(self) -> type(np.ndarray):
        res = np.ndarray((self.m, self.n))
        low, high = -1, 1
        rows = [np.random.uniform(low, high, self.n) for i in range(self.m)]
        for i, row in enumerate(rows):
            res[i] = row
        return res.transpose()

    def __get_u(self, function, theta) -> type(np.ndarray):
        res = np.ndarray(self.n)
        for i, row in enumerate(self.x):
            res[i] = function(row, theta)
        return res

    def __get_y(self, coeff: float):
        y = np.copy(self.u)
        w_squared = self.get_w_squared()
        noise = coeff * w_squared
        return y + noise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self:
            self.__save(self.filename)

    def __save(self, filename):
        data = pd.DataFrame()
        for i in range(self.n):
            current_iter = [{'X1': self.x[i][0], 'X2': self.x[i][1], 'U': self.u[i], 'Y': self.y[i]}]
            data = data.append(current_iter, ignore_index=True)
        #data.to_excel(filename)
