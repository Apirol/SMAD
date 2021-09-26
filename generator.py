import numpy as np
import pandas as pd
import matplotlib.pyplot as lpt


class Generator(object):
    def __init__(self, function, config, filename='data.xlsx'):
        self.m = config[0]
        self.n = config[1] * 4
        self.x = self.__get_x()
        self.u = self.__get_u(function, config[2])
        self.y = self.__get_y(config[3])
        self.filename: str = filename

    def __get_x(self) -> type(np.ndarray):
        res = np.ndarray((self.m, self.n))
        low, high = -1, 1
        rows = [np.random.uniform(low, high, self.n) for i in range(self.m)]
        for i, row in enumerate(rows):
            res[i] = row
        return res.transpose()

    def __get_u(self, function, theta) -> type(np.ndarray):
        return [function(i, theta) for i in self.x]

    def __get_y(self, coeff: float):
        y = np.copy(self.u)
        u_mean = np.full(self.n, np.mean(self.u))
        w_squared = np.dot((self.u - u_mean), (self.u - u_mean)) / (self.n - 1)
        noise = coeff * w_squared
        return y + noise

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self:
            self.__save(self.filename)

    def __save(self, filename):
        data = pd.DataFrame()
        for i in range(self.n):
            current_iter = [{'X1': self.x[i][0], 'X2': self.x[i][1], 'U': self.u[i], 'Y': self.y[i]}]
            data = data.append(current_iter, ignore_index=True)
        data.to_excel(filename)
