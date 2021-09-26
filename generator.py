import numpy as np


class Generator(object):
    def __init__(self, function, config, filename='data.csv'):
        self.m = config[0]
        self.n = self.m * 4
        self.x = self.__get_x()
        self.u = self.__get_u(function, config[3])
        self.y = self.__get_y(config[4])
        self.filename: str = filename

    def __get_x(self) -> type(np.ndarray):
        low, high = -1, 1
        test = [np.random.uniform(low, high, self.n) for j in range(self.n)]
        return np.ndarray([[np.random.uniform(low, high, self.n) for j in range(self.n)]
                           for i in range(self.m)])

    def __get_u(self, function, theta) -> type(np.ndarray):
        return [function(i, theta) for i in self.x]

    def __get_y(self, coeff: float):
        y = np.copy(self.u)
        u_mean = np.full(self.n, np.mean(self.u))
        w_squared = np.dot((self.u - u_mean), (self.u - u_mean)) / (self.n - 1)
        noise = coeff * w_squared
        return np.ndarray([i + noise for i in y])

    def __enter__(self):
        return None

    def __exit__(self):
        if self:
            self.__save(self.filename)

    def __save(self, filename):
        with open(filename, 'w') as file:
            file.write('i\t')
            for i in range(1, self.m + 1):
                file.write('x%d\t' % i)
            file.write('u\ty\n')
            for i in range(self.n):
                file.write('{:d}\t'.format(i))
                for j in range(self.m):
                    file.write('{:.17f}\t'.format(self.x[i][j]))

                file.write('{:.17f}\t{:.17f}\n'.format(self.u[i], self.y[i]))
