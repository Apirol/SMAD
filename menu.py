import numpy as np
from scipy import stats

from exercise import *
pd.options.display.max_rows = 100


class Menu:
    def __init__(self, attributes: list, function: callable, config):
        self.exercise = Exercise(config)
        self.item_counter = 0
        self.attributes = attributes
        self.__input_: int = 0
        self.function = function
        self.config = config

    def execute(self):
        self.__input_ = int(input('Please, type chosen variant: '))
        if self.__input_ == 1:
            gen = Generator(self.function, self.config)
            gen.generate()
            self.exercise.exercise_1(gen)
            print('')
            gen.save()
        if self.__input_ == 2:
            data = pd.read_excel('data.xlsx')
            data = data.drop(data.columns[[0]], axis=1)
            model = self.__get_model(data)
            model.fit(100, self.config[1])
            self.exercise.exercise_2(model, data)
        if self.__input_ == 3:
            data = pd.read_excel('data.xlsx')
            data = data.drop(data.columns[[0]], axis=1)
            u = data[['U']].to_numpy()

            model = self.__get_model(data)
            model.fit(100, self.config[1] - 1)

            model_modified = self.__get_model(data)
            signs = data[['X1', 'X2']].to_numpy().transpose()
            model_modified.X = model_modified.X.transpose()
            model_modified.X[5] = signs[1] ** 2
            model_modified.X = model_modified.X.transpose()
            model_modified.fit(100, self.config[1])
            exp_interval = self.__get_expected_interval(model.X, model.get_theta(), model.variance)
            y_interval = self.__get_y_calc_interval(model.X, model.get_theta(), model.variance)
            self.exercise.exercise_3(model, model_modified, u, exp_interval, y_interval)

    def get_input(self):
        return self.__input_

    def __get_model(self, data) -> Model:
        signs = data[['X1', 'X2']].to_numpy().transpose()
        X = self.__get_x(signs, self.config[1]).transpose()
        y = data[['Y']].to_numpy()
        model = Model(X, y, self.function)
        return model

    def __get_x(self, signs, m) -> np.ndarray:
        X = np.ones((m, 120))
        X[4] = signs[0] * signs[1]
        X[0] = signs[0]
        signs[0] = signs[0] ** 2
        X[1] = signs[0]
        X[2] = signs[1]
        signs[1] = signs[1] ** 3
        X[3] = signs[1]
        return X

    def __get_expected_interval(self, X: np.array, theta_calc: np.array, variance: float) -> list:
        x1_list = np.linspace(-1, 1, 100)
        x2 = 0
        theta_calc = [num[0] for num in theta_calc]
        lower, center, upper = [], [], []
        t = abs(stats.t.ppf(1 - 0.05, 100 - 6))
        for i in range(100):
            x1 = x1_list[i]
            x = [x1, x2]
            fx = self.function(x, theta_calc)
            fx_vector = np.array(self.__get_vector_function(x, theta_calc))
            tmp = np.dot(np.dot(fx_vector.transpose(), np.linalg.inv(np.dot(X.transpose(), X))), fx_vector)
            sigma = variance * (1 + tmp)
            lower.append(fx - t * sigma)
            center.append(fx)
            upper.append(fx + t * sigma)
        # TODO 1 ПЕРЕНЕСТИ В ДРУГОЙ КЛАСС
        plt.plot(x1_list, lower)
        plt.plot(x1_list, center)
        plt.plot(x1_list, upper)
        plt.show()
        return center

    def __get_y_calc_interval(self, X: np.array, theta_calc: np.array, variance: float) -> list:
        x2_list = np.linspace(-1, 1, 100)
        x1 = 0
        theta_calc = [num[0] for num in theta_calc]
        lower, center, upper = [], [], []
        t = abs(stats.t.ppf(1 - 0.05, 100 - 6))
        for i in range(100):
            x2 = x2_list[i]
            x = [x1, x2]
            fx = self.function(x, theta_calc)
            fx_vector = np.array(self.__get_vector_function(x, theta_calc))
            tmp = sqrt(np.dot(np.dot(fx_vector.transpose(), np.linalg.inv(np.dot(X.transpose(), X))), fx_vector))
            sigma = variance * tmp
            lower.append(fx - t * sigma)
            center.append(fx)
            upper.append(fx + t * sigma)
        # TODO 2 ПЕРЕНЕСТИ В ДРУГОЙ КЛАСС
        plt.plot(x2_list, lower)
        plt.plot(x2_list, center)
        plt.plot(x2_list, upper)
        plt.show()
        return center

    def __get_vector_function(self, X, theta) -> list:
        return [X[0] * theta[0], X[0]**2 * theta[1], X[1] * theta[2],
                X[1]**3 * theta[3], X[0]*X[1] * theta[4], X[1]**2 * theta[5]]

    def __enter__(self):
        if self.item_counter == len(self.attributes) - 1:
            print(f'-1 : {self.attributes[self.item_counter]}')
        else:
            print(f'{self.item_counter + 1} : {self.attributes[self.item_counter]}')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.item_counter += 1

