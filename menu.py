from model import Model, get_w_squared, get_variance
from generator import *

pd.options.display.max_rows = 100


class Menu:
    def __init__(self, attributes: list, function: callable, config):
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
            gen.lab_exercise()
            print('')
            gen.save()
        if self.__input_ == 2:
            data = pd.read_excel('data.xlsx')
            data = data.drop(data.columns[[0]], axis=1)
            model = self.__get_model(data)
            model.fit(100, self.config[1])
            model.lab_exercise(self.config, data)
        if self.__input_ == 3:
            data = pd.read_excel('data.xlsx')
            data = data.drop(data.columns[[0]], axis=1)
            u = data[['U']].to_numpy()

            model = self.__get_model(data)
            model.fit(100, self.config[1] - 1)
            print('Дисперсия ошибки незашумленного отлика и дисперсия, полученная на основе '
                  'остаточной суммы квадратов в модели без изменений')
            print(f"Var is {get_variance(get_w_squared(u, self.config)[0][0], self.config[3])} \
                                                                \nVar_calc is {model.variance}")

            model_modified = self.__get_model(data)
            signs = data[['X1', 'X2']].to_numpy().transpose()
            model_modified.X = model_modified.X.transpose()
            model_modified.X[5] = signs[1] ** 2
            model_modified.X = model_modified.X.transpose()
            model_modified.fit(100, self.config[1])
            print('Дисперсия ошибки незашумленного отлика и дисперсия, полученная на основе '
                  'остаточной суммы квадратов в модели с добавлением регрессора x2**2')
            print(f"Var is {get_variance(get_w_squared(u, self.config)[0][0], self.config[3])} \
                                                                            \nVar_calc is {model_modified.variance}")

    def get_input(self):
        return self.__input_

    def __get_model(self, data) -> Model:
        signs = data[['X1', 'X2']].to_numpy().transpose()
        X = self.__get_x(signs, self.config[1]).transpose()
        y = data[['Y']].to_numpy()
        model = Model(X, y, self.function)
        return model

    def __get_x(self, signs, m) -> np.ndarray:
        X = np.ones((m, 100))
        X[4] = signs[0] * signs[1]
        X[0] = signs[0]
        signs[0] = signs[0] ** 2
        X[1] = signs[0]
        X[2] = signs[1]
        signs[1] = signs[1] ** 3
        X[3] = signs[1]
        return X

    def __enter__(self):
        if self.item_counter == len(self.attributes) - 1:
            print(f'-1 : {self.attributes[self.item_counter]}')
        else:
            print(f'{self.item_counter + 1} : {self.attributes[self.item_counter]}')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.item_counter += 1

