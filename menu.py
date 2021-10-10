from model import Model
from generator import *

pd.options.display.max_rows = 100


class Menu:
    def __init__(self, attributes: list):
        self.item_counter = 0
        self.attributes = attributes
        self.__input_: int = 0

    def execute(self, function, config):
        self.__input_ = int(input('Please, type chosen variant: '))
        if self.__input_ == 1:
            gen = Generator(function, config)
            gen.generate()
            gen.lab_exercise()
            print('')
            gen.save()
        if self.__input_ == 2:
            data = pd.read_excel('data.xlsx')
            data = data.drop(data.columns[[0]], axis=1)

            X = np.ones((config[1], config[1] * 20))
            signs = data[['X1', 'X2']].to_numpy().transpose()
            X[4] = signs[0] * signs[1]
            X[0] = signs[0]
            signs[0] = signs[0] ** 2
            X[1] = signs[0]
            X[2] = signs[1]
            signs[1] = signs[1] ** 3
            X[3] = signs[1]
            X = X.transpose()
            y = data[['Y']].to_numpy()

            model = Model(X, y, function)
            model.fit(len(y), len(config[2]))
            model.lab_exercise(config, data)

    def __enter__(self):
        print(f'{self.item_counter + 1} : {self.attributes[self.item_counter]}')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.item_counter += 1

