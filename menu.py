from exercise import *


pd.options.display.max_rows = 100


class Menu:
    def __init__(self, attributes: list, function: callable, config):
        self.exercise = Exercise()
        self.item_counter = 0
        self.attributes = attributes
        self.__input_: int = 0

    def execute(self, function, config):
        self.__input_ = int(input('Please, type chosen variant: '))
        if self.__input_ == 1:
            gen = Generator(function, config)
            gen.generate()
            self.exercise.exercise_1(gen)
            print('')
            gen.save()
        if self.__input_ == 2:
            data = pd.read_excel('data/data.xlsx')
            data = data.drop(data.columns[[0]], axis=1)
            model = Model(data, config, function)
            model.fit(100, config[1])
            self.exercise.exercise_2(model, data, config)
        if self.__input_ == 3:
            data = pd.read_excel('data/data.xlsx')
            data = data.drop(data.columns[[0]], axis=1)
            u = data[['U']].to_numpy()

            model = Model(data, function, config)
            model.fit(100, config[1] - 1)

            model_modified = Model(data, function, config)
            signs = data[['X1', 'X2']].to_numpy().transpose()
            model_modified.X = model_modified.X.transpose()
            model_modified.X[5] = signs[1] ** 2
            model_modified.X = model_modified.X.transpose()
            model_modified.fit(100, config[1])
            exp_interval = model_modified.get_expected_interval()
            y_interval = model_modified.get_output_interval()
            self.exercise.exercise_3(model, model_modified, config, u, exp_interval, y_interval)

    def get_input(self):
        return self.__input_

    def __enter__(self):
        if self.item_counter == len(self.attributes) - 1:
            print(f'-1 : {self.attributes[self.item_counter]}')
        else:
            print(f'{self.item_counter + 1} : {self.attributes[self.item_counter]}')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.item_counter += 1

