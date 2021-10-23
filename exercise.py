import numpy as np

from model import Model, get_variance, get_w_squared
from generator import *


class Exercise:
    def __init__(self, config: list):
        self.config = config

    def exercise_1(self, generator: Generator):
        assert len(generator.x) > 0, 'Сначала запустите метод generate'
        generator.draw_separately()
        generator.draw()
        print('')
        print('W**2 = ' + str(generator.get_w_squared()))
        print('')
        print('Variance(e) = ' + str(generator.get_variance(self.config[3])))
        print('')

    def exercise_2(self, model: Model, data):
        assert len(model.get_theta()) > 0, 'Сначала запустите метод generate'
        plt.figure(figsize=(15, 10))
        plt.scatter(model.y, model.y - model.y_calc, c='blue', marker='o', label='Training data')
        plt.title('Диаграмма рассеивания остатков отклика', fontsize=22)
        plt.xlabel('Predicted values', fontsize=15)
        plt.ylabel('Residuals', fontsize=15)
        plt.show()

        print('СГЕНЕРИРОВАННЫЕ ДАННЫЕ')
        print(data[['X1', 'X2', 'Y']])
        print('')

        E = pd.Series([num[0] for num in model.e])
        Y_calc = pd.Series([num[0] for num in model.y_calc])
        data['E'] = E
        data['Y_calc'] = Y_calc
        print('Истинные значения функции, найденные значения отклика и вектор ошибки(**U,Y_CALC, Y - Y_CALC**)')
        print(data[['U', 'Y_calc', 'Y', 'E']])
        print('')
        data.to_excel('report.xlsx')

        theta = pd.DataFrame()
        theta['True'] = pd.Series(self.config[2])
        theta['Calc'] = pd.Series([num[0] for num in model.get_theta()])
        print('Вектор истинных коэффициентов, вектор найденных коеффициентов')
        print(theta)
        print('')
        theta.to_excel('theta.xlsx')

        u = data[['U']].to_numpy()
        u = np.array([num[0] for num in u])
        print('Дисперсия ошибки незашумленного отлика и дисперсия, полученная на основе остаточной суммы квадратов')
        print(f"Var is {get_variance(get_w_squared(u, self.config), self.config[3])} "
                                                f"\nVar_calc is {model.variance[0][0]}")
        print('')

        info = model.get_info(len(model.y), len(self.config[2]), get_variance(get_w_squared(u, self.config), self.config[3]))
        print('Значения F при проверке гипотезы об адекватности модели')
        if info[2]:
            print(f"Гипотеза не отвергается\nF is {info[0][0][0]}\nF_T is {info[1]}")
        else:
            print("Полученная модель неадекватна")

    def exercise_3(self, model: Model, model_mod: Model, u: np.array):
        print('Дисперсия ошибки незашумленного отлика и дисперсия, полученная на основе '
              'остаточной суммы квадратов в модели без изменений')
        print(f"Var is {get_variance(get_w_squared(u, self.config)[0][0], self.config[3])} \
                                                                        \nVar_calc is {model.variance}")
        print('Дисперсия ошибки незашумленного отлика и дисперсия, полученная на основе '
              'остаточной суммы квадратов в модели с добавлением регрессора x2**2')
        print(f"Var is {get_variance(get_w_squared(u, self.config)[0][0], self.config[3])} \
                                                                                  \nVar_calc is {model_mod.variance}")
