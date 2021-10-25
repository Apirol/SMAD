import pandas as pd

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

    def exercise_3(self, model: Model, model_mod: Model, u: np.array, exp_interval, y_interval):
        print('Дисперсия ошибки незашумленного отлика и дисперсия, полученная на основе '
              'остаточной суммы квадратов в модели без изменений')
        print(f"Var is {get_variance(get_w_squared(u, self.config)[0][0], self.config[3])} \
                                                                        \nVar_calc is {model.variance}\n\n")
        print('Дисперсия ошибки незашумленного отлика и дисперсия, полученная на основе '
              'остаточной суммы квадратов в модели с добавлением регрессора x2**2')
        print(f"Var is {get_variance(get_w_squared(u, self.config)[0][0], self.config[3])} \
                                                                                  \nVar_calc is {model_mod.variance}\n\n")
        info = model_mod.get_info(len(model.y), self.config[1],
                                  get_variance(get_w_squared(u, self.config)[0][0], self.config[3]))
        intervals_array = info[3]
        intervals = pd.DataFrame()
        intervals['Lower'] = pd.Series(intervals_array[0])
        intervals['Calc'] = pd.Series([num[0] for num in model.get_theta()])
        intervals['True'] = pd.Series(self.config[2])
        intervals['Upper'] = pd.Series(intervals_array[1])
        intervals.to_excel('intervals.xlsx')

        theta_significance_list = info[4]
        theta_significance = pd.DataFrame()
        theta_significance['Theta'] = pd.Series([num for num in range(self.config[1])])
        theta_significance['Significance'] = pd.Series(theta_significance_list)
        theta_significance.set_index('Theta')
        theta_significance.to_excel('theta_significance.xlsx')

        model_significance = info[5]
        print(f"F = {model_significance[3]}\nF_t = {model_significance[4]}\nTSS = {model_significance[1]}\n"
              f"RSS = {model_significance[2]}\n")
        if model_significance[0]:
            print('Гипотеза о незначимости регрессии принимается\n')
        else:
            print('Гипотеза о незначимости регрессии отвергается\n')

        interval_exp = pd.DataFrame()
        interval_exp['Y'] = pd.Series(y_interval)
        interval_exp['Expected'] = pd.Series(exp_interval)
        interval_exp.to_excel('interval_exp.xlsx')
