from model import *
from generator import *


class Exercise:
    def __init__(self):
        pass

    def exercise_1(self, generator: Generator):
        assert len(generator.x) > 0, 'Сначала запустите метод generate'
        generator.draw_separately()
        generator.draw()
        print('')
        print('W**2 = ' + str(generator.get_w_squared()))
        print('')
        print('Variance(e) = ' + str(generator.get_variance(generator.config[3])))
        print('')

    def exercise_2(self, model: Model, data, config):
        assert len(model.theta) > 0, 'Сначала запустите метод fit'
        model.draw_residuals()

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
        data.to_excel('data/report.xlsx')

        theta = pd.DataFrame()
        theta['True'] = pd.Series(config[2])
        theta['Calc'] = pd.Series([num[0] for num in model.theta])
        print('Вектор истинных коэффициентов, вектор найденных коеффициентов')
        print(theta)
        print('')
        theta.to_excel('data/theta.xlsx')

        u = data[['U']].to_numpy()
        u = np.array([num[0] for num in u])
        print('Дисперсия ошибки незашумленного отлика и дисперсия, полученная на основе остаточной суммы квадратов')

        print(f"Var is {get_variance(u, config[3])}\nVar_calc is {model.variance}")
        print('')

        info = model.get_info(len(model.y), len(config[2]), get_variance(u, config[3]))
        print('Значения F при проверке гипотезы об адекватности модели')
        if info[2]:
            print(f"Гипотеза не отвергается\nF is {info[0][0][0]}\nF_T is {info[1]}")
        else:
            print("Полученная модель неадекватна")

    def exercise_3(self, model: Model, model_mod: Model, config, u: np.array, exp_interval, y_interval):
        print('Дисперсия ошибки незашумленного отлика и дисперсия, полученная на основе '
              'остаточной суммы квадратов в модели без изменений')

        print(f"Var is {get_variance(u, config[3])[0][0]}\nVar_calc is {model.variance}\n\n")
        print('Дисперсия ошибки незашумленного отлика и дисперсия, полученная на основе '
              'остаточной суммы квадратов в модели с добавлением регрессора x2**2')

        print(f"Var is {get_variance(u, config[3])[0][0]}\nVar_calc is {model_mod.variance}\n\n")
        info = model_mod.get_info(len(model.y), config[1],
                                  get_variance(u, config[3])[0][0])
        intervals_array = info[3]
        intervals = pd.DataFrame()
        intervals['Lower'] = pd.Series(intervals_array[0])
        intervals['Calc'] = pd.Series([num[0] for num in model_mod.theta])
        intervals['True'] = pd.Series(config[2])
        intervals['Upper'] = pd.Series(intervals_array[1])
        intervals.to_excel('data/intervals.xlsx')

        theta_significance_list = info[4]
        theta_significance = pd.DataFrame()
        theta_significance['Theta'] = pd.Series([num for num in range(config[1])])
        theta_significance['Significance'] = pd.Series(theta_significance_list)
        theta_significance.set_index('Theta')
        theta_significance.to_excel('data/theta_significance.xlsx')

        model_significance = info[5]
        print(f"F = {model_significance[3]}\nF_t = {model_significance[4]}\nTSS = {model_significance[1]}\n"
              f"RSS = {model_significance[2]}\n")
        if model_significance[0]:
            print('Гипотеза о незначимости регрессии принимается\n')
        else:
            print('Гипотеза о незначимости регрессии отвергается\n')

        interval_exp = pd.DataFrame()
        interval_exp['Y'] = pd.Series(y_interval[1])
        interval_exp['Expected'] = pd.Series(exp_interval[1])
        interval_exp.to_excel('interval_exp.xlsx')
        model_mod.draw_intervals(exp_interval[1:], exp_interval[0], xlabel='interval',
                                 ylabel='result', title='График прогнозных значений для математического ожидания')
        model_mod.draw_intervals(y_interval[1:], y_interval[0], xlabel='interval',
                                 ylabel='result', title='График прогнозных значений для функции отклика')
