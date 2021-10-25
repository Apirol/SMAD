from menu import Menu


def function(x, theta):
    return theta[0] * x[0] + theta[1] * x[0] ** 2 + \
           theta[2] * x[1] + theta[3] * x[1] ** 3 + theta[4] * x[0] * x[1] + theta[5] * x[1]**2


config = [2, 6, [0.1, 10, 0.05, -7, 6, 0.1], 0.6]
attributes = [
    'ГЕНЕРАЦИЯ ЭКСПЕРИМЕНТАЛЬНЫХ ДАННЫХ ПО СХЕМЕ ИМИТАЦИОННОГО МОДЕЛИРОВАНИЯ',
    'ОЦЕНИВАНИЕ ПАРАМЕТРОВ РЕГРЕССИОННОЙ МОДЕЛИ ПО МЕТОДУ НАИМЕНЬШИХ КВАДРАТОВ',
    'ИНТЕРВАЛЬНОЕ ОЦЕНИВАНИЕ, ПРОВЕРКА ГИПОТЕЗ И ПРОГНОЗИРОВАНИЕ',
    'ВЫХОД',
]


def show_titles(content_manager: object):
    for i in range(len(attributes)):
        with content_manager:
            print('')


if __name__ == '__main__':
    menu = Menu(attributes, function, config)
    while menu.get_input() != -1:
        menu.item_counter = 0
        show_titles(menu)
        menu.execute()
