from menu import Menu


def function(x, theta):
    return theta[0] * x[0] + theta[1] * x[0] ** 2 + \
           theta[2] * x[1] + theta[3] * x[1] ** 3 + theta[4] * x[0] * x[1]


config = [2, 5, [0.1, 100, 0.1, -100, -5], 0.1]
attributes = [
    'ГЕНЕРАЦИЯ ЭКСПЕРИМЕНТАЛЬНЫХ ДАННЫХ ПО СХЕМЕ ИМИТАЦИОННОГО МОДЕЛИРОВАНИЯ',
    'ОЦЕНИВАНИЕ ПАРАМЕТРОВ РЕГРЕССИОННОЙ МОДЕЛИ ПО МЕТОДУ НАИМЕНЬШИХ КВАДРАТОВ',
    'ВЫХОД'
]


def show_titles(menu: object):
    for i in range(len(attributes)):
        with menu:
            print('')


if __name__ == '__main__':
    menu = Menu(attributes)
    while menu.get_input() != -1:
        menu.item_counter = 0
        show_titles(menu)
        menu.execute(function, config)
