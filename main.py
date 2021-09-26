from generator import Generator


def function(x, theta):
    return theta[0] + theta[1] * x[0] + theta[2] * x[0]**2 + \
           theta[3] * x[1] + theta[4] * x[1]**2 + theta[5] * x[1]**3 + x[0] * x[1]


if __name__ == '__main__':
    config = [2, 6, [0, 0.5, 0.4, 0.3, 0.2, 0.1], 0.1]
    with Generator(function, config, 'data.xlsx'):
        print('Generation done')
