from generator import Generator


def function(x, theta):
    return theta[0] + theta[1] * x[1] + theta[2] * x[1]**2 + \
           theta[3] * x[2] + theta[4] * x[2]**2 + theta[5] * x[2]**3 + x[1] * x[2]


if __name__ == '__main__':
    config = [6, [0, 0.5, 0.4, 0.3, 0.2, 0.1], 0.1]
    with Generator(function, config):
        print('Generation done')
