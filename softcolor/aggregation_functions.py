import numpy as np


def conjunction_min(x, y):
    return np.minimum(x, y)


def conjunction_product(x, y):
    return x * y


def conjunction_lukasiewicz(x, y):
    return np.maximum(0, x + y - 1)


def conjunction_drastic(x, y):
    result = x * 0
    result[x == 1] = y[x == 1]
    result[y == 1] = x[y == 1]
    return result


def r_implication(conjunction):
    if conjunction == conjunction_min:
        return implication_godel
    if conjunction == conjunction_product:
        return implication_goguen
    if conjunction == conjunction_lukasiewicz:
        return implication_lukasiewicz
    if conjunction == conjunction_drastic:
        return implication_weber
    raise AttributeError('Unknown conjunction: {}'.format(str(conjunction)))


def implication_goguen(x, y):
    result = y / x
    result[x <= y] = 1
    return result


def implication_godel(x, y):
    result = y
    result[x <= y] = 1
    return result


def implication_lukasiewicz(x, y):
    return np.minimum(1, 1 - x + y)


def implication_weber(x, y):
    result = y
    result[x < 1] = 1
