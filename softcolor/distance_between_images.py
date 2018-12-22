import numpy as np


def euclidean_distance(x, y):
    return np.linalg.norm(x-y, axis=2)


def l_one_distance(x, y):
    return np.linalg.norm(x-y, ord=1, axis=2)


def l_inf_distance(x, y):
    return np.linalg.norm(x-y, ord=np.inf, axis=2)
