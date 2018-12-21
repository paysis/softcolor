import numpy as np


def euclidean_distance(x, y):
    return np.linalg.norm(x-y, axis=3)


def l_1_distance(x, y):
    return np.linalg.norm(x-y, ord=1, axis=3)


def l_inf_distance(x, y):
    return np.linalg.norm(x-y, ord=np.inf, axis=3)
