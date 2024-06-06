import numpy as np


def coefficient_of_variation(x):
    # Where I got it from https://www.nature.com/articles/s41598-020-79136-x.pdf
    # "Implementtation" / defintion
    # https://en.wikipedia.org/wiki/Coefficient_of_variation
    return x.std() / x.mean()

def normalized_rmse(x, y):
    x_mean = np.mean(x)
    rmse = np.sqrt(np.mean((x - y) ** 2))
    nrmse = rmse / x_mean * 100
    return rmse, nrmse

