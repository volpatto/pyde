import numpy as np


def f_ackley(x, a, b, c):
    """
    Define the benchmark Ackley function.

    :param numpy.ndarray x:
        The function's argument array.

    :param float a:
        Function's constant.

    :param float b:
        Function's constant.

    :param float c:
        Function's constant.

    :return:
        The evaluated function at the given input array.
    :rtype: float
    """
    d = len(x)
    f = - a * np.exp(-b * np.sqrt((1.0 / d) * np.sum(x * x))) \
        - np.exp((1.0 / d) * np.sum(np.cos(c * x))) + a + np.exp(1)
    return f


def f_rosenbrock(x):
    """
    Define the benchmark Rosenbrock function.

    :param numpy.ndarray x:
        The function's argument array.

    :return:
        The evaluated function at the given input array.
    :rtype: float
    """
    dim = len(x)
    f = 0.0
    for i in range(dim-1):
        left_term = 100. * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i])
        right_term = (1. - x[i]) * (1. - x[i])
        f += left_term + right_term
    return f
