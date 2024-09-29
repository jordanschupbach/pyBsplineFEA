import math

import numpy as np


def range_transform(x, a, b, minx, maxx):
    return (b - a) / (maxx - minx) * x - minx + a

def big_spike(x):
    return 100 * np.exp(-abs(10 * x - 5)) + (10 * x - 5) ** 5 / 500

def mbig_spike(x):
    return range_transform(big_spike(x), -1, 1, 0, 100)

def mdoppler(x: np.ndarray, add_denom=0.3) -> np.ndarray:
    return np.sin(20 / (x + add_denom))

def cliff(x):
    return 90 / (1 + np.exp(-100 * (x - 0.4)))

def discontinuity(x):
    return np.where(x < 0.6, 1 / (0.01 + (x - 0.3) ** 2), 1 / (0.015 + (x - 0.65) ** 2))

def smooth_peak(x):
    return np.sin(x) + (2 * np.e) ** (-30 * x**2)


def second_smooth_peak(x):
    return np.sin(2 * x) + (2 * np.e) ** (-16 * x**2) + 2


def blocks(x):
    donoho = [0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81]
    constants = [14, -4, 7.1, -6.8, 10.1, -4.6, 2.4, 17.3, 7.3, 14.8]
    ret = np.zeros(x.shape)
    for i, constant in enumerate(constants):
        ret = np.where((x >= donoho[i]) & (x < donoho[i + 1]), constant, ret)
    return ret

def mblocks(x):
    return range_transform(blocks(x), -1, 1, -6.8, 17.3)


def bumps(x):
    donoho = [0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81]
    heights = [36, 44.5, 28.2, 39.7, 50, 40.5, 20.9, 41.3, 26.8, 45.3, 40.7]
    widths = [
        0.00175,
        0.0032,
        0.00263,
        0.00475,
        0.00432,
        0.01655,
        0.00314,
        0.00702,
        0.00228,
        0.00496,
        0.00261,
    ]
    ret = np.zeros(x.shape)
    for i in range(len(donoho)):
        ret += heights[i] / (1 + ((x - donoho[i]) / widths[i]) ** 4)
    return ret

def mbumps(x):
    return range_transform(bumps(x), -1, 1, 0.0, 50.13894136)

def heavi_sine(x):
    ret = 10 * np.sin(x * (4 * np.pi))
    ret = np.where((x >= 0.3) & (x < 0.72), ret - 4.5, ret)
    ret = np.where(x >= 0.72, ret - 0.8, ret)
    return ret

def mheavi_sine(x):
    return range_transform(heavi_sine(x), -1., 1., -14.5, 10.)
