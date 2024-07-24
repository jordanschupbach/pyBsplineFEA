import numpy as np
import math

def big_spike(x):
    return 100 * np.exp(-abs(10 * x-5)) + (10 * x - 5)**5/500

def mdoppler(x: np.ndarray, add_denom=0.3) -> np.ndarray:
    return np.sin(20 / (x + add_denom))

def cliff(x):
    return 90/(1+np.exp(-100*(x-0.4)))

def discontinuity(x):
    return np.where(x < 0.6, 1/(0.01+(x-0.3)**2), 1/(0.015+(x-0.65)**2))

def smooth_peak(x):
    return np.sin(x) + (2 * np.e) ** (-30 * x**2)

def second_smooth_peak(x):
    return np.sin(2*x) + (2 * np.e) ** (-16 * x**2) + 2

def blocks(x):
    donoho = [.1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81]
    constants = [14, -4, 7.1, -6.8, 10.1, -4.6, 2.4, 17.3, 7.3, 14.8]
    ret = np.zeros(x.shape)
    for i, constant in enumerate(constants):
        ret = np.where((x>=donoho[i]) & (x<donoho[i+1]), constant, ret)
    return ret

def bumps(x):
    donoho = [.1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81]
    heights = [36, 44.5, 28.2, 39.7, 50, 40.5, 20.9, 41.3, 26.8, 45.3, 40.7]
    widths = [0.00175, 0.0032, 0.00263, 0.00475, 0.00432, 0.01655, 0.00314, 0.00702, 0.00228, 0.00496, 0.00261]
    ret = np.zeros(x.shape)
    for i in range(len(donoho)):
        ret += heights[i]/(1+((x-donoho[i])/widths[i])**4)
    return ret

def heavi_sine(x):
    ret = 10 * np.sin(x*(4*np.pi))
    ret = np.where((x>=.3) & (x<.72), ret-4.5, ret)
    ret = np.where(x>=.72, ret-0.8, ret)
    return ret
