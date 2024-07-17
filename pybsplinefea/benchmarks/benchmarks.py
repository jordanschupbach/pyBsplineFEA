import numpy as np
import math

def big_spike(x):
    return 100 * np.exp(-abs(10 * x-5)) + (10 * x - 5)**5/500

def mdoppler(x: np.ndarray) -> np.ndarray:
    return np.sin(20 / (x + 0.15))

def cliff(x):
    return 90/(1+np.exp(-100*(x-0.4)))

def discontinuity(x):
    return np.where(x < 0.6, 1/(0.01+(x-0.3)**2), 1/(0.015+(x-0.65)**2))

def smooth_peak(x):
    return np.sin(x) + (2 * np.e) ** (-30 * x**2)

def second_smooth_peak(x):
    return np.sin(2*x) + (2 * np.e) ** (-16 * x**2) + 2
