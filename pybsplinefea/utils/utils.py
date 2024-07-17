import numpy as np

def make_noisy(x, sigma):
    return x + np.random.normal(scale=sigma, size=len(x))

def clamp_knots(knots, order):
    new_knots = np.concatenate((np.zeros(order), knots, np.ones(order)))
    return new_knots
