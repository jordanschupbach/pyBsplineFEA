import numpy as np


def mdoppler(x: np.ndarray) -> np.ndarray:
    return np.sin(20 / (x + 0.15))
