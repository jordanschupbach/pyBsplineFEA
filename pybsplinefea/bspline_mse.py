from functools import partial
from pybsplinefea.utils import clamp_knots

import math
import scipy.sparse.linalg as splinalg
import numpy as np
import scipy.sparse.linalg as splinalg
import splipy


class BSplineMSEFitness:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, knots):
        all_knots = clamp_knots(knots, 3)
        bsp = splipy.BSplineBasis(3, knots, -1)
        xmat = bsp.evaluate(self.x, 0, True, True)
        xt = xmat.transpose()
        LHS = xt @ xmat
        RHS = xt @ self.y
        theta, info  = splinalg.bicgstab(LHS, RHS)
        #print("theta: ", theta)
        yest = xmat @ theta
        mse = np.sum((self.y - yest)**2)/len(self.y)
        return mse
