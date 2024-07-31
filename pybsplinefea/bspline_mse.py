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

class CrossMSEFitness:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, knots):
        perm = np.random.permutation(len(self.y))
        all_knots = clamp_knots(knots, 3)
        bsp = splipy.BSplineBasis(3, knots, -1)
        MSE_list = []
        for i in range(5):
            low_idx = int(len(self.y) * i / 5)
            high_idx = int(len(self.y) * (i+1) / 5)
            train_x = np.delete(self.x, perm[low_idx:high_idx])
            train_y = np.delete(self.y, perm[low_idx:high_idx])
            train_x = bsp.evaluate(train_x, 0, True, True)
            test_x = self.x[perm[low_idx:high_idx]]
            test_y = self.y[perm[low_idx:high_idx]]
            test_x = bsp.evaluate(test_x, 0, True, True)
            xt = train_x.transpose()
            LHS = xt @ train_x
            RHS = xt @ train_y
            theta, info  = splinalg.bicgstab(LHS, RHS)
            y_hat = test_x @ theta
            mse = np.sum((test_y - y_hat)**2)/len(test_y)
            MSE_list.append(mse)
        return np.average(MSE_list)
