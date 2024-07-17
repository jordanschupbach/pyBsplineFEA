from functools import partial

import numpy as np
import scipy.sparse.linalg as splinalg
import splipy


class BSplineMSEFitness:
    def __init__(self, x, y):
        self._f = partial(self._full_eval, x=x, y=y)

    def _full_eval(self, knot_seq, x, y):
        n = len(y)
        # TODO: concat knot vec here?
        # knots = np.concatenate((np.array([0, 0]), knot_seq, np.array([1, 1])))
        bsp = splipy.BSplineBasis(3, knot_seq, -1)
        # bsp = splipy.BSplineBasis(3, knots, -1)
        xmat = bsp.evaluate(x, 0, True, True)
        xt = xmat.transpose()
        try:
            theta = (splinalg.inv(xt @ xmat) @ xt) @ y
        except:
            return 99999999999
        yhat = xmat @ theta
        mse = np.sum(np.square(y - yhat)) / n
        return mse

    def __call__(self, knots):
        return self._f(knots)
