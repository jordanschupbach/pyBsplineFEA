class FasterBsplineEval:
    def __init__(self, x, y):
        self.x = dm.Vector(x)
        self.y = dm.Vector(y)

    def __call__(self, knots):
        tx = dm.Vector(self.x)
        ty = dm.Vector(self.y)
        knotvec = dm.Vector(knots)
        start_time = time.time()
        xmat = dm.gsl_bspline_eval(tx, knotvec, 3, False)
        end_time = time.time()
        print("fill mat took", end_time - start_time)

        start_time = time.time()
        slm = dm.GSLSLM(xmat, ty)
        end_time = time.time()
        print("fitting took", end_time - start_time)
        return slm.get_mse()
