import numpy as np
from pyfea.base_algos import parallel_eval
from pyfea.fea.base_algos import FeaPSO
from pybsplinefea.base_algos.bspline_pso import BSplinePSO


class BSplineFeaPSO(BSplinePSO, FeaPSO):
    def _initialize(self, parallel=False, processes=4, chunksize=4):
        self.pop.sort()
        if parallel:
            self.pop_eval = parallel_eval(self.func, self.pop, processes, chunksize)
        else:
            self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
        self.pbest_eval = np.copy(self.pop_eval)
        self.gbest_eval = np.min(self.pbest_eval)
        self.gbest = np.copy(self.pbest[np.argmin(self.pbest_eval), :])
        self.velocities = self._init_velocities()

    def init_pop(self):
        """
        Initialize random particles.
        """
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        pop = lbound + area * np.random.random(size=(self.pop_size, area.shape[0]))
        pop.sort()
        return pop

    def stay_in_domain(self):
        """
        Ensure that the particles don't move outside the domain of our function by projecting
        them to the domain's boundary at that point.
        """
        area = self.domain[:, 1] - self.domain[:, 0]
        self.pop = np.where(
            self.domain[:, 0] > self.pop,
            self.domain[:, 0] + 0.1 * area * np.random.random(),
            self.pop,
        )
        self.pop = np.where(
            self.domain[:, 1] < self.pop,
            self.domain[:, 1] - 0.1 * area * np.random.random(),
            self.pop,
        )

    def update_bests(self):
        """
        Update the current personal and global best values based on the new positions of the particles.
        """
        self.stay_in_domain()
        self.order_knots()
        for pidx in range(self.pop_size):
            curr_eval = self.func(self.pop[pidx, :])
            self.nfitness_evals += 1
            self.pop_eval[pidx] = curr_eval
            if curr_eval < self.pbest_eval[pidx]:
                self.pbest[pidx, :] = np.copy(self.pop[pidx, :])
                self.pbest_eval[pidx] = curr_eval
                if curr_eval < self.gbest_eval:
                    self.gbest = np.copy(self.pop[pidx, :])
                    self.gbest_eval = curr_eval

    def order_knots(self):
        sort_idxs = self.pop.argsort()
        self.pop = np.array([p[s] for p, s in zip(self.pop, sort_idxs)])
        self.velocities = np.array([v[s] for v, s in zip(self.velocities, sort_idxs)])

    def base_reset(self, parallel=False, processes=4, chunksize=4):
        """
        Reset the algorithm in preparation for another run.
        """
        self.reinitialize_population()
        self.order_knots()
        self.velocities = super()._init_velocities()
        self.reset_fitness(parallel, processes, chunksize)
