import numpy as np
from pyfea.base_algos import DE, parallel_eval


class BSplineDE(DE):
    def _init_pop(self):
        """
        Initialize random particles.
        """
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        pop = lbound + area * np.random.random(size=(self.pop_size, area.shape[0]))
        pop.sort()
        return pop

    def _eval_pop(self, parallel=False, processes=4, chunksize=4):
        self.pop.sort()
        if not parallel:
            for pidx in range(self.pop_size):
                curr_eval = self.func(self.pop[pidx, :])
                self.nfitness_evals += 1
                self.pop_eval[pidx] = curr_eval
        else:
            self.pop_eval = parallel_eval(
                self.func, self.pop, processes=processes, chunksize=chunksize
            )
            self.nfitness_evals += self.pop_size

    def stay_in_domain(self):
        """
        Ensure that the particles don't move outside the domain of our function by projecting
        them to the domain's boundary at that point.
        """
        area = self.domain[:, 1] - self.domain[:, 0]
        self.mutant_pop = np.where(
            self.domain[:, 0] > self.mutant_pop,
            self.domain[:, 0] + 0.1 * area * np.random.random(),
            self.mutant_pop,
        )
        self.mutant_pop = np.where(
            self.domain[:, 1] < self.mutant_pop,
            self.domain[:, 1] - 0.1 * area * np.random.random(),
            self.mutant_pop,
        )

    def _selection(self, parallel=False, processes=4, chunksize=4):
        """
        The fitness evaluation and selection. Greedily selects whether to keep or throw out a value.
        Consider implementing and testing more sophisticated selection algorithms.
        """
        self.pop.sort()
        self.mutant_pop.sort()
        if parallel:
            mutant_pop_eval = [self.func(self.mutant_pop[i, :]) for i in range(self.pop_size)]
        else:
            mutant_pop_eval = parallel_eval(self.func, self.mutant_pop, processes, chunksize)
        self.nfitness_evals += self.pop_size
        for i in range(self.pop_size):
            fella_eval = mutant_pop_eval[i]
            if fella_eval < self.pop_eval[i]:
                self.pop_eval[i] = fella_eval
                self.pop[i, :] = np.copy(self.mutant_pop[i, :])
        self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval), :])
        self.best_eval = np.min(self.pop_eval)
