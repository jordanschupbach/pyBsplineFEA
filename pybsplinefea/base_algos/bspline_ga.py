import random

import numpy as np
from pyfea.base_algos import GA, parallel_eval


class BSplineGA(GA):
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

    def bounds_check(self, children):
        """
        Ensure that the particles don't move outside the domain of our function by projecting
        them to the domain's boundary at that point.
        """
        area = self.domain[:, 1] - self.domain[:, 0]
        children = np.where(
            self.domain[:, 0] > children,
            self.domain[:, 0] + 0.1 * area * np.random.random(),
            children,
        )
        children = np.where(
            self.domain[:, 1] < children,
            self.domain[:, 1] - 0.1 * area * np.random.random(),
            children,
        )

    def _mutation(self, children, parallel=False, processes=4, chunksize=4):
        """
        Mutates children through swapping and recombines that with the parent population.
        """
        for child in children:
            for i in range(len(child)):
                if random.random() < self.mutation_rate:
                    rand_value = random.uniform(-1 * self.mutation_range, self.mutation_range)
                    child[i] += rand_value
        self._bounds_check(children)
        children.sort()
        if not parallel:
            child_evals = []
            for child in children:
                child_evals.append(self.func(child))
                self.nfitness_evals += 1
        else:
            child_evals = parallel_eval(self.func, children, processes, chunksize)
            self.nfitness_evals += children.shape[0]
        for i, child in enumerate(children):
            index = np.argmax(self.pop_eval)
            if self.pop_eval[index] > child_evals[i]:
                self.pop_eval[index] = child_evals[i]
                self.pop[index,:] = child
