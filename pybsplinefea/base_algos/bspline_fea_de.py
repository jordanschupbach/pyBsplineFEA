import numpy as np
from pyfea.base_algos import parallel_eval
from pyfea.fea.base_algos import FeaDE

from pybsplinefea.base_algos import BSplineDE


class BSplineFeaDE(BSplineDE, FeaDE):

    #def __init__(
    #    self,
    #    function,
    #    domain,
    #    generations=100,
    #    pop_size=20,
    #    mutation_factor=0.5,
    #    crossover_rate=0.9,
    #    processes=4,
    #    chunksize=4,
    #    fitness_terminate=False,
    #):
    #    """
    #    @param function: the objective function to be minimized.
    #    @param domain: the domain on which we explore the function stored as a (dim,2) matrix,
    #    where dim is the number of dimensions we evaluate the function over.
    #    @param generations: the number of generations run before the algorithm terminates.
    #    @param pop_size: the number of individuals in the population.
    #    @param mutation_factor: the scalar factor used in the mutation step.
    #    @param crossover_rate: the probability of taking a mutated value during the crossover step.
    #    """
    #    self.fitness_terminate = fitness_terminate
    #    self.generations = generations
    #    self.pop_size = pop_size
    #    self.func = function
    #    self.domain = domain
    #    self.processes = processes
    #    self.chunksize = chunksize
    #    self.pop = self._init_pop()
    #    self.nfitness_evals = self.pop_size
    #    self.mutation_factor = mutation_factor
    #    self.crossover_rate = crossover_rate
    #    self.mutant_pop = np.zeros((self.pop_size, self.domain.shape[0]))
    #    self.ngenerations = 0
    #    self.average_pop_variance = []
    #    self.average_pop_eval = []
    #    self.fitness_list = []
    #    self.best_answers = []
    #    self.pop_eval = parallel_eval(
    #        self.func, self.pop, processes=self.processes, chunksize=self.chunksize
    #    )
    #    self.best_eval = np.min(self.pop_eval)
    #    self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval), :])

    def _init_pop(self):
        """
        Initialize random particles.
        """
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        pop = lbound + area * np.random.random(size=(self.pop_size, area.shape[0]))
        pop.sort()
        return pop

    def _stay_in_domain(self):
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

    def _pop_domain_check(self):
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

    def update_bests(self, parallel=False, processes=4, chunksize=4):
        """
        Update the evaluation of the objective function after a context vector update.
        """
        self._pop_domain_check()
        self.pop.sort()
        self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
        if parallel:
            self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
        else:
            self.pop_eval = parallel_eval(self.func, self.pop, processes, chunksize)

        self.nfitness_evals += self.pop_size
        self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval), :])
        self.best_eval = np.min(self.pop_eval)

    #def run(self, progress=True, parallel=False, processes=4, chunksize=4):
    #    """
    #    Run the minimization algorithm.
    #    """
    #    self._initialize(parallel, processes, chunksize)
    #    if self.fitness_terminate:
    #        while self.nfitness_evals < self.generations:
    #            self.ngenerations += 1
    #            self._mutate()
    #            self._stay_in_domain()
    #            self._crossover()
    #            self._selection()
    #            self._track_vals()
    #    else:
    #        super().run(progress, parallel, processes, chunksize)
    #    return self.best_eval
