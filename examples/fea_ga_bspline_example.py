import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as splinalg
import splipy
from pyfea import linear_factorizer

from pybsplinefea import BSplineFeaGA, BSplineMSEFitness
from pybsplinefea.benchmarks import mdoppler
from pybsplinefea.bsplinefea import BSplineFEA

n = 10000
x = np.random.random(n)
ytrue = mdoppler(x)
func_width = np.max(ytrue) - np.min(ytrue)
sigma = func_width / 20
epsilon = np.random.normal(0, sigma, n)
y = ytrue + epsilon

k = 100
kseq = np.linspace(0.0, 1.0, k)

plt.scatter(x, y)
plt.show()

dom = (0.0, 1.0)

fea = BSplineFEA(
    factors=linear_factorizer(20, 10, k),
    function=BSplineMSEFitness(x, y),
    iterations=10,
    dim=k,
    domain=dom,
    base_algo=BSplineFeaGA,
    diagnostics_amount=1,
    generations=10,
)

fea.run()
# fea.get_soln()
# soln = fea.get_soln_fitness()

soln = fea.context_variable
fea.diagnostic_plots()
plt.show()

# Plot fitted model at bspline soln
bsp = splipy.BSplineBasis(3, soln, -1)
xmat = bsp.evaluate(x, 0, True, True)
xt = xmat.transpose()
theta = (splinalg.inv(xt @ xmat) @ xt) @ y
nx = 1000
xseq = np.linspace(0.0, 1.0, nx)
xseqmat = bsp.evaluate(xseq, 0, True, True)
yest = xseqmat @ theta

fig, ax = plt.subplots(figsize=(8, 4))
plt.plot(xseq, yest)
ax.set_ylim(-2, 2)
ax.plot(soln, [-1.95] * len(soln), "|", color="k")
plt.show()
