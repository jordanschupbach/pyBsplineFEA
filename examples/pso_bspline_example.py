import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as splinalg
import splipy

from pybsplinefea import BSplineMSEFitness, BSplinePSO
from pybsplinefea.benchmarks import mdoppler

# Parameters
n = 10000
k = 100

# Create testing data
x = np.random.rand(n)
ytrue = mdoppler(x)
y = ytrue + np.random.normal(0.0, 0.2, n)

# plt.scatter(x, y)
# plt.show()

dom = np.zeros((k, 2))
dom[:, 1] = 1.0

pso = BSplinePSO(BSplineMSEFitness(x, y), dom, 100, 20)
pso.run(progress=True)

pso.diagnostic_plots()
plt.show()

soln = pso.get_soln()
print(soln)
pso.get_soln_fitness()

print(soln)

clamp_knots = np.concatenate((np.array([0, 0]), soln, np.array([1, 1])))
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
