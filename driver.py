import numpy as np

from de.optimization import optimize
from de.benchmarks.functions import f_ackley, f_rosenbrock


# Defining an objective function
def fobj(x):
    return f_ackley(x, 20.0, 0.2, 2.0*np.pi)


# Problem dimension
dim = 10

sol, fobj_sol, conv_flag, log = optimize(
    fobj=fobj,
    dim=dim,
    low_limit=-6.0,
    high_limit=6.0,
    N=100
)
print(log)
