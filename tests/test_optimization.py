import pytest
import numpy as np

import de.benchmarks
import de.optimization


# Defining an objective function
def fobj(x):
    return de.benchmarks.f_ackley(x, 20.0, 0.2, 2.0*np.pi)


@pytest.mark.parametrize("dim", [3, 4, 5])
def test_ackley(dim):

    sol, fobj_sol, conv_flag, log = de.optimization.optimize(
        fobj=fobj,
        dim=dim,
        low_limit=-6.0,
        high_limit=6.0,
        N=50
    )

    assert sol == pytest.approx(np.zeros(dim), abs=1e-5)
    assert fobj_sol == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_rosenbrock(dim):

    sol, fobj_sol, conv_flag, log = de.optimization.optimize(
        fobj=de.benchmarks.f_rosenbrock,
        dim=dim,
        low_limit=-6.0,
        high_limit=6.0,
        N=100
    )

    assert sol == pytest.approx(np.ones(dim), rel=5e-2)
    assert fobj_sol == pytest.approx(0.0, abs=1e-5)
