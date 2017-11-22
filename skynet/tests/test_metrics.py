import numpy as np

from ..utils.error_utils import rel_error
from ..utils.metrics import *

def test_calculate_pmf():
    y = np.array([1, 1, 2, 3, 3, 4, 4, 4])
    pmf = calculate_pmf(y)
    assert pmf == {
        1: 0.25,
        2: 0.125,
        3: 0.25,
        4: 3/8,
    }

    y = np.array([1])
    pmf = calculate_pmf(y)
    assert pmf == {
        1: 1,
    }
    pass

def test_calculate_entropy():
    y = np.array([1])
    pmf = calculate_pmf(y)
    assert calculate_entropy(pmf) == 0
    y = np.array([1, 2, 2, 2])
    pmf = calculate_pmf(y)
    assert rel_error(calculate_entropy(pmf), 0.81127812445913283) < 1e-16

def test_calculate_variance():
    y = np.array([1])
    assert calculate_variance(y) == 0
    y = np.array([1, 1])
    assert calculate_variance(y) == 0
    y = np.array([1, 2])
    assert calculate_variance(y) == 0.5
