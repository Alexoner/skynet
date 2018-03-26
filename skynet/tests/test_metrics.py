import numpy as np

from ..utils.error_utils import rel_error
from ..metrics import *

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

def test_precision_recall_f1_score():
    y_true = np.array([0, 1, 1, 0, 1, 1, 1])
    y_pred = [1, 1, 1, 0, 0, 1, 0]
    assert(precision_score(y_pred, y_true) == 0.75)
    assert(recall_score(y_pred, y_true) == 0.6)
    assert(format(f1_score(y_pred, y_true), '.6f') == '0.666667')
