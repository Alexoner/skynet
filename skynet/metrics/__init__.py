from collections import Counter
import numpy as np

def calculate_pmf(y):
    """
    calculate the probability mass function given sample data.
    """
    assert(len(y.shape) == 1)
    # values = np.unique(y)
    counter = Counter(y)
    for k, v in counter.items():
        counter[k] = v / len(y)
    return counter

def calculate_entropy(pmf):
    """
    calculate entropy of a probability mass function

    Inputs
    ------
    - pmf: list or numpy array, dimension (N, ), probability mass function, or dict
    """
    if isinstance(pmf, dict):
        pmf = list(pmf.values())
    entropy = sum([-p * np.log2(p) for p in pmf])
    print("entropy: ", entropy, pmf)
    return entropy

def calculate_gini_index(pmf: list):
    if isinstance(pmf, dict):
        pmf = list(pmf.values())
    return 1 - sum([p**2 for p in pmf])

def calculate_variance(X):
    mu = np.mean(X)
    return np.sum([(x - mu)**2 for x in X])

def calculate_cross_entropy(Y, T):
    return sum([-T[i] * np.log2(y) for i, y in enumerate(Y)])

