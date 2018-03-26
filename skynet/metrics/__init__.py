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

def precision_score(Y, T):
    """
    calculate true positive ratio for binary classification: 0 and 1 labeled
    """
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    if not isinstance(T, np.ndarray):
        T = np.array(T)
    tp = (Y[T == 1] == 1) # true positive
    pp = Y == 1 # predicted positive: true positive + false positive
    precision = np.sum(tp) / np.sum(pp)
    return precision

def recall_score(Y, T):
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    if not isinstance(T, np.ndarray):
        T = np.array(T)
    tp = (Y[T == 1] == 1) # true positive
    sp = T == 1# all sample positives: true positive + false negative
    recall = np.sum(tp) / np.sum(sp)
    return recall

def f1_score(Y, T):
    """
    F1 score is the harmonic mean of precision and recall
    """
    precision = precision_score(Y, T)
    recall = recall_score(Y, T)
    f1 = 2*precision*recall/(precision + recall)
    print(precision, recall, f1, format(f1, '.6f'))
    return f1
