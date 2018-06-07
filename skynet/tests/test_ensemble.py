import numpy as np
from sklearn import tree, ensemble
from sklearn.datasets import make_classification, make_regression
import matplotlib.pyplot as plt

def test_sklearn_ensemble_regression():
    clf = ensemble.GradientBoostingRegressor()
    # X, y = make_regression()
    X = 3 * np.random.rand(100, 1)
    w = np.array([[3.11],])
    b = np.random.rand()
    y = X @ w + b # + np.random.rand(100, 1) # plus noise

    clf.fit(X, y.ravel())
    X_test = np.array([
        [-3],
        [-2],
        [-1],
        [1],
        [1.1],
        [1.2],
        [1.3],
        [1.4],
        [1.5],
        [2],
        [2.5],
        [3],
        [4],
        [5],
        [10],
    ])
    plt.plot(X, y, '.')
    plt.plot(X, y, '-')
    y_test = X_test * w + b
    y_test_pred = clf.predict(X_test)

    # plt.plot(X_test, y_test, '.')
    # plt.plot(X_test, y_test_pred, '-')
    print("Prediction, target: " )
    print(y_test_pred)
    print(y_test.reshape(-1))

    plt.show()
    assert(False)

def test_sklearn_tree_classification():
    clf = tree.DecisionTreeClassifier()
    X, y = make_classification()
    pass

