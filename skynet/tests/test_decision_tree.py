import sys
import numpy as np
from sklearn import tree, ensemble
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from ..tree.decision_tree import CART
from ..utils.vis_utils import visualize_decision_boundary

def notest_sklearn_tree_regression():
    clf = tree.DecisionTreeRegressor()
    # X, y = make_regression()
    X = 3 * np.random.rand(100, 1)
    w = np.array([[3.11],])
    b = np.random.rand()
    # y = X @ w + b # + np.random.rand(100, 1) # plus noise
    y = np.sin(X) + b # + np.random.rand(100, 1) # plus noise

    X_test = 3* np.random.rand(10, 1)
    y_test = np.sin(X_test) + b

    clf.fit(X, y)
    y_test_pred = clf.predict(X_test)

    residual = y_test_pred - y_test.reshape(-1)
    sys.stderr.write("residual: \n%s\n" % residual)

    # plt.show()
    assert(np.max(residual) < 0.1)

def test_tree_classification():
    # X, y = make_classification()
    X, y = make_classification(5000, 2, 2, 0, weights=[.5, .5], random_state=15)

    plt.plot(X, y, '.')
    # plt.plot(X, y, '-')
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # initialize model
    clf = CART()
    clf.train(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    # plt.plot(X_test, y_test, '.')
    # plt.plot(X_test, y_test_pred, '-')
    residual = y_test_pred == y_test.reshape(-1)
    accuracy = np.mean(y_test_pred == y_test)
    print("target: \n%s\n" % y_test)
    print("prediction: \n%s\n" % y_test_pred.reshape(-1, 1))
    print("residual: \n%s\n" % (residual.reshape(-1, 1)))
    # plt.plot(X_test, y_test, '.')
    # plt.plot(X_test, y_test_pred, '.')
    print(y_test_pred)
    print(y_test.reshape(-1))
    visualize_decision_boundary(lambda x: clf.predict(x), X, y)

    # plt.show()
    assert(accuracy >= 0.9)

def test_tree_regression():
    # synthesize training and test data
    # X, y = make_classification()
    X = 3 * np.random.rand(100, 1)
    w = np.array([[3.11],])
    b = np.random.rand()
    y = X @ w + b # + np.random.rand(100, 1) # plus noise

    X_test = 3* np.random.rand(10, 1)
    # X_test = np.array([
        # [-3],
        # [-2],
        # [-1],
        # [1],
        # [1.1],
        # [1.2],
        # [1.3],
        # [1.4],
        # [1.5],
        # [2],
        # [2.5],
        # [3],
        # [4],
        # [5],
        # [10],
    # ])

    f, ax = plt.subplots(figsize=(8, 6))
    ax.plot(X, y, '.')
    # ax.plot(X, y, '-')
    y_test = X_test @ w + b

    # initialize model
    clf = CART()
    clf.min_leaf_size = 2
    clf.train(X, y)
    y_test_pred = clf.predict(X_test)

    # ax.plot(X_test, y_test, '.')
    ax.plot(X_test, y_test_pred, '.')
    ax.plot(X_test, y_test_pred, '-')
    residual = y_test_pred - y_test.reshape(-1)
    print("target: \n%s\n" % y_test)
    print("prediction: \n%s\n" % y_test_pred.reshape(-1, 1))
    print("residual: \n%s\n" % (residual.reshape(-1, 1)))
    print(y_test_pred)
    print(y_test.reshape(-1))

    # plt.show()
    plt.show()
    assert(np.max(residual) < 0.1)
