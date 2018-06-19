import numpy as np
import matplotlib.pyplot as plt

from ..utils.vis_utils import visualize_decision_boundary

def test_contour():
    nx, ny = 100, 100
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)

    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)

    plt.contourf(x, y, z)
    plt.show()

def test_visualize_decision_boundary():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    X, y = make_classification(1000, 2, 2, 0, weights=[.5, .5], random_state=15)
    clf = LogisticRegression().fit(X[:100], y[:100])
    visualize_decision_boundary(lambda x: clf.predict_proba(x)[:, 1], X[100:], y[100:])

def test_line():
    # plot regression curve
    # sample data
    x = np.arange(10)
    # y = 5*x + 10
    y = np.sin(x) + 10

    # fit with np.polyfit
    # m, b = np.polyfit(x, y, 1)

    plt.plot(x, y, '.')
    plt.plot(x, y, '-')
    # plt.plot(x, m*x + b, '-')
    plt.title("regression curve")
    plt.show()
    pass
