import numpy as np
import matplotlib.pyplot as plt

def test_decision_boundary():
    nx, ny = 100, 100
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)

    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)

    plt.contourf(x, y, z)
