import numpy as np

from ...base import BaseRegressor
from .mean_squared_error import mean_squared_error

# TODO: bayesian linear regression
# TODO: EM algorithm optimization

class LinearRegression(BaseRegressor):

    def __init__(self, reg=0.0, dtype=np.float32):
        super().__init__(reg, dtype)

    @property
    def W(self):
        return self.params['W']

    @W.setter
    def W(self, W):
        self.params['W'] = W

    @property
    def b(self):
        return self.params['b']

    @b.setter
    def b(self, b):
        self.params['b'] = b

    def _closed_form(self, X: np.array, y, reg=0.0):
        # DONE: add a intercept term 1 to X before this call
        _, D = X.shape
        self.params['W'] = np.linalg.inv(reg * np.identity(D) + X.T @ X) @ X.T @ y
        self.params['b'] = self.W[:, -1]

    def loss(self, X, y=None, reg=0.0):
        scores = X.dot(self.params['W'])
        if y is None:
            return scores
        loss, dW = mean_squared_error(self.W, X, y, reg)
        return loss, dW
