import numpy as np
from random import shuffle

def mean_squared_error(W, X, y, reg):
    """
    Mean squared loss function

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    N, D = X.shape
    scores = X.dot(W) # N x C

    loss_data = 1/2 * (y - scores).T @ (y - scores) / N
    loss_reg = reg / 2 * np.sum(W**2)
    loss = np.sum(loss_data + loss_reg)

    dscores = -(y - scores) / N
    dW_data = X.T @ dscores

    dW_reg = reg * W
    dW = dW_data + dW_reg

    print('loss:', loss)
    return loss, dW
