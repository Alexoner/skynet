import numpy as np
from ....base import BaseClassifier
from .linear_svm import *
from .softmax import *

class LinearClassifier(BaseClassifier):
    pass

class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  # TODO: migrate to new interface defined in super class
  # def loss(self, X, y=None):
    # """
    # Compute the loss function and its derivative for a minibatch data.
    # Subclasses will override this.

    # Inputs:
    # - X: The design matrix, a numpy array of shape (N, d_1, ..., d_k) containing a
      # minibatch of N data points; each point has dimension D = d_1*...*d_k.
    # - y: A numpy array of shape (N,) containing labels/value for X[i].
    # - reg: (float) regularization strength.

    # Returns:
    # If y is None, then run a test-time forward pass of the model and return:
    # - scores: Array of shape (N, C) giving classification scores, where
      # scores[i, c] is the classification score for X[i] and class c.

    # If y is not None, then run a training-time forward and backward pass and
    # return a tuple of:
    # - loss: Scalar value giving the loss
    # - grads: Dictionary with the same keys as self.params, mapping parameter
      # names to gradients of the loss with respect to those parameters.
    # """
    # return svm_loss_vectorized(self.W, X, y, self.reg)

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  # TODO: migrate to new interface defined in super class
  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)


