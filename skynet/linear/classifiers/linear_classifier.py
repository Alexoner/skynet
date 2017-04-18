import numpy as np
from ...base import BaseClassifier
from .linear_svm import *
from .softmax import *

class LinearClassifier(BaseClassifier):

  @property
  def W(self):
      return self.params['W']

  @W.setter
  def W(self, W):
      self.params['W'] = W

  def predict(self, X):
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    scores = self.loss(X)
    y_pred = np.argmax(scores, axis=-1) # top scoring class

    return y_pred

class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  # TODO: migrate to new interface defined in super class
  def loss(self, X_batch, y_batch=None, reg=0.0):
    if y_batch is None:
        return X_batch.dot(self.params['W'])
    return svm_loss_vectorized(self.params['W'], X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  # TODO: migrate to new interface defined in super class
  def loss(self, X_batch, y_batch=None, reg=0.0):
    if y_batch is None:
        return X_batch.dot(self.params['W'])
    return softmax_loss_vectorized(self.params['W'], X_batch, y_batch, reg)


