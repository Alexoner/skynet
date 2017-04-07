import numpy as np

class BaseEstimator(object):

  def __init__(self, reg=0.0, dtype=np.float32):
    """
    Initialize a model.

    Inputs:
    - reg: Scalar giving L2 regularization strength.
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

  @abstract_method
  def loss(self, X, y=None):
    """
    Compute the loss function and its derivative for a minibatch data.
    Subclasses will override this.

    Inputs:
    - X: The design matrix, a numpy array of shape (N, d_1, ..., d_k) containing a
      minibatch of N data points; each point has dimension D = d_1*...*d_k.
    - y: A numpy array of shape (N,) containing labels/value for X[i].
    - reg: (float) regularization strength.

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    pass

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this model using stochastic gradient descent.
    This is a naive implementation, for better performance, consider using
    skynet.solvers.solver.

    Inputs:
    - X: A numpy array of shape (N, d_1, .., d_k) containing training data; there
      are N training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      batch_mask = np.random.choice(num_train, batch_size)
      X_batch = X[batch_mask, ...]
      y_batch = y[batch_mask]

      pass
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      step = -learning_rate * grad
      self.W += step
      pass
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: N x D array of training data. Each column is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in scores.            #
    ###########################################################################
    scores = self.loss(X)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return scores

class BaseRegressor(BaseEstimator):
    pass

class BaseClassifier(BaseEstimator):

  def predict(self, X):
    scores = super().predict(X)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    y_pred = np.argmax(scores, axis=-1) # top scoring class
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
