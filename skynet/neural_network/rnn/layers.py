import numpy as np
from ..layers import *



def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    # Compute output
    mu = x.mean(axis=0)
    xc = x - mu
    var = np.mean(xc ** 2, axis=0)
    std = np.sqrt(var + eps)
    xn = xc / std
    out = gamma * xn + beta

    cache = (mode, x, gamma, xc, std, xn, out)

    # Update running average of mean
    running_mean *= momentum
    running_mean += (1 - momentum) * mu

    # Update running average of variance
    running_var *= momentum
    running_var += (1 - momentum) * var
  elif mode == 'test':
    # Using running mean and variance to normalize
    std = np.sqrt(running_var + eps)
    xn = (x - running_mean) / std
    out = gamma * xn + beta
    cache = (mode, x, xn, gamma, beta, std)
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  mode = cache[0]
  if mode == 'train':
    mode, x, gamma, xc, std, xn, out = cache

    N = x.shape[0]
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(xn * dout, axis=0)
    dxn = gamma * dout
    dxc = dxn / std
    dstd = -np.sum((dxn * xc) / (std * std), axis=0)
    dvar = 0.5 * dstd / std
    dxc += (2.0 / N) * xc * dvar
    dmu = np.sum(dxc, axis=0)
    dx = dxc - dmu / N
  elif mode == 'test':
    mode, x, xn, gamma, beta, std = cache
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(xn * dout, axis=0)
    dxn = gamma * dout
    dx = dxn / std
  else:
    raise ValueError(mode)

  return dx, dgamma, dbeta

