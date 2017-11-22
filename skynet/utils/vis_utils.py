from math import sqrt, ceil
import numpy as np
import matplotlib.pyplot as plt

def visualize_grid(Xs, ubound=255.0, padding=1):
  """
  Reshape a 4D tensor of image data to a grid for easy visualization.

  Inputs:
  - Xs: Data of shape (N, H, W, C)
  - ubound: Output grid will have values scaled to the range [0, ubound]
  - padding: The number of blank pixels between elements of the grid
  """
  (N, H, W, C) = Xs.shape
  grid_size = int(ceil(sqrt(N)))
  grid_height = H * grid_size + padding * (grid_size - 1)
  grid_width = W * grid_size + padding * (grid_size - 1)
  grid = np.zeros((grid_height, grid_width, C))
  next_idx = 0
  y0, y1 = 0, H
  for y in range(grid_size):
    x0, x1 = 0, W
    for x in range(grid_size):
      if next_idx < N:
        img = Xs[next_idx]
        low, high = np.min(img), np.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        # grid[y0:y1, x0:x1] = Xs[next_idx]
        next_idx += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding
  # grid_max = np.max(grid)
  # grid_min = np.min(grid)
  # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
  return grid

def vis_grid(Xs):
  """ visualize a grid of images """
  (N, H, W, C) = Xs.shape
  A = int(ceil(sqrt(N)))
  G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
  G *= np.min(Xs)
  n = 0
  for y in range(A):
    for x in range(A):
      if n < N:
        G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n,:,:,:]
        n += 1
  # normalize to [0,1]
  maxg = G.max()
  ming = G.min()
  G = (G - ming)/(maxg-ming)
  return G

def vis_nn(rows):
  """ visualize array of arrays of images """
  N = len(rows)
  D = len(rows[0])
  H,W,C = rows[0][0].shape
  Xs = rows[0][0]
  G = np.ones((N*H+N, D*W+D, C), Xs.dtype)
  for y in range(N):
    for x in range(D):
      G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]
  # normalize to [0,1]
  maxg = G.max()
  ming = G.min()
  G = (G - ming)/(maxg-ming)
  return G

from ..utils.data_utils import generate_decision_boundary_data
def visualize_decision_boundary(f, X, y, save_path=None):
    '''
    Plot the decision boundary given by function f.

    @Input
    - f: function
    - X: scatter points' coordinate
    - y: scatter points' label
    '''
    # generate grid data
    xx, yy, grid = generate_decision_boundary_data(X)

    probs = f(grid) # predict using the classifier
    print('xx, yy, grid, probs shape: ', xx.shape, yy.shape, grid.shape, probs.shape)
    probs = probs.reshape(xx.shape)

    # plot contour with x coordinates, y coordinates, and corresponding function value
    f, ax = plt.subplots(figsize=(8, 6))
    plt.title("decision boundary with contour")
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X[:, 0], X[:, 1], c=y[:], s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")

    # plot decision boundary
    f, ax = plt.subplots(figsize=(8, 6))
    plt.title("decision boundary")
    ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6) # grey color map

    ax.scatter(X[:, 0], X[:, 1], c=y[:], s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")
    if not save_path:
        plt.show()
    else:
        plt.savefig(save_path)

def visualize_regression(f, X, y):
    pass

def visualize_cross_validation(results: dict):
    """
    Visualize cross validation accuracy in scatter plot.

    Inputs
    ------
    - results: dict, maps from <learning rate, regularization strength> to accuracy

    Returns
    -------
    None
    """
    import math
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    # plot training accuracy
    marker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors) # default size of markers is 20
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('weight scale')
    plt.title('cross validation performance')
    plt.show()
