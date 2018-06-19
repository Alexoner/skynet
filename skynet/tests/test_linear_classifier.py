import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from skynet.linear.k_nearest_neighbor import KNearestNeighbor
from skynet.linear.linear_classifier import Softmax, LinearSVM
from skynet.utils.data_utils import generate_decision_boundary_data
from skynet.linear.softmax import softmax_loss_naive
from skynet.utils.error_utils import rel_error
from ..utils.vis_utils import visualize_decision_boundary


X_dev, y_dev = make_classification(n_features=3073)
W = np.random.randn(3073, 10) * 0.0001

def test_softmax_sanity_check():
    # Generate a random softmax weight matrix and use it to compute the loss.
    loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

    # As a rough sanity check, our loss should be something close to -log(0.1).
    print('loss: %f' % loss)
    print('sanity check: %f' % (-np.log(0.1)))
    assert(rel_error(loss, -np.log(0.1)) <= 2e-2)

def test_softmax_gradient_check():
    # Complete the implementation of softmax_loss_naive and implement a (naive)
    # version of the gradient that uses nested loops.
    loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

    # As we did for the SVM, use numeric gradient checking as a debugging tool.
    # The numeric gradient should be close to the analytic gradient.
    from skynet.utils.gradient_check import grad_check_sparse
    f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
    grad_check_sparse(f, W, grad, 10)

    # similar to SVM case, do another gradient check with regularization
    loss, grad = softmax_loss_naive(W, X_dev, y_dev, 1e2)
    f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 1e2)[0]
    grad_check_sparse(f, W, grad, 10)

    # Now that we have a naive implementation of the softmax loss function and its gradient,
    # implement a vectorized version in softmax_loss_vectorized.
    # The two versions should compute the same results, but the vectorized version should be
    # much faster.
    tic = time.time()
    loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.00001)
    toc = time.time()
    print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

    from skynet.linear.softmax import softmax_loss_vectorized
    tic = time.time()
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.00001)
    toc = time.time()
    print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

    # As we did for the SVM, we use the Frobenius norm to compare the two versions
    # of the gradient.
    loss_difference = np.abs(loss_naive - loss_vectorized)
    grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
    print('Loss difference: %f' % loss_difference)
    print('Gradient difference: %f' % grad_difference)
    # print('Loss difference:', loss_difference, loss_difference <= 4e-16, grad_difference)
    assert(loss_difference <= 1e-15)
    assert(grad_difference <= 1e-14)

def test_softmax_small_data():

    def generate_data():
        # X, y = make_classification(n_features=2, n_samples=1000)
        X, y = make_classification(10000, 2, 2, 0, weights=[.5, .5], random_state=15)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        return X_train, y_train, X_test, y_test

    X_train, y_train, X_test, y_test = generate_data()
    model = Softmax()
    model.train(X_train, y_train, learning_rate=1e-4)
    y_test_pred = model.predict(X_test)

    # plot
    # xx, yy, grid = generate_decision_boundary_data()
    # z = probs = model.predict(grid).reshape(xx.shape)
    # print("z: ", z.shape, "xx: ", xx.shape)

    # fig, ax = plt.subplots(figsize=(8, 6))
    # contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          # vmin=0, vmax=1)

    # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test[:], s=50,
               # cmap="RdBu", vmin=-.2, vmax=1.2,
               # edgecolor="white", linewidth=1)
    # ax_c = fig.colorbar(contour)
    # ax_c.set_label("$P(y = 1)$")
    # ax_c.set_ticks([0, .25, .5, .75, 1])
    # ax.set(aspect="equal",
           # xlim=(-5, 5), ylim=(-5, 5),
           # xlabel="$X_1$", ylabel="$X_2$")
    # plt.savefig('linear_softmax_classification.png')
    visualize_decision_boundary(lambda x: model.predict(x), X_test, y_test, "linear_softmax.png")

    test_accuracy = np.mean(y_test == y_test_pred)
    print('softmax on small data final test set accuracy: %f' % (test_accuracy, ))
    assert(test_accuracy >= .90)

def test_softmax_large_data():
    # if number of features larger than number of samples, add prior/regularization
    X, y = make_classification(10_000, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = Softmax()
    model.train(X_train, y_train, reg=1e-6)

    y_test_pred = model.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('softmax test set accuracy: %f' % (test_accuracy, ))
    assert(test_accuracy >= .85)


def test_svm_small_data():

    def generate_data():
        # X, y = make_classification(n_features=2, n_samples=1000)
        X, y = make_classification(10000, 2, 2, 0, weights=[.5, .5], random_state=15)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        return X_train, y_train, X_test, y_test

    X_train, y_train, X_test, y_test = generate_data()
    model = LinearSVM()
    model.train(X_train, y_train, learning_rate=1e-4, batch_size=500)
    y_test_pred = model.predict(X_test)

    # plot
    visualize_decision_boundary(lambda x: model.predict(x), X_test, y_test, "linear_svm.png")

    test_accuracy = np.mean(y_test == y_test_pred)
    print('svm on small data final test set accuracy: %f' % (test_accuracy, ))
    assert(test_accuracy >= .90)

def test_knn_small_data():

    def generate_data():
        # X, y = make_classification(n_features=2, n_samples=1000)
        X, y = make_classification(50, 2, 2, 0, weights=[.5, .5], random_state=15)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        return X_train, y_train, X_test, y_test

    X_train, y_train, X_test, y_test = generate_data()
    model = KNearestNeighbor()
    model.train(X_train, y_train)
    y_test_pred = model.predict(X_test, k=1)

    # plot
    visualize_decision_boundary(lambda x: model.predict(x), X_test, y_test, "linear_knn.png")

    test_accuracy = np.mean(y_test == y_test_pred)
    print('knn on small data final test set accuracy: %f' % (test_accuracy, ))
    assert(test_accuracy >= .70)
