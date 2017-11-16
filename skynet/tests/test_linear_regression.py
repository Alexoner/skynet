import numpy as np
from skynet.linear.linear_regression import LinearRegression
from skynet.utils.gradient_check import eval_numerical_gradient
from skynet.linear.mean_squared_error import mean_squared_error
import matplotlib.pyplot as plt

def test_linear_regression_closed_form_1x1():
    N = 100
    X = np.random.rand(100, 1)
    # y = np.zeroes((N, 1,))
    w = np.array([[3.11]])
    b = 6.9
    y = X @ w + b

    # add ones, intercept terms to X
    X1 = np.ones((X.shape[0], X.shape[1] + 1,))
    X1[..., :-1] = X
    print(X1.shape)

    model = LinearRegression()
    model._closed_form(X1, y)

    w_hat, b_hat = model.W, model.b
    print('Real parameters:\n', w, b)
    print('fitted:\n', w_hat[:-1, ...], b_hat)
    assert np.isclose(w_hat[:-1, ...], w).all()

def test_linear_regression_closed_form_20x10():
    N = 10000
    X = np.random.rand(100, 20)
    w = np.random.rand(20, 10)
    b = np.random.rand()
    y = X @ w + b

    # add ones, intercept terms to X
    X1 = np.ones((X.shape[0], X.shape[1] + 1,))
    X1[..., :-1] = X
    print(X1.shape)

    model = LinearRegression()
    model._closed_form(X1, y)

    w_hat, b_hat = model.W, model.b
    print('Real parameters:\n', w, b)
    print('fitted:\n', w_hat, b_hat)
    assert w_hat.shape == (21, 10)
    assert np.isclose(w_hat[:-1, ...], w).all()

def test_linear_regression_gradient_check_1x1():
    N = 1000
    X = np.random.rand(100, 1)
    # w = np.array([[3.11]])
    w = np.random.rand(1, 1)
    b = np.random.rand()
    y = X @ w + b

    # add ones, intercept terms to X
    X1 = np.ones((X.shape[0], X.shape[1] + 1,))
    X1[..., :-1] = X
    print(X1.shape)

    w1 = np.random.rand(2, 1)
    # model = LinearRegression()
    # loss, grad = model.loss(X1, y, reg=0.0)
    loss, grad = mean_squared_error(w1, X1, y, reg=1.0)
    print('analytic loss:', loss, ', gradient: ', grad)
    # model.train(X1, y, reg=0.0, num_iters=1000)
    f = lambda w: mean_squared_error(w, X1, y, reg=1.0)[0]

    grad_numerical = eval_numerical_gradient(f, w1, verbose=True)
    print('numerical loss:', loss, ', gradient: ', grad_numerical)

    assert np.isclose(grad_numerical, grad).all()

    # w_hat, b_hat = model.W, model.b
    # print('Real parameters:\n', w, b)
    # print('fitted:\n', w_hat, b_hat)
    # assert w_hat.shape == (2, 1)
    # assert np.isclose(w_hat[:-1, ...], w).all()

def test_linear_regression_train_1x1():
    N = 1000
    X = np.random.rand(100, 1)
    w = np.array([[3.11],])
    b = np.random.rand()
    y = X @ w + b

    # add ones, intercept terms to X
    X1 = np.ones((X.shape[0], X.shape[1] + 1,))
    X1[..., :-1] = X
    print(X1.shape)

    model = LinearRegression()
    # model._closed_form(X1, y)
    losses = model.train(X1, y, learning_rate=1, reg=0.0, num_iters=1000)

    w_hat, b_hat = model.W, model.b
    print('Real parameters:\n', w, b)
    print('fitted:\n', w_hat, b_hat)
    assert w_hat.shape == (2, 1)
    assert np.isclose(w_hat[:-1, ...], w).all()

def test_linear_regression_train_4x1():
    N = 1000
    X = np.random.rand(100, 4)
    w = np.array([[3.11] ,
                  [6.11],
                  [9.11],
                  [12.11]])
    b = np.random.rand()
    y = X @ w + b

    # add ones, intercept terms to X
    X1 = np.ones((X.shape[0], X.shape[1] + 1,))
    X1[..., :-1] = X
    print(X1.shape)

    model = LinearRegression()
    # model._closed_form(X1, y)
    model.train(X1, y, learning_rate=5e-1, reg=0.0, num_iters=1000)

    w_hat, b_hat = model.W, model.b
    print('Real parameters:\n', w, b)
    print('fitted:\n', w_hat, b_hat)
    assert w_hat.shape == (5, 1)
    assert np.isclose(w_hat[:-1, ...], w).all()

def test_linear_regression_train_20x10():
    N = 1000
    X = np.random.rand(100, 20)
    w = np.random.rand(20, 10)
    b = np.random.rand()
    y = X @ w + b

    # add ones, intercept terms to X
    X1 = np.ones((X.shape[0], X.shape[1] + 1,))
    X1[..., :-1] = X
    print(X1.shape)

    model = LinearRegression()
    losses = model.train(X1, y, learning_rate=2e-1, reg=1e-30, num_iters=5000)

    w_hat, b_hat = model.W, model.b
    print('Real parameters:\n', w, b)
    print('fitted:\n', w_hat, b_hat)
    assert w_hat.shape == (21, 10)
    assert np.isclose(losses[-1].max(), 1e-8)
    plt.plot(losses)
    plt.ylim((0, min(.5, np.max(losses))))
    plt.savefig('linear_linear_regression_20x10.png')
    # plt.show()
