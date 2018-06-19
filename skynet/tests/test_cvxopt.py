from cvxopt import matrix, solvers

def test_matrix():
    A = matrix(1 + 1j, (1, 4))
    print(A)
    # assert False

def test_linear_programming():
    c = matrix([-4., -5.])
    G = matrix([[2., 1., -1., 0.], [1., 2., 0., -1.]])
    h = matrix([3., 3., 0., 0.])
    sol = solvers.lp(c, G, h)

    print(sol)
    assert not sol

import cvxpy as cvx

# Create two scalar optimization variables.
x = cvx.Variable()
y = cvx.Variable()

# Create two constraints.
constraints = [x + y == 1,
               x - y >= 1]

# Form objective.
obj = cvx.Minimize((x - y)**2)

# Form and solve problem.
prob = cvx.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)
