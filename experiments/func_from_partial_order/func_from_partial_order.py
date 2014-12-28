# Goal:
#
# Given:
# - a set of samples (S)
# - an unknown evaluation function f of S (f: S -> [0; 1])
# - a partial order c of f over S:
#   - comparisons of pairs of samples (C(i, j) in {-1,0,1}):
#   c(i,j) = -1 iff f(S_i) < f(S_j)
#   c(i,j) =  0 iff f(S_i) = f(S_j)
#   c(i,j) =  1 iff f(S_i) > f(S_j)
#   ie. c(i, j) = sign(f(S_i) - f(S_j))
#
# we'd like to estimate the function values of f at samples S
# that best fit with the partial order c.
#
# While it is possible that some comparisons in c would be conflicting,
# the goal is to minimize the error.
#
# 
# notes:
# - we could compute g(S_i) = sum(c(i, j) for j in S if i != j)
#   - the range of g would be [-N; N] for N = len(S)
# - in order to squash g to [0;1] we could use the logistic function
# - in order words:
#   - samples compared mostly as higher then others would get high value of f
#     and vice versa

import numpy as np

def partial_order(a, b, eps=0.2):
    '''
    a, b - values of a function with range [0;1]
    '''
    # return sign(a - b)
    return eps_sign(a - b, eps)

# note: for continuous partial order use: f(a) - f(b)

def eps_sign(x, eps):
    '''signum function with 0 value for abs(x) < eps'''
    return sign(sign(x) * np.maximum(abs(x) - eps, 0))

def f(x):
    return np.exp(-0.5 * x**2)

def random_samples(n, a=0, b=1):
     return (b - a) * np.random.random(n) + a

def reconstruct(x, f, eps=0.2):
    y_0 = f(x)
    def rescale(v):
        return (v + 1) * 0.5
        # return v
    def normalize(v):
        return (v - np.min(v)) / (np.max(v) - np.min(v))
    n = len(x)
    return normalize(sum(partial_order(y_0, f(x[i]), eps) for i in range(0, n)))
