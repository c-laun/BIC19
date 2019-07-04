"""Example MLP script for one single hidden layer. This version differs slightly
from the lecture in that it uses a momentum term. We have chosen to add it to speed the simulations up a little bit.

"""

import numpy as np
import pylab as pl
pl.style.use('ggplot')
from scipy.special import expit

fname = 'xor_data'
SEED = 734589
f_act = lambda v: np.maximum(v, 0.0)
f_act_prime = lambda v: np.where(v>0.0, np.ones_like(v), np.zeros_like(v))
K = 3
eta_b = 0.2 / K
eta_V = 0.2 / K
eta_W = 0.2 / K
mu_W = 0.5
mu_V = 0.5
mu_b = 0.5
T = np.random.randint(0, 4, 10000)

Cost = lambda Y, YT: -1./2. * np.mean((Y - YT)**2)

np.random.seed(SEED)
data = np.load(fname+".npz")
V = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
W = np.array([[1.0, -2.0, 1.0]])
b = np.array([0.0, -1.0, 0.0])

def propagate(X):
    U1 = np.dot(V, X.T).T + b
    Z = f_act(U1)
    U2 =  np.dot(W, Z.T).T
    Y = U2
    return Y, Z, U1, U2

def backpropagate(X, YT):
    Y, Z, U1, U2 = propagate(X)

    ###########################
    # INSERT YOUR CODE HERE !!!
    cost = Cost(Y, YT)

    dLdY = -(Y-YT)[None, ...]
    dYdU2 = np.ones((1, 1))
    dU2dZ = W
    dU2dW = Z[None, ...]
    dZdU1 = f_act_prime(U1)[None, ...]
    dU1db = np.ones((3, 3))
    dU1dV = X[None, ...]

    dLdW = dLdY@dYdU2@dU2dW
    dLdZ = dLdY@dYdU2@dU2dZ
    dLdU1 = dLdZ*dZdU1
    dLdV = dLdU1.T@dU1dV
    dLdb = dLdU1@dU1db

    dW = dLdW
    dV = dLdV
    db = dLdb[0]

    ###########################
    return db, dV, dW, cost


X = data['inp']
YT = data['out']
cost = np.zeros(len(T))
# prepare momentum term variables
delta_W = np.zeros((1, K))
delta_b = np.zeros(K)
delta_V = np.zeros((K, 1))


def runandplot():
    for run, inp in enumerate(T):
        db, dV, dW, cost[run] = backpropagate(X[inp], YT[inp])

        # calculate weight update with momentum
        delta_b = eta_b * db + mu_b * delta_b
        delta_V = eta_V * dV + mu_W * delta_V
        delta_W = eta_W * dW + mu_W * delta_W
        # update weights
        b += delta_b
        V += delta_V
        W += delta_W

    pl.plot(cost)
    pl.show()

runandplot()
W[0, 1] = 0
runandplot()





