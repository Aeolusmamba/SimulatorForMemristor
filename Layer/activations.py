import numpy as np


def relu(x):
    return max(x, 0)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1-np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def drelu(x):
    y = np.zeros_like(x)
    y[x>0] = 1
    return y

def dtanh(x):
    return 1-np.tanh(x)**2

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def dsoftmax(X, y_i):  # with cross_entropy
    X[y_i] = X[y_i] - 1
    return X
