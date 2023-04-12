import numpy as np
import layer
import activations


class Linear(layer):
    def __init__(self, shape, in_dim, out_dim, act=None, bias=False):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.weight = np.random.randn(out_dim, in_dim)
        if self.bias:
            weight_bias = np.random.randn(out_dim, 1)
            self.weight = np.c_(weight_bias, self.weight)
        if act == "relu":
            self.act = activations.relu
            self.dact = activations.drelu
        elif act == "tanh":
            self.act = activations.tanh
            self.dact = activations.dtanh
        elif act == "sigmoid":
            self.act = activations.sigmoid
            self.dact = activations.dsigmoid
        elif act == "softmax":
            self.act = activations.softmax
        else:
            self.act = None
        # store the input and output of this layer for future backward
        self.X_in_history = []
        self.y_out_history = []

        self.grad = None  # the gradient of each parameter of this layer

    def forward(self, X):
        if self.bias:
            bias = np.ones((X.shape[0], 1))
            X = np.c_(bias, X)  # concat
        out = np.dot(X * self.weight.T)  # use dot function instead of *(element-wise mul)
        if self.act:
            out = self.act(out)
        self.X_in_history.append(X)
        self.y_out_history.append(out)
        return out

    def backward(self, dy):
        """
        Input: dy: the delta of this layer
        Output: dx: the delta of the previous layer (to compute)
        """
        X_in = self.X_in_history.pop()
        y_out = self.y_out_history.pop()

        if self.dact:  # calculate the delta of activation
            dy = dy * self.dact(y_out)

        # calculate the gradient
        self.grad = np.dot(dy, X_in.T)

        # compute the delta of previous layer
        dx = np.dot(self.weight.T, dy)
        return dx
