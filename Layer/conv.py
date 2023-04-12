import layer
import numpy as np
import activations
import math
from C_Graph.variable import Variable
from C_Graph.operator import Operator


class Conv2D(Operator):
    def __init__(self, kernel_shape:list, input_variable:Variable, name:str, stride=1, padding=0, act=None, bias=False):
        # kernel_shape: [out_channel, in_channel, kernel_height, kernel_width]
        for i in kernel_shape:
            if not isinstance(i, int):
                raise Exception("Operator Conv2D name: %s kernel shape is not list of int" % self.name)

        if not isinstance(input_variable, Variable):
            raise Exception("Operator Conv2D name: %s's input_variable is not instance of Variable" % self.name)

        if len(input_variable.shape) != 4:
            raise Exception("Operator Conv2D name: %s's input_variable's shape != 4d Variable!" % self.name)

        self.out_channel = kernel_shape[0]
        self.in_channel = kernel_shape[1]
        self.kernel_height = kernel_shape[2]
        self.kernel_width = kernel_shape[3]
        self.bias = bias
        self.shape = input_variable.shape  # [N, C, H, W] N=Batch_size; C=channels; H=height; W=width
        self.batch_size = self.shape[0]
        self.X_h = self.shape[2]
        self.X_w = self.shape[3]

        self.weight = Variable(kernel_shape, scope=name, name='weight', grad=True, learnable=True)
        if self.bias:
            self.weight_bias = Variable([self.out_channel], scope=name, name='weight_bias', grad=True, learnable=True)

        # weight_scale = math.sqrt(self.kernel_height * self.kernel_width * self.in_channel / 2)  # msra method
        # self.weight = np.random.randn(out_channel, in_channel, self.kernel_height, self.kernel_width) / weight_scale
        # if self.bias:
        #     self.weight_bias = np.random.randn(out_channel) // weight_scale
        # self.w_gradient = np.zeros(self.weight.shape)
        # self.b_gradient = np.zeros(self.weight_bias.shape)

        if stride and isinstance(stride, tuple):
            self.stride_h, self.stride_w = stride
        elif stride and isinstance(stride, int):
            self.stride_w = self.stride_h = stride
        else:
            self.stride_w = self.stride_h = 1

        if padding and isinstance(padding, tuple):
            self.padding_h, self.padding_w = padding
        elif padding and isinstance(padding, int):
            self.padding_w = self.padding_h = padding
        else:
            self.padding_h = self.padding_w = 0

        # if act == "relu":
        #     self.act = activations.relu
        #     self.dact = activations.drelu
        # elif act == "tanh":
        #     self.act = activations.tanh
        #     self.dact = activations.dtanh
        # elif act == "sigmoid":
        #     self.act = activations.sigmoid
        #     self.dact = activations.dsigmoid
        # elif act == "softmax":
        #     self.act = activations.softmax
        # else:
        #     self.act = None

        # eta: the partial derivative of loss w.r.t conv_out (the same shape with conv_out)
        self.eta = np.zeros(self.batch_size,
                            self.out_channel,
                            (self.X_h - self.kernel_height + 2 * self.padding_h) // self.stride_h + 1,
                            (self.X_w - self.kernel_width + 2 * self.padding_w) // self.stride_w + 1)
        super(Conv2D, self).__init__()

    def forward(self, X):
        """
        Using im2col method to forward (refer to: https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf)
        :param X: [N, C, H, W] N=Batch_size; C=channels; H=height; W=width
        :return: the output of this batch, shape = [N,
                                                    out_channel,
                                                    (X_h - self.kernel_height + 2 * self.padding_h) / self.stride_h + 1,
                                                    (X_w - self.kernel_width + 2 * self.padding_w) / self.stride_w + 1]
        """
        print("self.Unknown is : " + self.Unknown)

        if self.padding_w != 0 and self.padding_h != 0:
            X = np.pad(X, ((0, 0), (0, 0), (self.padding_h, self.padding_h), (self.padding_w, self.padding_w)),
                       'constant', constant_values=0)
        row_weight = self.weight.reshape(self.out_channel,
                                         -1)  # shape = out_channel * (in_channel * kernel_height * kernel_width)

        self.col_X = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batch_size):
            col_x = self.im2col(X[i])
            self.col_X.append(col_x)
            if self.bias:
                conv_out[i] = np.reshape(np.dot(row_weight, col_x.T) + self.weight_bias[:, np.newaxis, np.newaxis],
                                         self.eta[0].shape)
            else:
                conv_out[i] = np.reshape(np.dot(row_weight, col_x.T), self.eta[0].shape)
        self.col_X = np.array(self.col_X)
        return conv_out

    def im2col(self, x):
        """
        convert an image to columns
        :param x: with shape [in_channel, H, W] (H and W may include padding)
        :return col_x: with shape [((H-kernel_height) / stride_h + 1) * ((W-kernel_width) / stride_w + 1), in_channel * kernel_height * kernel_width]
        """
        col_x = []
        for i in range(0, x.shape[1], self.stride_h):
            for j in range(0, x.shape[2], self.stride_w):
                col = x[:, i:i + self.kernel_height, j:j + self.kernel_width].reshape(
                    [-1])  # C * kernel_height * kernel_width
                col_x.append(col)
        col_x = np.array(col_x)
        return col_x

    def gradient(self, eta):
        # step 1: calculate a(loss) / a(weight)
        self.eta = eta
        col_eta = np.reshape(self.eta, (self.batch_size, self.out_channel, -1))
        for i in range(self.batch_size):
            self.w_gradient += np.reshape(np.dot(col_eta[i], self.col_X[i]), self.weight.shape)
        if self.bias:
            self.b_gradient += np.sum(col_eta, axis=(0, 2))
        # step 2: calculate the delta passed to the previous layer
        pad_eta = np.pad(self.eta, (0, 0), (0, 0),
                         (self.stride_h * (self.X_h - 1) + self.kernel_height - self.eta.shape[2]) // 2,
                         (self.stride_w * (self.X_w - 1) + self.kernel_width - self.eta.shape[3]) // 2,
                         'constant', constant_values=0)
        # rotate weights 180 degree
        flipped_weight = self.rotate180(self.weight)
        # transpose the weight
        flipped_weight = flipped_weight.swapaxes(0, 1)
        col_flip_weight = flipped_weight.reshape([self.in_channel, -1])
        col_pad_eta = np.array([self.im2col(pad_eta[i]) for i in range(self.batch_size)])
        next_eta = []
        for i in range(self.batch_size):
            next_eta_i = np.reshape(np.dot(col_flip_weight, col_pad_eta[i].T), (self.in_channel, self.X_h, self.X_w))
            next_eta.append(next_eta_i)
        next_eta = np.array(next_eta)
        return next_eta

    def rotate180(self, weight):
        rot_weight = np.zeros(weight.shape)
        for out_channel in range(weight.shape[0]):
            for in_channel in range(weight.shape[1]):
                rot_weight[out_channel, in_channel, :, :] = np.flipud(
                    np.fliplr(rot_weight[out_channel, in_channel, :, :]))
        return rot_weight

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.b_gradient
        # reset the gradient
        self.w_gradient = np.zeros(self.weight.shape)
        self.b_gradient = np.zeros(self.weight_bias.shape)


if __name__ == "__main__":
    # img = np.random.standard_normal((2, 32, 32, 3))
    img = np.ones((1, 32, 32, 3))
    img *= 2
    conv = Conv(img.shape, 12, 3, 1)
    next = conv.forward(img)
    next1 = next.copy() + 1
    conv.gradient(next1 - next)
    print(conv.w_gradient)
    print(conv.b_gradient)
    conv.backward()
