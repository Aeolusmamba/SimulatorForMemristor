import layer
import numpy as np


class MaxPooling(layer):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        self.in_channel = in_channel
        self.out_channel = out_channel
        if isinstance(kernel_size, tuple):
            self.kernel_height, self.kernel_width = kernel_size
        else:
            self.kernel_height = self.kernel_width = kernel_size

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

    def forward(self, X):
        """
        Forward pass for maxPool layer
        :param X: [N, C, H, W] N=Batch_size; C=channels; H=height; W=width
        :return: the output of this batch, shape = [N,
                                                    out_channel,
                                                    (X_h - self.kernel_height + 2 * self.padding_h) / self.stride_h + 1,
                                                    (X_w - self.kernel_width + 2 * self.padding_w) / self.stride_w + 1]
        """
        self.in_shape = X.shape
        self.max_index = np.zeros(self.in_shape)  # using as a mask to record the location of the maximum element
        output = np.zeros([self.in_shape[0], self.in_shape[1],
                          (self.in_shape[2] - self.kernel_height + 2 * self.padding_h) // self.stride_h + 1,
                          (self.in_shape[3] - self.kernel_width + 2 * self.padding_w) // self.stride_w + 1])
        for n in self.in_shape[0]:
            for o in self.in_shape[1]:
                for i in range(0, self.in_shape[2], self.stride_h):
                    for j in range(0, self.in_shape[3], self.stride_w):
                        output[n, o, i // self.stride_h, j // self.stride_w] = np.max(X[n, o, i:i+self.stride_h, j:j+self.stride_w])
                        max_index = np.argmax(X[n, o, i:i+self.stride_h, j:j+self.stride_w])
                        self.max_index[n, o, i + max_index // self.kernel_height, j + max_index % self.kernel_width] = 1
        return output

    def backward(self, eta):
        return np.repeat(np.repeat(eta, self.kernel_height, axis=2), self.kernel_width, axis=3) * self.max_index


class AvgPooling(layer):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        self.in_channel = in_channel
        self.out_channel = out_channel
        if isinstance(kernel_size, tuple):
            self.kernel_height, self.kernel_width = kernel_size
        else:
            self.kernel_height = self.kernel_width = kernel_size

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

    def forward(self, X):
        self.in_shape = X.shape
        output = np.zeros([self.in_shape[0], self.in_shape[1],
                           (self.in_shape[2] - self.kernel_height + 2 * self.padding_h) // self.stride_h + 1,
                           (self.in_shape[3] - self.kernel_width + 2 * self.padding_w) // self.stride_w + 1])
        for n in self.in_shape[0]:
            for o in self.in_shape[1]:
                for i in range(0, self.in_shape[2], self.stride_h):
                    for j in range(0, self.in_shape[3], self.stride_w):
                        output[n, o, i // self.stride_h, j // self.stride_w] = np.mean(X[n, o, i:i+self.stride_h, j:j+self.stride_w])
        return output

    def backward(self, eta):
        return np.repeat(np.repeat(eta, self.stride_h, axis=2), self.stride_w, axis=3) / (self.kernel_height * self.kernel_width)
