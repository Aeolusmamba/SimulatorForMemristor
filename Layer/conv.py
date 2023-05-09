import numpy as np
from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
from C_Graph.operator import Operator


class Conv2D(Operator):
    def __init__(self, kernel_shape: list, input_variable: Variable, name: str, hyper_p: dict, stride=1, padding=0, bias=False):
        # kernel_shape: [out_channel, in_channel, kernel_height, kernel_width]
        for i in kernel_shape:
            if not isinstance(i, int):
                raise Exception("Operator Conv2D name: %s's kernel shape is not a list of int" % self.name)

        if not isinstance(input_variable, Variable):
            raise Exception("Operator Conv2D name: %s's input_variable is not an instance of Variable" % self.name)

        if len(input_variable.shape) != 4:
            raise Exception("Operator Conv2D name: %s's input_variable's shape != 4d Variable!" % self.name)

        self.out_channel = kernel_shape[0]
        self.in_channel = kernel_shape[1]
        self.kernel_height = kernel_shape[2]
        self.kernel_width = kernel_shape[3]
        self.bias = bias
        self.in_shape = input_variable.shape  # [N, C, H, W] N=Batch_size; C=channels; H=height; W=width
        self.batch_size = self.in_shape[0]
        self.X_h = self.in_shape[2]
        self.X_w = self.in_shape[3]
        self.hyper_p = hyper_p

        self.weight = Variable(kernel_shape, scope=name, name='weight', grad=True, learnable=True)
        if self.bias:
            self.weight_bias = Variable([self.out_channel], scope=name, name='weight_bias', grad=True, learnable=True)

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


        output_shape = [self.batch_size,
                        self.out_channel,
                        (self.X_h - self.kernel_height + 2 * self.padding_h) // self.stride_h + 1,
                        (self.X_w - self.kernel_width + 2 * self.padding_w) // self.stride_w + 1]
        self.output_variable = Variable(output_shape, scope=name, name='output', grad=True, learnable=False)
        self.input_variable = input_variable
        super(Conv2D, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            if self.bias:
                self._conv(self.input_variable, self.output_variable, self.weight.data, self.weight_bias.data)
            else:
                self._conv(self.input_variable, self.output_variable, self.weight.data)
            self.wait_forward = False
            # print(f"{self.name} output: ", self.output_variable.data)
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            if self.bias:
                self._deconv(self.input_variable, self.output_variable, self.weight, self.weight_bias)
            else:
                self._deconv(self.input_variable, self.output_variable, self.weight)
            self.wait_forward = True
            return

    def _conv(self, input_variable: Variable, output_variable: Variable, weight: np.ndarray, weight_bias=None):
        """
                Using im2col method to forward (refer to: https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf)
                :param X: [N, C, H, W] N=Batch_size; C=channels; H=height; W=width
                :return: the output of this batch, shape = [N,
                                                            out_channel,
                                                            (X_h - self.kernel_height + 2 * self.padding_h) / self.stride_h + 1,
                                                            (X_w - self.kernel_width + 2 * self.padding_w) / self.stride_w + 1]
        """
        if self.padding_w != 0 or self.padding_h != 0:
            X = np.pad(input_variable.data,
                       ((0, 0), (0, 0), (self.padding_h, self.padding_h), (self.padding_w, self.padding_w)),
                       'constant', constant_values=0)
        else:
            X = input_variable.data

        # reshape weight to row_weight: [out_channel, (in_channel * kernel_height * kernel_width)]
        row_weight = weight.reshape(self.out_channel, -1)

        self.col_X = []
        conv_out = np.zeros(output_variable.data.shape)
        for i in range(self.batch_size):
            col_x = self.im2col(X[i])
            # print("shape: ", col_x.shape)
            self.col_X.append(col_x)
            if self.bias:
                conv_out[i] = np.reshape(np.dot(row_weight, col_x.T) + weight_bias[:, np.newaxis],
                                         output_variable.data[0].shape)
            else:
                conv_out[i] = np.reshape(np.dot(row_weight, col_x.T), output_variable.data[0].shape)
        self.col_X = np.array(self.col_X)
        output_variable.data = conv_out
        return

    def _deconv(self, input_variable: Variable, output_variable: Variable, weight: Variable, weight_bias=None):
        # step 1: calculate a(loss) / a(weight)
        eta = output_variable.diff
        col_eta = np.reshape(eta, (self.batch_size, self.out_channel, -1))
        for i in range(self.batch_size):
            weight.diff += np.reshape(np.dot(col_eta[i], self.col_X[i]), self.weight.shape)
        if self.bias:
            # print(f"{self.name}: col_eta", col_eta)
            weight_bias.diff += np.sum(col_eta, axis=(0, 2))

        # step 2: calculate the delta passed to the previous layer
        # print("which layer: ", self.name)
        # print("eta shape: ", eta.shape)
        # print("pad_width: ", (self.stride_h * (eta.shape[2] - 1) + self.kernel_height - self.X_h) // 2)
        pad_eta = np.pad(eta, ((0, 0), (0, 0),
                               ((self.stride_h * (eta.shape[2] - 1) + self.kernel_height - self.X_h) // 2,
                                (self.stride_h * (eta.shape[2] - 1) + self.kernel_height - self.X_h) // 2),
                               ((self.stride_w * (eta.shape[3] - 1) + self.kernel_width - self.X_w) // 2,
                                (self.stride_w * (eta.shape[3] - 1) + self.kernel_width - self.X_w) // 2)),
                         'constant', constant_values=0)
        col_pad_eta = np.array([self.im2col(pad_eta[i]) for i in range(self.batch_size)])
        # rotate weights 180 degree
        flipped_weight = self.rotate180(weight)
        # transpose the weight
        flipped_weight = flipped_weight.swapaxes(0, 1)  # change positions of out_channel and in_channel
        col_flip_weight = flipped_weight.reshape((self.in_channel, -1))
        # if self.name == 'conv_5':
        #     print("col_flip_weight: ", col_flip_weight)

        next_eta = []
        for i in range(self.batch_size):
            next_eta_i = np.reshape(np.dot(col_flip_weight, col_pad_eta[i].T), (self.in_channel, self.X_h, self.X_w))
            next_eta.append(next_eta_i)
        next_eta = np.array(next_eta)
        # if self.name == 'conv_5':
        #     print("next_eta: ", next_eta)
        input_variable.diff = next_eta
        return next_eta

    def im2col(self, x):
        """
        convert an image to columns
        :param x: with shape [in_channel, H, W] (H and W may include padding)
        :return col_x: with shape [((H-kernel_height) / stride_h + 1) * ((W-kernel_width) / stride_w + 1), in_channel * kernel_height * kernel_width]
        """
        col_x = []
        for i in range(0, x.shape[1]-self.kernel_height+1, self.stride_h):
            for j in range(0, x.shape[2]-self.kernel_width+1, self.stride_w):
                col = x[:, i:i + self.kernel_height, j:j + self.kernel_width].reshape(
                    -1)  # C * kernel_height * kernel_width
                # if self.name == 'conv_5':
                #     print("x[:, i:i + self.kernel_height, j:j + self.kernel_width].shape", x[:, i:i + self.kernel_height, j:j + self.kernel_width].shape)
                col_x.append(col)
        col_x = np.array(col_x)
        return col_x

    def rotate180(self, weight):
        rot_weight = np.zeros(weight.shape)
        for out_channel in range(weight.shape[0]):
            for in_channel in range(weight.shape[1]):
                rot_weight[out_channel, in_channel, :, :] = np.flipud(
                    np.fliplr(weight.data[out_channel, in_channel, :, :]))
        return rot_weight


if __name__ == "__main__":
    # img = np.random.standard_normal((2, 32, 32, 3))
    img = np.ones((1, 32, 32, 3))
    img *= 2
    # conv = Conv2D(img.shape, 12, 3, 1)
    # next = conv.forward(img)
    # next1 = next.copy() + 1
    # conv.gradient(next1 - next)
    # print(conv.w_gradient)
    # print(conv.b_gradient)
    # conv.backward()
