import numpy as np
from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
from C_Graph.operator import Operator


class MaxPooling(Operator):
    def __init__(self, kernel_shape:list, input_variable: Variable, name: str, stride=1, padding=0):
        for i in kernel_shape:
            if not isinstance(i, int):
                raise Exception("Operator MaxPooling name: %s kernel shape is not list of int" % self.name)

        if not isinstance(input_variable, Variable):
            raise Exception("Operator MaxPooling name: %s's input_variable is not instance of Variable" % name)

        if len(input_variable.shape) != 4:
            raise Exception("Operator MaxPooling name: %s's input_variable's shape != 4d Variable!" % self.name)

        self.in_shape = input_variable.shape
        self.batch_size = self.in_shape[0]
        self.in_channel = self.in_shape[1]
        self.X_h = self.in_shape[2]
        self.X_w = self.in_shape[3]
        self.kernel_height = kernel_shape[2]
        self.kernel_width = kernel_shape[3]

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
                        self.in_channel,  # in_channel == out_channel
                        (self.X_h - self.kernel_height + 2 * self.padding_h) // self.stride_h + 1,
                        (self.X_w - self.kernel_width + 2 * self.padding_w) // self.stride_w + 1]
        self.output_variable = Variable(output_shape, scope=name, name='output', grad=True, learnable=False)
        self.input_variable = input_variable
        super(MaxPooling, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self._pool()
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            for n in range(self.batch_size):
                for o in range(self.in_channel):
                    for i in range(0, self.X_h-self.kernel_height+1, self.stride_h):
                        for j in range(0, self.X_w-self.kernel_width+1, self.stride_w):
                            self.input_variable.diff[n, o, i:i+self.kernel_height, j:j+self.kernel_width] += \
                                self.output_variable.diff[n, o, i // self.stride_h, j // self.stride_w] * \
                                self.max_index[n, o, i:i+self.kernel_height, j:j+self.kernel_width]
            self.wait_forward = True
            return

    def _pool(self):
        """
                Forward pass for maxPool layer
                :param X: [N, C, H, W] N=Batch_size; C=channels; H=height; W=width
                :return: the output of this batch, shape = [N,
                                                            out_channel,
                                                            (X_h - self.kernel_height + 2 * self.padding_h) / self.stride_h + 1,
                                                            (X_w - self.kernel_width + 2 * self.padding_w) / self.stride_w + 1]
        """

        self.max_index = np.zeros(self.in_shape)  # using as a mask to record the location of the maximum element
        output = np.zeros(self.output_variable.shape)
        for n in range(self.batch_size):
            for o in range(self.in_channel):
                for i in range(0, self.X_h-self.kernel_height+1, self.stride_h):
                    for j in range(0, self.X_w-self.kernel_width+1, self.stride_w):
                        output[n, o, i // self.stride_h, j // self.stride_w] = np.max(
                            self.input_variable.data[n, o, i:i + self.kernel_height, j:j + self.kernel_width])
                        max_index = np.argmax(self.input_variable.data[n, o, i:i + self.kernel_height, j:j + self.kernel_width])
                        self.max_index[n, o, i + max_index // self.kernel_height, j + max_index % self.kernel_width] = 1
        self.output_variable.data = output
        return

class AvgPooling(Operator):
    def __init__(self, kernel_shape:list, input_variable: Variable, name: str, stride=1, padding=0):
        for i in kernel_shape:
            if not isinstance(i, int):
                raise Exception("Operator MaxPooling name: %s kernel shape is not list of int" % self.name)

        if not isinstance(input_variable, Variable):
            raise Exception("Operator MaxPooling name: %s's input_variable is not instance of Variable" % name)

        if len(input_variable.shape) != 4:
            raise Exception("Operator MaxPooling name: %s's input_variable's shape != 4d Variable!" % self.name)

        self.in_shape = input_variable.shape
        self.batch_size = self.in_shape[0]
        self.in_channel = self.in_shape[1]
        self.X_h = self.in_shape[2]
        self.X_w = self.in_shape[3]
        self.kernel_height = kernel_shape[2]
        self.kernel_width = kernel_shape[3]

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
                        self.in_channel,  # in_channel == out_channel
                        (self.X_h - self.kernel_height + 2 * self.padding_h) // self.stride_h + 1,
                        (self.X_w - self.kernel_width + 2 * self.padding_w) // self.stride_w + 1]
        self.output_variable = Variable(output_shape, scope=name, name='output', grad=True, learnable=False)
        self.input_variable = input_variable
        super(AvgPooling, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self._pool()
            self.wait_forward = False
        else:
            pass


    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            scale = 1. / (self.kernel_width * self.kernel_height)
            for n in range(self.batch_size):
                for o in range(self.in_channel):
                    for i in range(0, self.X_h-self.kernel_height+1, self.stride_h):
                        for j in range(0, self.X_w-self.kernel_width+1, self.stride_w):
                            self.input_variable.diff[n, o, i:i+self.kernel_height, j:j+self.kernel_width] += \
                            self.output_variable.diff[n, o, i//self.kernel_height, j//self.kernel_width] * scale
            self.wait_forward = True
        return


    def _pool(self):
        output = np.zeros(self.output_variable.shape)
        for n in range(self.batch_size):
            for o in range(self.in_channel):  # in_channel == out_channel
                for i in range(0, self.X_h-self.kernel_height+1, self.stride_h):
                    for j in range(0, self.X_w-self.kernel_width+1, self.stride_w):
                        output[n, o, i // self.stride_h, j // self.stride_w] = np.mean(
                            self.input_variable.data[n, o, i:i + self.kernel_height, j:j + self.kernel_width])
        self.output_variable.data = output
        return

