from C_Graph.operator import Operator
from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
import numpy as np


class Sigmoid(Operator):
    def __init__(self, input_variable: Variable, name: str):
        self.input_variable = input_variable
        self.output_variable = Variable(self.input_variable.shape, name='output', scope=name, init='None')
        super(Sigmoid, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self, phase):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval(phase)
            # y = 1/(1+exp(-x))
            self.output_variable.data = 1.0 / (1.0 + np.exp(-self.input_variable.data))
            if phase == 'train':
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
            # eta_x = eta_y * y * (1-y)
            self.input_variable.diff = self.output_variable.diff * self.output_variable.data * (
                    1 - self.output_variable.data)
            self.wait_forward = True
            return


class Tanh(Operator):
    def __init__(self, input_variable: Variable, name: str):
        self.input_variable = input_variable
        self.output_variable = Variable(self.input_variable.shape, name='output', scope=name, init='None')
        super(Tanh, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self, phase):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval(phase)
            self.output_variable.data = (1 - np.exp(-2 * self.input_variable.data)) / (
                    1 + np.exp(-2 * self.input_variable.data))
            if phase == 'train':
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
            self.input_variable.diff = self.output_variable.diff * (
                    1 - self.output_variable.data * self.output_variable.data)
            self.wait_forward = True
            return


class ReLU(Operator):
    def __init__(self, input_variable: Variable, name: str):
        self.input_variable = input_variable
        self.output_variable = Variable(self.input_variable.shape, name='output', scope=name, init='None')
        super(ReLU, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self, phase):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval(phase)
            self.output_variable.data = np.maximum(self.input_variable.data, 0)
            if phase == 'train':
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
            self.input_variable.diff = self.output_variable.diff
            self.input_variable.diff[self.input_variable.data < 0] = 0
            self.wait_forward = True
            return


class LReLU(Operator):  # Leaky Relu
    def __init__(self, input_variable: Variable, name: str, alpha=0.01):
        self.input_variable = input_variable
        self.output_variable = Variable(self.input_variable.shape, name='output', scope=name, init='None')
        self.alpha = alpha
        super(LReLU, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self, phase):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval(phase)
            self.output_variable.data = np.maximum(self.input_variable.data, 0) + self.alpha * np.minimum(
                self.input_variable.data, 0)
            if phase == 'train':
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
            self.input_variable.diff = self.output_variable.diff
            self.input_variable.diff[self.input_variable.data < 0] *= self.alpha
            self.wait_forward = True
            return


class ELU(Operator):  # Exponential Linear Unit
    def __init__(self, input_variable: Variable, name: str, alpha=0.1):
        self.input_variable = input_variable
        self.output_variable = Variable(self.input_variable.shape, name='output', scope=name, init='None')
        self.alpha = alpha
        super(ELU, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self, phase):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval(phase)
            self.output_variable.data = np.maximum(self.input_variable.data, 0) + \
                                        self.alpha * (np.exp(np.minimum(self.input_variable.data, 0)) - 1)
            if phase == 'train':
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
            self.input_variable.diff = self.output_variable.diff
            self.input_variable.diff[self.input_variable.data <= 0] *= self.alpha * \
                                                                       np.exp(self.input_variable.data[
                                                                                  self.input_variable.data <= 0])
            self.wait_forward = True
            return


class PReLU(Operator):  # Parametric Rectified Linear Unit
    def __init__(self, input_variable: Variable, name: str, alpha=0.25):
        self.input_variable = input_variable
        self.output_variable = Variable(self.input_variable.shape, name='output', scope=name, init='None')
        self.alpha = alpha
        self.momentum = 0.9
        self.eta = 1e-4
        super(PReLU, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self, phase):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval(phase)
            self.output_variable.data = np.maximum(self.input_variable.data, 0) + self.alpha * np.minimum(
                self.input_variable.data, 0)
            if phase == 'train':
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
            self.alpha = self.momentum * self.alpha + self.eta * np.sum(np.minimum(self.input_variable.data, 0))
            self.input_variable.diff = self.output_variable.diff
            self.input_variable.diff[self.input_variable.data < 0] *= self.alpha
            self.wait_forward = True
            return


if __name__ == "__main__":
    # check grad
    shape = ([10])
    a = Variable(shape, 'a')

    test_layer = ReLU(a, 'test')
    b = test_layer.output_variable

    epsilon = 1e-7

    a.data -= epsilon
    print('a -eps ', a.data)
    out1 = b.eval()

    # refresh graph
    b.wait_bp = False
    a.wait_bp = False
    test_layer.wait_forward = True

    a.data += 2 * epsilon
    print('a +eps ', a.data)
    out2 = b.eval()

    # refresh graph
    b.wait_bp = False
    a.wait_bp = False
    test_layer.wait_forward = True

    a.data -= epsilon
    b.eval()

    b.diff = np.array([1.0] * 10, dtype=float)
    print('bp:        ', a.diff_eval())
    print('grad_check:', (out2 - out1) / 2 / epsilon)
