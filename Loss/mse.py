from C_Graph.operator import Operator
from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
import numpy as np
import functools


class MSE(Operator):
    def __init__(self, input_variable: [Variable], name: str):
        if not isinstance(input_variable, list):  # requires two inputs
            raise Exception("Operator name: %s shape is not list of Variable (it requires two inputs)" % self.name)
        self.input_variable = input_variable  # [y_pred, y_truth]
        self.output_variable = Variable([1], name='loss', scope=name, init='None')
        super(MSE, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self, phase):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval(phase)
            y_pred = self.input_variable[0].data
            y_truth = self.input_variable[1].data
            self.output_variable.data = 0.5 * np.mean((y_pred - y_truth) ** 2)
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
            y_pred = self.input_variable[0].data
            y_truth = self.input_variable[1].data
            self.input_variable[0].diff = (y_pred - y_truth) / functools.reduce(lambda x, y: x * y, y_pred.shape)
            self.wait_forward = True
            return

if __name__ == "__main__":
    # check grad
    shape = ([1, 10])
    a = Variable(shape, 'a')
    label = Variable(shape, 'label')
    label.data = np.array([[0,0,1,0,0,0,0,0,0,0]])

    test_layer = MSE([a, label], 'test')
    b = test_layer.output_variable

    epsilon = 1e-7

    a.data[0][0] -= epsilon
    print('a[0] -eps ', a.data[0][0])
    out1 = b.eval()

    # refresh graph
    b.wait_bp = False
    a.wait_bp = False
    test_layer.wait_forward = True

    a.data[0][0] += 2 * epsilon
    print('a[0] +eps ', a.data[0][0])
    out2 = b.eval()

    # refresh graph
    b.wait_bp = False
    a.wait_bp = False
    test_layer.wait_forward = True
    a.data[0][0] -= epsilon
    b.eval()
    b.diff = np.array([[1.0]], dtype=float)
    print('bp:        ', a.diff_eval())
    print('grad_check:', (out2 - out1) / 2 / epsilon)
