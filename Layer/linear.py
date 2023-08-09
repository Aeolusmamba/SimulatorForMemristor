import numpy as np
from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
from C_Graph.operator import Operator


class Linear(Operator):
    def __init__(self, input_variable: Variable, name: str, out_dim: int, hyper_p: dict, bias=False):
        if not isinstance(input_variable, Variable):
            raise Exception("Operator Linear name: %s's input_variable is not an instance of Variable" % self.name)
        self.input_variable = input_variable
        self.out_dim = out_dim
        self.hyper_p = hyper_p
        self.bias = bias
        self.weight = Variable([out_dim, self.input_variable.shape[1]], name='weight', scope=name, grad=True,
                               learnable=True, init='const')
        if self.bias:
            self.weight_bias = Variable([out_dim], name='weight_bias', scope=name, grad=True, learnable=True,
                                        init='const')

        self.output_variable = Variable([self.input_variable.shape[0], self.out_dim], name='output', scope=name)
        super(Linear, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self, phase):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval(phase)
            self.output_variable.data = np.dot(self.input_variable.data, self.weight.data.T)
            if self.bias:
                self.output_variable.data += self.weight_bias.data
            if phase == 'train':
                self.wait_forward = False
            # print(f"{self.name} output: ", self.output_variable.data)
            return
        else:
            pass

    def backward(self):
        """
        :input: self.output_variable.diff -> dy: the delta of this layer
        :return: dx: self.input_variable.diff -> the delta of the previous layer (to compute)
        """
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            # calculate the gradient of weight (and bias)
            self.weight.diff += np.dot(self.output_variable.diff.T,
                                       self.input_variable.data)  # += can be used for gradient accumulation
            if self.bias:
                self.weight_bias.diff += np.sum(self.output_variable.diff, axis=0)

            # compute the delta of previous layer
            self.input_variable.diff += np.dot(self.output_variable.diff, self.weight.data)
            self.wait_forward = True
            return


if __name__ == "__main__":
    # check grad
    shape = ([1, 10])
    a = Variable(shape, 'a')

    test_layer = Linear(a, 'test', out_dim=1, hyper_p={})
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
