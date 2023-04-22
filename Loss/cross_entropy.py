from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
from C_Graph.operator import Operator
import numpy as np


class CrossEntropy(Operator):
    def __init__(self, input_variable: [Variable], name: str):
        if not isinstance(input_variable, list):  # requires two inputs
            raise Exception("Operator CrossEntropy name: %s's input_variable is not a list of Variable (it requires two inputs)" % self.name)
        self.input_variable = input_variable  # [y_pred, label]
        self.output_variable = Variable([1], name='loss', scope=name, init='None')
        super(CrossEntropy, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self):
        """
        :input:
        self.input_variable[0].data: y_pred, with shape [N, M],
        which indicates that there are N samples, and each contains M(classes) probabilities.
        self.input_variable[1].data: label, with shape [N, M],
        which indicates that there are N samples, and each contains an M-dimensional one-hot label.

        :return: cross entropy cost result, which is a scalar.
        """
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            y_pred = self.input_variable[0].data
            label = self.input_variable[1].data
            self.output_variable.data = -1 * np.mean(np.log(y_pred[label == 1]))
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        """
        :input:
        self.output_variable.data, a.k.a. cost, with shape (1,)

        :return: self.input_variable[0].diff, a.k.a. differential of y_pred, with shape [N, M]
        """
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            y_pred = self.input_variable[0].data
            label = self.input_variable[1].data
            self.input_variable[0].diff[label == 1] = -1 * (1 / y_pred[label == 1])
            self.wait_forward = True
            return
