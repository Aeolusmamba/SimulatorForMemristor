from C_Graph.operator import Operator
from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
import numpy as np


class MSE(Operator):
    def __init__(self, input_variable: [Variable], name: str):
        if not isinstance(input_variable, list):  # requires two inputs
            raise Exception("Operator name: %s shape is not list of Variable (it requires two inputs)" % self.name)
        self.input_variable = input_variable  # [y_pred, y_truth]
        self.output_variable = Variable([1], name='loss', scope=name)
        super(MSE, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
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
            self.input_variable[0].diff = np.mean(y_pred - y_truth)
            self.wait_forward = True
            return
