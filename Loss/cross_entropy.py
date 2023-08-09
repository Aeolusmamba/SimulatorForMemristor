from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
from C_Graph.operator import Operator
import numpy as np

# softmax with Loss
class CrossEntropy(Operator):
    def __init__(self, input_variable: [Variable], name: str):
        if not isinstance(input_variable, list):  # requires two inputs
            raise Exception("Operator CrossEntropy name: %s's input_variable is not a list of Variable (it requires two inputs)" % self.name)
        self.input_variable = input_variable  # [y_pred, label]
        self.output_variable = Variable([1], name='loss', scope=name, init='None')
        super(CrossEntropy, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self, phase):
        """
        :input:
        self.input_variable[0].data: y_pred, with shape [N, M],
        which indicates that there are N samples, and each contains M (classes) "scores" (to be put into softmax first).
        self.input_variable[1].data: label, with shape [N, M],
        which indicates that there are N samples, and each contains an M-dimensional one-hot label.

        :return: cross entropy cost result, which is a scalar.
        """
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval(phase)
            y_scores = self.input_variable[0].data
            label = self.input_variable[1].data
            # softmax
            denominator = np.sum(np.exp(y_scores - np.max(y_scores, axis=1, keepdims=True)), axis=1,
                                 keepdims=True)
            self.y_prob = np.exp(
                y_scores - np.max(y_scores, axis=1, keepdims=True)) / denominator
            # cross entropy loss
            epsilon = 1e-8
            self.output_variable.data = -1 * np.mean(np.log(self.y_prob[label == 1] + epsilon))
            self.wait_forward = False
            return
        else:
            pass

    def backward(self):
        """
        :input:
        self.output_variable.data, a.k.a. cost, with shape (1,)

        :return: self.input_variable[0].diff, a.k.a. differential of y_scores, with shape [N, M]
        """
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            label = self.input_variable[1].data
            self.input_variable[0].diff[label == 1] = -1 * (1 - self.y_prob[label == 1])
            self.input_variable[0].diff[label != 1] = self.y_prob[label != 1]
            # print("output diff: ", self.input_variable[0].diff)
            # print("self.input_variable[0].diff.shape: ", self.input_variable[0].diff.shape)
            self.wait_forward = True
            return

if __name__ == "__main__":
    # check grad
    shape = ([1, 10])
    a = Variable(shape, 'a')
    label = Variable(shape, 'label')
    label.data = np.array([[0,0,1,0,0,0,0,0,0,0]])

    test_layer = CrossEntropy([a, label], 'test')
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
