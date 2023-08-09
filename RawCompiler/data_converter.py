from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
from C_Graph.operator import Operator
import numpy as np


class DataConverter:

    def __init__(self):
        pass

    def im2col(self, input_variable: Variable, W_size: list, stride: list, padding: list):
        '''
        :param input_variable: input variable with size (N, in_channel, H, W)
        :param W_size:  weight size (out_channel, in_channel, K_h, K_w)
        :return:
        '''
        col_X = []
        out_channel, in_channel, K_h, K_w = W_size
        if stride and isinstance(stride, tuple):
            stride_h, stride_w = stride
        elif stride and isinstance(stride, int) and stride >= 1:
            stride_w = stride_h = stride
        else:
            stride_w = stride_h = 1

        if padding and isinstance(padding, tuple):
            padding_h, padding_w = padding
        elif padding and isinstance(padding, int) and padding >= 0:
            padding_w = padding_h = padding
        else:
            padding_h = padding_w = 0

        if padding_w != 0 or padding_h != 0:
            X = np.pad(input_variable.data,
                       ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)),
                       'constant', constant_values=0)
        else:
            X = input_variable.data

        for n in range(X.shape[0]):
            col_temp = []
            for i in range(0, X.shape[2] - K_h + 1, stride_h):
                for j in range(0, X.shape[3] - K_w + 1, stride_w):
                    col = X[n, :, i:i + K_h, j:j + K_w].reshape(-1)  # C * kernel_height * kernel_width
                    col_temp.append(col)
            col_X.append(col_temp)
        col_X = np.array(col_X)
        return col_X

    def getConvertedData(self, input_variable: Variable, W_size: list, stride: list, padding: list):
        converted_data = self.im2col(input_variable, W_size, stride, padding)
        converted_data_var = Variable(list(converted_data.shape), scope="compiler", name="converted_" + input_variable.name)
        converted_data_var.data = converted_data
        return converted_data_var


if __name__ == "__main__":
    data_converter = DataConverter()
    X = Variable([64, 1, 28, 28], name="test")
    col_X = data_converter.im2col(X, [128, 1, 3, 3], [1, 1], [1, 1])
    print(col_X.shape)

