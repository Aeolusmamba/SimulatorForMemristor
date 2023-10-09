from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
from C_Graph.operator import Operator
import numpy as np


class ConvDataConverter:

    def __init__(self):
        pass

    def im2col(self, input_variable: Variable, W_shape: list, stride: list, padding: list):
        '''
        :param input_variable: input variable with size (N, in_channel, H, W)
        :param W_shape:  weight size (out_channel, in_channel, K_h, K_w)
        :return:
        '''
        col_X = []
        out_channel, in_channel, K_h, K_w = W_shape
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

    def im2col_rep(self, input_variable: Variable, W_shape: list, stride: list, padding: list, replica_num):
        col_X = []  # 变形后的X
        out_channel, in_channel, K_h, K_w = W_shape
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
            col_temp = []  # X变形以后的一个样本
            col_ext = []  # X变形以后的一个样本中的一个列（向量）
            cnt = replica_num
            for i in range(0, X.shape[2] - K_h + 1, stride_h):
                for j in range(0, X.shape[3] - K_w + 1, stride_w):
                    col = X[n, :, i:i + K_h, j:j + K_w].reshape(-1)
                    col_ext.append(col)
                    cnt -= 1
                    if cnt == 0:
                        col_temp.append(np.reshape(col_ext, (-1)))
                        cnt = replica_num
                        col_ext = []
            col_X.append(col_temp)

        col_X = np.array(col_X)
        # print("col_X shape: ", col_X.shape)
        return col_X

    def im2col_overlap(self, input_variable: Variable, W_shape: list, stride: list, padding: list, replica_num):
        col_X = []
        out_channel, in_channel, K_h, K_w = W_shape
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
            col_temp = []  # X变形以后的一个样本
            col_ext = []  # X变形以后的一个样本中的一个列（向量）
            cnt = replica_num
            for i in range(0, X.shape[2] - K_h + 1, stride_h):

                col = X[n, :, i : i + K_h, 0 : K_w].reshape(-1)
                col_ext.append(col)
                cnt -= 1
                if stride_w > X.shape[3] - K_w:
                    # 如果横向只能走一步，直接跳到下一行
                    col_temp.append(np.reshape(col_ext, (-1)))
                    cnt = replica_num
                    col_ext = []
                    continue

                for j in range(stride_w, X.shape[3] - K_w + 1, stride_w):  # 从第二步开始
                    col = X[n, :, i : i + K_h, j - stride_w + K_w : j + K_w]  # offset
                    col_ext.append(col)
                    cnt -= 1
                    if j == X.shape[3] - K_w:
                        # 到达该行的最后，需要另外形成列向量，（理想情况：X的完整一行形成一列）
                        col_temp.append(np.reshape(col_ext, (-1)))
                        cnt = replica_num
                        col_ext = []
                    elif cnt == 0 and j == X.shape[3] - K_w - stride_w:
                        # 还未到达该行的最后，还差一步走到最后，但复制的权重矩阵不够用了：走完最后一步然后跳到下一行
                        col_temp.append(np.reshape(col_ext, (-1)))  # 前面的形成一列
                        col_ext = []
                        col_temp.append(X[n, :, i : i + K_h, j : j + K_w].reshape(-1))  # 最后一步单独形成一列
                        cnt = replica_num
                        break  # 跳到下一行
                    elif cnt == 0:
                        # 还未到达该行的最后，也还未走到倒数第二步，但复制的权重矩阵不够用了：把剩下的走完
                        col_temp.append(np.reshape(col_ext, (-1)))  # 前面的形成一列
                        col_ext = []
                        cnt = replica_num
                        col_temp.append(X[n, :, i : i + K_h, j : j + K_w].reshape(-1))
                        j += stride_w  # 手动走一步
                        cnt -= 1
            col_X.append(col_temp)
        col_X = np.array(col_X)
        return col_X

    def im2col_block(self, input_variable: Variable, W_shape: list, stride: list, padding: list, XBar_hshape: list, XBar_wshape: list):
        col_X = []
        out_channel, in_channel, K_h, K_w = W_shape
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
                    idx = 0
                    # blocking
                    for k in XBar_hshape:
                        for w in range(len(XBar_wshape)):
                            if idx + k > len(col):
                                raise Exception("Blocking failed! Out of range error.")
                            col_temp.append(col[idx: idx + k])  # 列分块后，col[idx: idx + k]重复添加
                        idx += k

            col_X.append(col_temp)
        col_X = np.array(col_X)
        return col_X

    # regular im2col
    def getConvertedData(self, input_variable: Variable, W_shape: list, stride: list, padding: list):
        converted_data = self.im2col(input_variable, W_shape, stride, padding)
        converted_data_var = Variable(list(converted_data.shape), scope="compiler", name="converted_" + input_variable.name)
        converted_data_var.data = converted_data
        return converted_data_var

    # 非重叠复制的im2col
    def getConvertedDataRep(self, input_variable: Variable, W_shape: list, stride: list, padding: list, replica_num):
        converted_data = self.im2col_rep(input_variable, W_shape, stride, padding, replica_num)
        converted_data_var = Variable(list(converted_data.shape), scope="compiler", name="converted_" + input_variable.name)
        converted_data_var.data = converted_data
        return converted_data_var

    # 重叠复制的im2col
    def getConvertedDataOverlap(self, input_variable: Variable, W_shape: list, stride: list, padding: list, replica_num):

        if replica_num == 1:
            converted_data = self.im2col(input_variable, W_shape, stride, padding)
        elif replica_num > 1:
            converted_data = self.im2col_overlap(input_variable, W_shape, stride, padding, replica_num)
        else:
            raise Exception("Something unexpected happened.")
        converted_data_var = Variable(list(converted_data.shape), scope="compiler",
                                      name="converted_" + input_variable.name)
        converted_data_var.data = converted_data
        return converted_data_var

    # 分块的im2col
    def getConvertedDataBlock(self, input_variable: Variable, W_shape: list, stride: list, padding: list, XBar_hshape: list, XBar_wshape: list):
        converted_data = self.im2col_block(input_variable, W_shape, stride, padding, XBar_hshape, XBar_wshape)
        converted_data_var = Variable(list(converted_data.shape), scope="compiler",
                                      name="converted_" + input_variable.name)
        converted_data_var.data = converted_data
        return converted_data_var

class LinearDataConverter:
    def __init__(self):
        pass

    def getConvertedData(self, input_variable: Variable):
        converted_data = np.reshape(input_variable.data, (input_variable.shape[0], -1))
        converted_data_var = Variable(list(converted_data.shape), scope="compiler",
                                      name="converted_" + input_variable.name)
        converted_data_var.data = converted_data
        return converted_data_var

    def getConvertedDataBlock(self, input_variable: Variable, XBar_hshape: list):
        tmp_data = np.reshape(input_variable.data, (input_variable.shape[0], -1))
        converted_data = []
        for n in range(tmp_data.shape[0]):
            idx = 0
            col = []
            for h in XBar_hshape:
                if idx + h > len(tmp_data[n]):
                    raise Exception("Blocking failed! Out of range error.")
                col.append(tmp_data[n, idx : idx + h])
                idx += h
            converted_data.append(col)
        converted_data = np.array(converted_data)
        converted_data_var = Variable(list(converted_data.shape), scope="compiler",
                                      name="converted_" + input_variable.name)
        converted_data_var.data = converted_data
        return converted_data_var


if __name__ == "__main__":
    conv_data_converter = ConvDataConverter()
    X = Variable([64, 1, 28, 28], name="test")
    col_X = conv_data_converter.im2col(X, [128, 1, 3, 3], [1, 1], [1, 1])
    print(col_X.shape)

