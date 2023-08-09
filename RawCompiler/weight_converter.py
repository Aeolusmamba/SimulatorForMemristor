from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
from C_Graph.operator import Operator
import numpy as np


class WeightConverter:

    def __init__(self):
        pass

    def getConvertedWeight(self, W: Variable, base_size) -> [Variable]:  # base_size=256 for now
        # convert to IPU-containable lists.
        converted_weight = self.convertWeight(W)
        # check its size
        if converted_weight.shape[0] > base_size and converted_weight.shape[1] <= base_size:
            rows = converted_weight.shape[0]
            sliced_weights = slice_in_row_major(converted_weight.data, rows, base_size)
            sliced_weights_var = []
            i = 0
            for w in sliced_weights:
                w_var = Variable(list(w.shape), scope=W.name + " in compiler", name="sliced_converted_w"+str(i))
                w_var.data = w
                sliced_weights_var.append(w_var)
                i += 1
            return sliced_weights_var
        elif converted_weight.shape[0] <= base_size and converted_weight.shape[1] > base_size:
            cols = converted_weight.shape[1]
            sliced_weights = slice_in_col_major(converted_weight.data, cols, base_size)
            sliced_weights_var = []
            i = 0
            for w in sliced_weights:
                w_var = Variable(list(w.shape), scope=W.name + " in compiler", name="sliced_converted_w" + str(i))
                w_var.data = w
                sliced_weights_var.append(w_var)
                i += 1
            return sliced_weights_var
        elif converted_weight.shape[0] > base_size and converted_weight.shape[1] > base_size:
            rows = converted_weight.shape[0]
            cols = converted_weight.shape[1]
            row_sliced_weights = slice_in_row_major(converted_weight.data, rows, base_size)
            sliced_weights = []  # tow dims
            for w in row_sliced_weights:
                col_sliced_weights = slice_in_col_major(w, cols, base_size)
                sliced_weights.append(col_sliced_weights)
            sliced_weights_var = []
            index = 0
            for i in sliced_weights:
                for w in i:
                    w_var = Variable(list(w.shape), scope="compiler", name="sliced_" + W.name)
                    w_var.data = w
                    sliced_weights_var.append(w_var)
                    index += 1
            return sliced_weights_var
        return [converted_weight]

    def convertWeight(self, W: Variable):
        c_W = np.reshape(W.data, (W.shape[0], -1))  # reshape weight to row_weight: [out_channel, (in_channel * kernel_height * kernel_width)]
        c_W = np.swapaxes(c_W, 0, 1)  # swap axes to load in memory
        converted_W = Variable(list(c_W.shape), scope="compiler", name="converted_" + W.name)
        converted_W.data = c_W
        return converted_W

def slice_in_row_major(W, rows, base) -> list:
    slice_num = rows // base
    rows_temp = rows
    if base * slice_num < rows:
        slice_num += 1
    sliced_weights = []
    for i in range(slice_num):
        sliced_weight = W[i*base : ((i+1)*base if rows_temp>base else rows), :]
        rows_temp -= base
        sliced_weights.append(sliced_weight)
    return sliced_weights

def slice_in_col_major(W, cols, base) -> list:
    slice_num = cols // base
    cols_temp = cols
    if base * slice_num < cols:
        slice_num += 1
    sliced_weights = []
    for i in range(slice_num):
        sliced_weight = W[:, i*base : ((i+1)*base if cols_temp>base else cols)]
        cols_temp -= base
        sliced_weights.append(sliced_weight)
    return sliced_weights
