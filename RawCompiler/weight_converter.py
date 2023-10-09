from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
from C_Graph.operator import Operator
import numpy as np


class WeightConverter:

    def __init__(self):
        pass

    def getConvertedWeight(self, W: Variable, base_size) -> [Variable]:  # base_size=256 for now
        # convert to IPU-containable lists.
        converted_weight = self.convertWeight(W)
        return converted_weight

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
