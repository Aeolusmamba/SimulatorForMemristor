import numpy as np
import os
import sys
import pandas as pd
from Layer.conv import Conv2D
from Layer.linear import Linear
from C_Graph.operator import Operator
from C_Graph.variable import Variable
from Layer.activation import *
from Layer.pooling import *


class Model():
    def __init__(self, file_path, input: list, out_diff: np.array, batch_size: int):
        if not os.path.exists(file_path):
            print("ERROR: File {} does not exist".format(file_path))
            sys.exit(1)
        df = pd.read_csv(file_path)  # dataframe
        self.input = input
        self.out_diff = out_diff
        self.conv_list = []
        self.linear_list = []
        max_flag = False
        average_flag = False
        linear_cnt = 0
        conv_cnt = 0
        mp_cnt = 0
        ap_cnt = 0
        for i, row in df.iterrows():
            # for each layer i
            if row['kernel height'] == 1 and row['kernel width'] == 1:  # which is a linear layer
                in_dim = row['IFM channel depth']
                out_dim = row['kernel depth']
                input = Variable([batch_size, in_dim], name='input', init='None')
                linear = Linear(input, name='Linear_' + str(linear_cnt), out_dim=out_dim, bias=False)
                self.linear_list.append(linear)

                if row['activation'] == 'relu':
                    act_input = Variable([batch_size, out_dim], name='input', init='None')
                    activation = ReLU(act_input, name='relu')
                elif row['activation'] == 'lrelu':
                    act_input = Variable([batch_size, out_dim], name='input', init='None')
                    activation = LReLU(act_input, name='lrelu')
                elif row['activation'] == 'elu':
                    act_input = Variable([batch_size, out_dim], name='input', init='None')
                    activation = ELU(act_input, name='elu')
                elif row['activation'] == 'prelu':
                    act_input = Variable([batch_size, out_dim], name='input', init='None')
                    activation = PReLU(act_input, name='prelu')
                elif row['activation'] == 'sigmoid':
                    act_input = Variable([batch_size, out_dim], name='input', init='None')
                    activation = Sigmoid(act_input, name='sigmoid')
                elif row['activation'] == 'tanh':
                    act_input = Variable([batch_size, out_dim], name='input', init='None')
                    activation = Tanh(act_input, name='tanh')
                elif row['activation'] == 'softmax':
                    act_input = Variable([batch_size, out_dim], name='input', init='None')
                    activation = Softmax(act_input, name='softmax')
                else:
                    activation = None
                if activation:
                    self.linear_list.append(activation)
                linear_cnt += 1
            elif max_flag:
                in_shape = [batch_size, row['IFM channel depth'], row['IFM height'], row['IFM width']]
                kernel_shape = [row['kernel depth'], row['IFM channel depth'], row['kernel height'], row['kernel width']]
                stride = row['stride']
                padding = row['padding']
                input = Variable(in_shape, name='input', init='None')
                maxPool = MaxPooling(kernel_shape, input, name='MaxPool_'+str(mp_cnt), stride=stride, padding=padding)
                self.conv_list.append(maxPool)
                mp_cnt += 1
                max_flag = False
            elif average_flag:
                in_shape = [batch_size, row['IFM channel depth'], row['IFM height'], row['IFM width']]
                kernel_shape = [row['kernel depth'], row['IFM channel depth'], row['kernel height'],
                                row['kernel width']]
                stride = row['stride']
                padding = row['padding']
                input = Variable(in_shape, name='input', init='None')
                avgPool = AvgPooling(kernel_shape, input, name='AvgPool_' + str(ap_cnt), stride=stride, padding=padding)
                self.conv_list.append(avgPool)
                ap_cnt += 1
                average_flag = False
            else:  # which is a conv layer
                in_shape = [batch_size, row['IFM channel depth'], row['IFM height'], row['IFM width']]
                kernel_shape = [row['kernel depth'], row['IFM channel depth'], row['kernel height'],
                                row['kernel width']]
                stride = row['stride']
                padding = row['padding']
                input = Variable(in_shape, name='input', init='None')
                conv = Conv2D(kernel_shape, input, name='conv_' + str(conv_cnt), stride=stride, padding=padding)
                self.conv_list.append(conv)

                output_shape = [batch_size,
                                kernel_shape[0],
                                (in_shape[2] - kernel_shape[2] + 2 * padding) // stride + 1,
                                (in_shape[3] - kernel_shape[3] + 2 * padding) // stride + 1]
                if row['activation'] == 'relu':
                    act_input = Variable(output_shape, name='input', init='None')
                    activation = ReLU(act_input, name='relu')
                elif row['activation'] == 'lrelu':
                    act_input = Variable(output_shape, name='input', init='None')
                    activation = LReLU(act_input, name='lrelu')
                elif row['activation'] == 'elu':
                    act_input = Variable(output_shape, name='input', init='None')
                    activation = ELU(act_input, name='elu')
                elif row['activation'] == 'prelu':
                    act_input = Variable(output_shape, name='input', init='None')
                    activation = PReLU(act_input, name='prelu')
                elif row['activation'] == 'sigmoid':
                    act_input = Variable(output_shape, name='input', init='None')
                    activation = Sigmoid(act_input, name='sigmoid')
                elif row['activation'] == 'tanh':
                    act_input = Variable(output_shape, name='input', init='None')
                    activation = Tanh(act_input, name='tanh')
                elif row['activation'] == 'softmax':
                    act_input = Variable(output_shape, name='input', init='None')
                    activation = Softmax(act_input, name='softmax')
                else:
                    activation = None
                if activation:
                    self.conv_list.append(activation)
                conv_cnt += 1


    def forward_pass(self):
        # assign the real input to the first layer's input_variable.data
        self.conv_list[0].input_variable.data = self.input
        for conv in self.conv_list:
            conv.forward()
        for linear in self.linear_list:
            linear.forward()

    def backPropagation(self):
        # assign the real out_diff to the last layer's output_variable.diff
        self.linear_list[-1].output_variable.diff = self.out_diff
        for linear in reversed(self.linear_list):
            linear.backward()
        for conv in reversed(self.conv_list):
            conv.backward()
