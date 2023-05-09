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
    def __init__(self, file_path, batch_size: int, hyper_p: dict):
        if not os.path.exists(file_path):
            print("ERROR: File {} does not exist".format(file_path))
            sys.exit(1)
        df = pd.read_csv(file_path)  # dataframe
        self.conv_list = []
        self.linear_list = []
        self.parameters = []
        self.batch_size = batch_size
        self.hyper_p = hyper_p
        max_flag = False
        average_flag = False
        linear_begin = True
        linear_cnt = 0
        conv_cnt = 0
        mp_cnt = 0
        ap_cnt = 0
        # temporary in & out variable
        in_out = Variable([batch_size, int(df.loc[0]['IFM channel depth']), int(df.loc[0]['IFM height']), int(df.loc[0]['IFM width'])],
                          name='input', scope='conv_0', init='None')
        for i, row in df.iterrows():
            # for each layer i
            if row['kernel height'] == 1 and row['kernel width'] == 1:  # which is a linear layer
                in_dim = row['IFM channel depth']
                out_dim = row['kernel depth']
                name = 'Linear_' + str(linear_cnt)
                if linear_begin:
                    in_out = Variable([batch_size, in_dim], name='input', scope=name, init='None')
                    linear_begin = False
                if row['bias'] == 1:
                    bias = True
                else:
                    bias = False
                linear = Linear(input_variable=in_out, name=name, out_dim=out_dim, bias=bias, hyper_p=hyper_p)
                in_out = linear.output_variable
                self.parameters.append(linear.weight)
                if bias:
                    self.parameters.append(linear.weight_bias)
                self.linear_list.append(linear)

                # optimizer
                if not pd.isna(row['optimizer']) and row['optimizer'].lower() == 'adam':
                    linear.weight.set_method_adam(beta1=hyper_p['beta1'], beta2=hyper_p['beta2'],
                                                epsilon=hyper_p['epsilon'])
                    if bias:
                        linear.weight_bias.set_method_adam(beta1=hyper_p['beta1'], beta2=hyper_p['beta2'],
                                                         epsilon=hyper_p['epsilon'])
                elif not pd.isna(row['optimizer']) and row['optimizer'].lower() == 'nga':
                    linear.weight.set_method_nga(momentum=hyper_p['momentum'])
                    if bias:
                        linear.weight_bias.set_method_nga(momentum=hyper_p['momentum'])
                elif not pd.isna(row['optimizer']) and row['optimizer'].lower() == 'momentum':
                    linear.weight.set_method_momentum(momentum=hyper_p['momentum'])
                    if bias:
                        linear.weight_bias.set_method_momentum(momentum=hyper_p['momentum'])
                else:  # default is SGD
                    linear.weight.set_method_sgd()
                    if bias:
                        linear.weight_bias.set_method_sgd()

                # activation
                if row['activation'] == 'relu':
                    activation = ReLU(input_variable=in_out, name=name + '_relu')
                    in_out = activation.output_variable
                elif row['activation'] == 'lrelu':
                    activation = LReLU(input_variable=in_out, name=name + '_lrelu')
                    in_out = activation.output_variable
                elif row['activation'] == 'elu':
                    activation = ELU(input_variable=in_out, name=name + '_elu')
                    in_out = activation.output_variable
                elif row['activation'] == 'prelu':
                    activation = PReLU(input_variable=in_out, name=name + '_prelu')
                    in_out = activation.output_variable
                elif row['activation'] == 'sigmoid':
                    activation = Sigmoid(input_variable=in_out, name=name + '_sigmoid')
                    in_out = activation.output_variable
                elif row['activation'] == 'tanh':
                    activation = Tanh(input_variable=in_out, name=name + '_tanh')
                    in_out = activation.output_variable
                elif row['activation'] == 'softmax':
                    activation = Softmax(input_variable=in_out, name=name + '_softmax')
                    in_out = activation.output_variable
                else:
                    activation = None
                if activation:
                    self.linear_list.append(activation)
                linear_cnt += 1
            elif max_flag:
                kernel_shape = [row['kernel depth'], row['IFM channel depth'], row['kernel height'], row['kernel width']]
                stride = row['stride']
                padding = row['padding']
                name = 'MaxPool_'+str(mp_cnt)
                maxPool = MaxPooling(kernel_shape, input_variable=in_out, name=name, stride=stride, padding=padding)
                in_out = maxPool.output_variable
                self.conv_list.append(maxPool)
                mp_cnt += 1
                max_flag = False
            elif average_flag:
                kernel_shape = [row['kernel depth'], row['IFM channel depth'], row['kernel height'],
                                row['kernel width']]
                stride = row['stride']
                padding = row['padding']
                name = 'AvgPool_' + str(ap_cnt)
                avgPool = AvgPooling(kernel_shape, input_variable=in_out, name=name, stride=stride, padding=padding)
                in_out = avgPool.output_variable
                self.conv_list.append(avgPool)
                ap_cnt += 1
                average_flag = False
            else:  # which is a conv layer
                kernel_shape = [row['kernel depth'], row['IFM channel depth'], row['kernel height'],
                                row['kernel width']]
                stride = row['stride']
                padding = row['padding']
                name = 'conv_' + str(conv_cnt)
                if row['bias'] == 1:
                    bias = True
                else:
                    bias = False
                conv = Conv2D(kernel_shape, input_variable=in_out, name=name, hyper_p=hyper_p, stride=stride, padding=padding, bias=bias)
                in_out = conv.output_variable
                self.parameters.append(conv.weight)
                if bias:
                    self.parameters.append(conv.weight_bias)
                self.conv_list.append(conv)

                # optimizer
                if not pd.isna(row['optimizer']) and row['optimizer'].lower() == 'adam':
                    conv.weight.set_method_adam(beta1=hyper_p['beta1'], beta2=hyper_p['beta2'], epsilon=hyper_p['epsilon'])
                    if bias:
                        conv.weight_bias.set_method_adam(beta1=hyper_p['beta1'], beta2=hyper_p['beta2'], epsilon=hyper_p['epsilon'])
                elif not pd.isna(row['optimizer']) and row['optimizer'].lower() == 'nga':
                    conv.weight.set_method_nga(momentum=hyper_p['momentum'])
                    if bias:
                        conv.weight_bias.set_method_nga(momentum=hyper_p['momentum'])
                elif not pd.isna(row['optimizer']) and row['optimizer'].lower() == 'momentum':
                    conv.weight.set_method_momentum(momentum=hyper_p['momentum'])
                    if bias:
                        conv.weight_bias.set_method_momentum(momentum=hyper_p['momentum'])
                else:  # default is SGD
                    conv.weight.set_method_sgd()
                    if bias:
                        conv.weight_bias.set_method_sgd()

                # activation
                if row['activation'] == 'relu':
                    activation = ReLU(input_variable=in_out, name=name+'_relu')
                    in_out = activation.output_variable
                elif row['activation'] == 'lrelu':
                    activation = LReLU(input_variable=in_out, name=name+'_lrelu')
                    in_out = activation.output_variable
                elif row['activation'] == 'elu':
                    activation = ELU(input_variable=in_out, name=name+'_elu')
                    in_out = activation.output_variable
                elif row['activation'] == 'prelu':
                    activation = PReLU(input_variable=in_out, name=name+'_prelu')
                    in_out = activation.output_variable
                elif row['activation'] == 'sigmoid':
                    activation = Sigmoid(input_variable=in_out, name=name+'_sigmoid')
                    in_out = activation.output_variable
                elif row['activation'] == 'tanh':
                    activation = Tanh(input_variable=in_out, name=name+'_tanh')
                    in_out = activation.output_variable
                elif row['activation'] == 'softmax':
                    activation = Softmax(input_variable=in_out, name=name+'_softmax')
                    in_out = activation.output_variable
                else:
                    activation = None
                if activation:
                    self.conv_list.append(activation)

                if row['max pooling'] == 1:
                    max_flag = True
                elif row['avg pooling'] == 1:
                    average_flag = True
                conv_cnt += 1


    def forward_propagation(self, X) -> Variable:
        # assign the real input to the first layer's input_variable.data
        self.conv_list[0].input_variable.data = X
        for conv in self.conv_list:
            conv.forward()
        self.linear_list[0].input_variable.data = self.conv_list[-1].output_variable.data.reshape(self.linear_list[0].input_variable.shape)
        for linear in self.linear_list:
            linear.forward()
        return self.linear_list[-1].output_variable

    def back_propagation(self):
        # assign the real out_diff to the last layer's output_variable.diff
        for linear in reversed(self.linear_list):
            linear.backward()
        self.conv_list[-1].output_variable.diff = self.linear_list[0].input_variable.diff.reshape(self.conv_list[-1].output_variable.shape)
        for conv in reversed(self.conv_list):
            conv.backward()

    def update(self):
        # apply_gradient
        for k in GLOBAL_VARIABLE_SCOPE:
            s = GLOBAL_VARIABLE_SCOPE[k]
            if isinstance(s, Variable) and s.learnable:
                s.update(learning_rate=self.hyper_p['lr'], decay_rate=self.hyper_p['dr'], batch_size=self.batch_size)
            if isinstance(s, Variable):
                s.diff = np.zeros(s.shape)  # reset diff
