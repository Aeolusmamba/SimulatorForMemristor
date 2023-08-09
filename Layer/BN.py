from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
from C_Graph.operator import Operator
import numpy as np
import math


class BatchNorm(Operator):
    def __init__(self, input_variable: Variable, name: str):
        self.input_variable = input_variable
        self.batch_size = self.input_variable.shape[0]
        self.output_variable = Variable(self.input_variable.shape, name='output', scope=name, init='None')
        self.gamma = Variable(self.input_variable.shape, name='gamma', scope=name, grad=True,
                              learnable=True, init='None')
        self.gamma.data = np.random.uniform(0.9, 1.1, self.gamma.shape)
        self.beta = Variable(self.input_variable.shape, name='beta', scope=name, grad=True,
                             learnable=True, init='None')
        self.beta.data = np.random.uniform(-0.1, 0.1, self.beta.shape)

        # overall mean
        self.overall_mean = np.zeros(self.input_variable.shape)
        # overall variance
        self.overall_var = np.zeros(self.input_variable.shape)
        self.epsilon = 1e-5
        self.moving_decay = 0.9  # for estimation in a weighted sum
        super(BatchNorm, self).__init__(name, self.input_variable, self.output_variable)

    def forward(self, phase='train'):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval(phase)
            if phase == 'train':
                if self.input_variable.data.ndim == 2:  # linear output
                    self.var = np.var(self.input_variable.data, axis=0)
                    self.mean = np.mean(self.input_variable.data, axis=0)
                elif self.input_variable.data.ndim == 4:  # conv output
                    self.var = np.var(self.input_variable.data, axis=(0, 1, 2))
                    self.mean = np.mean(self.input_variable.data, axis=(0, 1, 2))
                # initialize shadow_variable with mean, var
                if np.sum(self.overall_mean) == 0 and np.sum(self.overall_var) == 0:
                    self.overall_mean = self.mean
                    self.overall_var = self.var
                # update shadow_variable with mean, var, moving_decay
                else:
                    self.overall_mean = self.moving_decay * self.overall_mean + (1 - self.moving_decay) * self.mean
                    self.overall_var = self.moving_decay * self.overall_var + (1 - self.moving_decay) * self.var
                self.normalized_x = (self.input_variable.data - self.mean) / np.sqrt(
                    self.var + self.epsilon)
                self.wait_forward = False
            else:  # test phase
                self.normalized_x = (self.input_variable.data - self.overall_mean) / np.sqrt(
                    self.overall_var + self.epsilon)
            self.output_variable.data = self.gamma.data * self.normalized_x + self.beta.data
            return
        else:
            pass

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            if self.input_variable.data.ndim == 2:
                self.gamma.diff = np.sum(self.output_variable.diff * self.normalized_x, axis=0)
                self.beta.diff = np.sum(self.output_variable.diff, axis=0)
            elif self.input_variable.data.ndim == 4:
                self.gamma.diff = np.sum(self.output_variable.diff * self.normalized_x, axis=(0, 1, 2))
                self.beta.diff = np.sum(self.output_variable.diff, axis=(0, 1, 2))

            d_norm_x = self.output_variable.diff * self.gamma.data
            if self.input_variable.data.ndim == 2:
                dx = self.batch_size * d_norm_x - np.sum(d_norm_x, axis=0) - self.normalized_x * np.sum(
                    d_norm_x * self.normalized_x, axis=0)
            else:
                dx = self.batch_size * d_norm_x - np.sum(d_norm_x, axis=0) - self.normalized_x * np.sum(
                    d_norm_x * self.normalized_x, axis=(0, 1, 2))
            dx *= (1.0 / self.batch_size) / np.sqrt(self.var + self.epsilon)
            self.input_variable.diff = dx
            self.wait_forward = True
            return
