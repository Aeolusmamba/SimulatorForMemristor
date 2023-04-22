import numpy as np
import torch

y = np.array([[.3, .3, .4], [.3, .4, .3], [.1, .2, .7]])
label = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
y_diff = np.zeros(y.shape)
y_diff[label == 1] = -1 * (1 / y[label==1])
print(y_diff)
s = np.array([5, 4, 3])
s = s.reshape(s.size, 1)
print(s.shape)
print(y / s)
print(y ** 2)
