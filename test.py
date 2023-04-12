import numpy as np
import torch

y = np.array([[0, 12, 2], [3,70,5], [3,85,5]])
# print(y.shape)
print(np.argmax(y[0:2, 0:2]))
print(y / 5)
z = np.zeros(y.shape, dtype=float)
print(z)
