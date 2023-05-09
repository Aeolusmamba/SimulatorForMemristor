import numpy as np
import torch
import pandas as pd


a = np.array([[[1,3,4], [1,3,4], [1,3,4]], [[1,1,4], [1,6,4], [1,3,5]]])
print(np.sum(a, axis=(0,2)))
