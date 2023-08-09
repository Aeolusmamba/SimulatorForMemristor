import pandas as pd
import numpy as np

a = "0010100"
index = [i for i in range(len(a)) if list(a)[i] == "1"]
print(index)
