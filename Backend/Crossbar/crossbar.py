import numpy as np


class Crossbar:

    def __init__(self, size):
        if isinstance(size, list):
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            raise Exception("Size must be int or list of ints!")
        self.conductance = np.zeros(self.size)  # actual crossbars
        self.row_mask = [1 * self.size[0]]
        self.col_mask = [1 * self.size[1]]
        self.row_mask = None  # conductance * row_mask == activated rows
        self.col_mask = None  # conductance * col_mask == activated cols

    def dot(self, row_mask, V):
        self.row_mask = row_mask  # 未被使用
        # print(f"dot conductance shape: {self.conductance.shape}")
        if V.shape[0] != self.conductance.shape[0]:
            raise Exception(f"The shapes of V{V.shape} and conductance{self.conductance.shape} do not match")
        else:
            I_out = np.dot(V, self.conductance)  # automatic broadcasting
            return I_out

    def mul(self, C):
        if C.shape != self.conductance.shape:
            raise Exception(f"The shapes of C({C.shape}) and conductance({self.conductance.shape}) do not match")
        else:
            return C * self.conductance

    def add(self, C):
        if C.shape != self.conductance.shape:
            raise Exception(f"The shapes of C({C.shape}) and conductance({self.conductance.shape}) do not match")
        else:
            return C + self.conductance

    def setRowMask(self, row_mask):
        self.row_mask = row_mask

    def setColMask(self, col_mask):
        self.col_mask = col_mask

    def read_conductance(self):
        return self.conductance

    def write_conductance(self, C):
        if not self.conductance.shape[0] > C.shape[0] and not self.conductance.shape[1] > C.shape[1]:
            raise Exception(f"The shapes of C({C.shape}) and conductance({self.conductance.shape}) do not match")
        else:
            self.conductance = C
            # print(f"conductance.shape: {self.conductance.shape}")

    def read_cell(self, i, j):
        if i > self.conductance.shape[0] or j > self.conductance.shape[1]:
            raise Exception(f"i {i} or j {j} exceed the maximum shape of conductance -- {self.conductance.shape}")
        return self.conductance[i, j]

    def write_cell(self, i, j, c):
        if i > self.conductance.shape[0] or j > self.conductance.shape[1]:
            raise Exception(f"i {i} or j {j} exceed the maximum shape of conductance -- {self.conductance.shape}")
        self.conductance[i, j] = c


